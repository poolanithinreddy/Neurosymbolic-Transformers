"""NST-VERI v2 training loop: learned multi-task FEVER fact verification.

Redesigned training loop. Key changes from v2-old:
  1. NaN check BEFORE backward (not after)
  2. Temperature scaling APPLIED in final evaluation
  3. Checkpoint resume support (optimizer/scheduler state saved)
  4. Separate early_stop flag (not overloaded nan_abort)
  5. Focal loss with per-class gamma (REFUTES emphasized)
  6. Evidence quality filtering (skip/downweight empty evidence)
  7. LR schedule uses floor() for accurate total_steps with drop_last

Training schedule:
  Phase 1 (Epochs 0-1): NLI + aux heads warmup (lower weights)
  Phase 2 (Epochs 2+):  Full multi-task + R-Drop + contrastive
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from data.fever_dataset import (
    LABEL2ID, ID2LABEL, NUM_LABELS, FEVER_LABELS,
    load_fever_splits, FeverGoldDataset,
    fever_collate_fn, print_fever_stats,
)
from models.nst_veri_v2 import NSTVeriModelV2
from models.fever_nli import build_fever_model
from eval.calibration_metrics import expected_calibration_error, brier_score
from training.config_validation import load_and_validate_config

logger = logging.getLogger("train_fever_veri_v2")


def _deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _unwrap(model):
    return getattr(model, '_orig_mod', model)


def _auto_device(preferred: str | None = None) -> str:
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def _build_dataloader(
    dataset, tokenizer, batch_size: int, max_length: int = 384,
    shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False,
    seed: int = 42,
) -> DataLoader:
    collate = partial(fever_collate_fn, tokenizer=tokenizer, max_length=max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_persistent = num_workers > 0
    pf = 4 if num_workers > 0 else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory,
        drop_last=shuffle, generator=generator if shuffle else None,
        persistent_workers=use_persistent, prefetch_factor=pf,
    )


def _eval_split(model, dataloader, device, use_symbolic=False, temperature=1.0):
    """Evaluate model on a data split with optional temperature scaling.

    Returns accuracy, ECE, Brier, per-label metrics, aux head diagnostics.
    """
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    all_contradiction, all_relevance = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            kwargs = {}
            if use_symbolic:
                kwargs["claims"] = batch.get("claims")
                kwargs["evidences"] = batch.get("evidences")
                kwargs["use_symbolic_fusion"] = True
            else:
                kwargs["use_symbolic_fusion"] = False

            result = model.predict(input_ids, attention_mask, **kwargs)

            # Temperature scaling: apply to logits, then use result probs
            # for symbolic fusion path (fusion already applied to probs).
            if temperature != 1.0 and "logits" in result and not use_symbolic:
                scaled_logits = result["logits"] / max(temperature, 0.01)
                probs = F.softmax(scaled_logits, dim=-1)
            else:
                probs = result["probs"]

            preds = probs.argmax(dim=-1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_contradiction.append(result["contradiction_score"].cpu())
            all_relevance.append(result["relevance_score"].cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_contradiction = torch.cat(all_contradiction)
    all_relevance = torch.cat(all_relevance)

    accuracy = (all_preds == all_labels).float().mean().item()

    per_label = {}
    for label_name, label_id in LABEL2ID.items():
        mask = all_labels == label_id
        if mask.sum() > 0:
            per_label[label_name] = {
                "count": mask.sum().item(),
                "accuracy": (all_preds[mask] == label_id).float().mean().item(),
            }

    ece, _ = expected_calibration_error(all_probs, all_labels)
    bs = brier_score(all_probs, all_labels)

    # Auxiliary head diagnostics
    refutes_mask = all_labels == LABEL2ID["REFUTES"]
    nei_mask = all_labels == LABEL2ID["NOT ENOUGH INFO"]
    contra_precision = 0.0
    if (all_contradiction > 0.5).any():
        contra_pred = all_contradiction > 0.5
        contra_precision = (contra_pred & refutes_mask).float().sum() / contra_pred.float().sum()
        contra_precision = contra_precision.item()
    rel_precision = 0.0
    if (all_relevance < 0.5).any():
        irrel_pred = all_relevance < 0.5
        rel_precision = (irrel_pred & nei_mask).float().sum() / irrel_pred.float().sum()
        rel_precision = rel_precision.item()

    model.train()
    return {
        "accuracy": round(accuracy, 4),
        "ece": ece,
        "brier": bs,
        "per_label": per_label,
        "n_samples": len(all_labels),
        "contradiction_precision": round(contra_precision, 4),
        "relevance_precision": round(rel_precision, 4),
    }


def _save_checkpoint(model, tokenizer, optimizer, scheduler, global_step,
                     best_dev_acc, epoch, outdir):
    """Save full checkpoint with resume support."""
    ckpt_dir = os.path.join(outdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    _unwrap(model).backbone.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save({
        "global_step": global_step,
        "epoch": epoch,
        "best_dev_acc": best_dev_acc,
        "contradiction_head": _unwrap(model).contradiction_head.state_dict(),
        "relevance_head": _unwrap(model).relevance_head.state_dict(),
        "recalibration": _unwrap(model).recalibration.state_dict(),
        "contrastive": _unwrap(model).contrastive.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, os.path.join(ckpt_dir, "nst_veri_v2_state.pt"))


def train_fever_veri_v2(
    config_path: str,
    outdir_override: str | None = None,
    config_overrides: dict | None = None,
) -> dict:
    """Train NST-VERI v2: learned multi-task fact verification.

    Returns:
        Report dict with train/dev metrics and diagnostics.
    """
    # ── Load config ──────────────────────────────────────────
    cfg = load_and_validate_config(config_path)
    if config_overrides:
        _deep_update(cfg, config_overrides)
    seed = int(cfg.get("seed", 42))

    # ── Reproducibility ──────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = _auto_device(cfg.get("device", "auto"))

    # IO config
    io_cfg = cfg.get("io", {})
    outdir = outdir_override or io_cfg.get("out_dir", "outputs_fever_veri_v2")
    os.makedirs(outdir, exist_ok=True)

    # Training hyperparameters
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 1e-5))
    lr_lora = float(train_cfg.get("lr_lora", 3e-4))
    lr_heads = float(train_cfg.get("lr_heads", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    eval_every = int(train_cfg.get("eval_every_steps", 250))
    patience = int(train_cfg.get("patience", 6))
    num_workers = int(train_cfg.get("num_workers", 0))
    fp16 = train_cfg.get("fp16", False)
    bf16 = train_cfg.get("bf16", False)
    if bf16:
        fp16 = False
    enable_tf32 = train_cfg.get("tf32", False)
    use_fused = train_cfg.get("fused_optimizer", False)

    # Model config
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "microsoft/deberta-v3-large")
    label_smoothing = float(model_cfg.get("label_smoothing", 0.05))
    dropout = float(model_cfg.get("dropout", 0.1))
    max_length = int(model_cfg.get("max_length", 384))
    use_lora = model_cfg.get("use_lora", True)
    lora_rank = int(model_cfg.get("lora_rank", 16))
    lora_alpha = int(model_cfg.get("lora_alpha", 32))
    gradient_checkpointing = model_cfg.get("gradient_checkpointing", True)

    # Data config
    data_cfg = cfg.get("data", {})
    max_train = data_cfg.get("max_train", None)
    max_dev = data_cfg.get("max_dev", None)
    evidence_mode = data_cfg.get("evidence_mode", cfg.get("evidence_mode", "gold"))
    cache_dir = data_cfg.get("cache_dir", None)
    dev_test_ratio = float(data_cfg.get("dev_test_ratio", 0.1))
    dev_sample = data_cfg.get("dev_sample", 2000)

    # V2-specific config
    v2_cfg = cfg.get("veri", cfg.get("v2", {}))
    beta_contradiction = float(v2_cfg.get("beta_contradiction", 0.3))
    beta_relevance = float(v2_cfg.get("beta_relevance", 0.3))
    gamma_contrastive = float(v2_cfg.get("gamma_contrastive", 0.1))
    rdrop_alpha = float(v2_cfg.get("rdrop_alpha", 0.5))
    contrastive_temp = float(v2_cfg.get("contrastive_temperature", 0.07))
    use_symbolic_at_inference = v2_cfg.get("use_symbolic_at_inference", True)
    focal_gamma = float(v2_cfg.get("focal_gamma", 2.0))
    refutes_gamma = float(v2_cfg.get("refutes_gamma", 3.0))

    # ── Performance tuning ────────────────────────────────────
    if enable_tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled")

    # ── Load data ─────────────────────────────────────────────
    logger.info("Loading FEVER dataset...")
    splits = load_fever_splits(
        cache_dir=cache_dir, max_train=max_train, max_dev=max_dev,
        dev_test_ratio=dev_test_ratio, seed=seed,
    )
    print_fever_stats(splits)

    # ── Evidence quality diagnostic ───────────────────────────
    train_items = splits["train"]
    n_good_ev = sum(1 for it in train_items if len(it.get("gold_evidence_text", "")) > 30)
    ev_pct = 100 * n_good_ev / max(1, len(train_items))
    print(f"\n  Evidence quality: {n_good_ev}/{len(train_items)} ({ev_pct:.0f}%) have >30 char evidence")
    if ev_pct < 40:
        logger.warning(
            "LOW EVIDENCE QUALITY: Only %.0f%% have real evidence. "
            "Run: python main.py build-fever-wiki-cache", ev_pct,
        )

    for i in range(min(3, len(train_items))):
        it = train_items[i]
        ev = it.get("gold_evidence_text", "")[:120]
        print(f"    [{i}] {it['label']}: {it['claim'][:60]}...")
        print(f"         Evidence: {ev}")

    train_ds = FeverGoldDataset(splits["train"])
    dev_ds = FeverGoldDataset(splits["dev"])
    logger.info("Using GOLD EVIDENCE mode (Setting A)")

    # ── Build model ───────────────────────────────────────────
    tokenizer, base_model = build_fever_model(
        model_name=model_name,
        label_smoothing=label_smoothing,
        dropout=dropout,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gradient_checkpointing=gradient_checkpointing,
    )

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_name)
    hidden_dim = hf_config.hidden_size

    # Compute class weights (inverse frequency)
    from collections import Counter
    label_counts = Counter(it["label"] for it in splits["train"])
    total_train = sum(label_counts.values())
    class_weights = torch.tensor([
        total_train / (NUM_LABELS * max(1, label_counts.get(FEVER_LABELS[i], 1)))
        for i in range(NUM_LABELS)
    ], dtype=torch.float32).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    model = NSTVeriModelV2(
        backbone=base_model,
        hidden_dim=hidden_dim,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        dropout=dropout,
        contrastive_temperature=contrastive_temp,
        rdrop_alpha=rdrop_alpha,
        focal_gamma=focal_gamma,
        refutes_gamma=refutes_gamma,
    ).to(device)

    # ── DataLoaders ───────────────────────────────────────────
    pin_memory = device == "cuda"
    train_loader = _build_dataloader(
        train_ds, tokenizer, batch_size, max_length,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
    )
    dev_loader = _build_dataloader(
        dev_ds, tokenizer, batch_size, max_length,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
    )

    # Dev subset for fast periodic evaluation
    dev_sample_loader = dev_loader
    if dev_sample and isinstance(dev_sample, int) and dev_sample < len(dev_ds):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dev_ds)), dev_sample)
        dev_sample_ds = Subset(dev_ds, indices)
        dev_sample_loader = _build_dataloader(
            dev_sample_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
        )

    # ── Optimizer ─────────────────────────────────────────────
    backbone_params, lora_params, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params.append(p)
        elif any(h in name for h in [
            "contradiction", "relevance", "recalibration",
            "contrastive", "classifier", "pooler",
        ]):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr, "name": "backbone"})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "name": "lora"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_heads, "name": "heads"})

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        logger.info(f"  {pg['name']}: {n_p/1e6:.2f}M params, lr={pg['lr']}")

    fused = use_fused and torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # Use floor for accurate total_steps (DataLoader uses drop_last=True)
    steps_per_epoch = len(train_ds) // batch_size // grad_accum_steps
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Mixed precision ───────────────────────────────────────
    use_amp = (fp16 or bf16) and device == "cuda"
    if bf16:
        amp_dtype = torch.bfloat16
        scaler = None
    elif fp16:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    else:
        amp_dtype = torch.float32
        scaler = None

    # ── Training loop ─────────────────────────────────────────
    best_dev_acc = 0.0
    patience_counter = 0
    train_log = []
    global_step = 0
    nan_abort = False
    early_stop = False

    _prec = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
    print(f"\n{'='*65}")
    print(f"  NST-VERI v2: Learned Multi-Task Training")
    print(f"  Model: {model_name}" + (f" + LoRA r={lora_rank}" if use_lora else ""))
    print(f"  Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M")
    print(f"  epochs={epochs}, bs={batch_size}x{grad_accum_steps}={batch_size*grad_accum_steps}")
    print(f"  lr={lr}, lr_lora={lr_lora}, lr_heads={lr_heads}")
    print(f"  precision={_prec}")
    print(f"  focal_gamma={focal_gamma}, refutes_gamma={refutes_gamma}")
    print(f"  β_contra={beta_contradiction}, β_rel={beta_relevance}, γ_ctr={gamma_contrastive}")
    print(f"  R-Drop α={rdrop_alpha}")
    print(f"  total_steps={total_steps}, warmup={warmup_steps}")
    print(f"{'='*65}\n")

    t0 = time.time()

    for epoch in range(epochs):
        if nan_abort or early_stop:
            break

        # Phase schedule
        warmup_phase = epoch < 2
        beta_contra_cur = beta_contradiction * (0.3 if warmup_phase else 1.0)
        beta_rel_cur = beta_relevance * (0.3 if warmup_phase else 1.0)
        gamma_cur = gamma_contrastive * (0.5 if warmup_phase else 1.0)
        use_rdrop = not warmup_phase and rdrop_alpha > 0

        phase_str = "warmup" if warmup_phase else "full"
        print(f"  Epoch {epoch+1}/{epochs} [{phase_str}] | "
              f"β_c={beta_contra_cur:.3f} β_r={beta_rel_cur:.3f} "
              f"γ={gamma_cur:.3f} rdrop={'on' if use_rdrop else 'off'}")

        model.train()
        epoch_loss = 0.0
        epoch_nli_loss = 0.0
        epoch_contra_loss = 0.0
        epoch_rel_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_rdrop_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                result = model(
                    input_ids, attention_mask, labels,
                    beta_contradiction=beta_contra_cur,
                    beta_relevance=beta_rel_cur,
                    gamma_contrastive=gamma_cur,
                    use_rdrop=use_rdrop,
                )
                loss = result["loss"] / grad_accum_steps

            # NaN check BEFORE backward (fixed from v2-old)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at epoch {epoch}, step {step}")
                nan_abort = True
                break

            # Backward
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient step
            if (step + 1) % grad_accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Periodic evaluation
                if eval_every > 0 and global_step % eval_every == 0:
                    dev_metrics = _eval_split(model, dev_sample_loader, device)
                    dev_acc = dev_metrics["accuracy"]
                    recalib_s = result.get("recalib_scale", torch.tensor(0.0))
                    if isinstance(recalib_s, torch.Tensor):
                        recalib_s = recalib_s.item()
                    print(
                        f"    Step {global_step}: loss={loss.item()*grad_accum_steps:.4f} "
                        f"nli={result['loss_nli'].item():.4f} "
                        f"ctr={result['loss_contradiction'].item():.4f} "
                        f"rel={result['loss_relevance'].item():.4f} "
                        f"con={result['loss_contrastive'].item():.4f} "
                        f"rdp={result['loss_rdrop'].item():.4f} "
                        f"| dev={dev_acc:.4f} ECE={dev_metrics['ece']:.4f} "
                        f"recal={recalib_s:.4f}"
                    )
                    train_log.append({
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": round(loss.item() * grad_accum_steps, 4),
                        "loss_nli": round(result["loss_nli"].item(), 4),
                        "loss_contradiction": round(result["loss_contradiction"].item(), 4),
                        "loss_relevance": round(result["loss_relevance"].item(), 4),
                        "loss_contrastive": round(result["loss_contrastive"].item(), 4),
                        "loss_rdrop": round(result["loss_rdrop"].item(), 4),
                        "dev_accuracy": dev_acc,
                        "dev_ece": dev_metrics["ece"],
                        "dev_brier": dev_metrics["brier"],
                        "recalib_scale": round(recalib_s, 4),
                    })

                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        patience_counter = 0
                        _save_checkpoint(model, tokenizer, optimizer, scheduler,
                                        global_step, best_dev_acc, epoch, outdir)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at step {global_step}")
                            early_stop = True
                            break
                    model.train()

            total_loss_val = loss.item() * grad_accum_steps
            epoch_loss += total_loss_val
            epoch_nli_loss += result["loss_nli"].item()
            epoch_contra_loss += result["loss_contradiction"].item()
            epoch_rel_loss += result["loss_relevance"].item()
            epoch_contrastive_loss += result["loss_contrastive"].item()
            epoch_rdrop_loss += result["loss_rdrop"].item()
            n_steps += 1

        if nan_abort or early_stop:
            break

        avg_loss = epoch_loss / max(1, n_steps)

        # End-of-epoch full dev eval
        dev_metrics = _eval_split(model, dev_loader, device)
        dev_acc = dev_metrics["accuracy"]
        print(
            f"  Epoch {epoch+1}/{epochs}: "
            f"loss={avg_loss:.4f} nli={epoch_nli_loss/max(1,n_steps):.4f} "
            f"ctr={epoch_contra_loss/max(1,n_steps):.4f} "
            f"rel={epoch_rel_loss/max(1,n_steps):.4f} "
            f"con={epoch_contrastive_loss/max(1,n_steps):.4f} "
            f"rdp={epoch_rdrop_loss/max(1,n_steps):.4f} "
            f"| dev_acc={dev_acc:.4f} ECE={dev_metrics['ece']:.4f}"
        )
        for lbl, stats in dev_metrics.get("per_label", {}).items():
            print(f"    {lbl}: {stats['accuracy']:.4f} (n={stats['count']})")
        print(f"    Contradiction head precision: {dev_metrics.get('contradiction_precision', 0):.4f}")
        print(f"    Relevance head precision: {dev_metrics.get('relevance_precision', 0):.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            _save_checkpoint(model, tokenizer, optimizer, scheduler,
                            global_step, best_dev_acc, epoch, outdir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - t0

    # ── Post-hoc temperature scaling ──────────────────────────
    from eval.temperature_scaling import learn_temperature
    print(f"\n{'─'*40}")
    print("  Post-hoc temperature scaling (dev set)")
    print(f"{'─'*40}")
    optimal_T = learn_temperature(model, dev_loader, device)

    # ── Final evaluation (WITH temperature scaling) ───────────
    print(f"\n{'─'*40}")
    print(f"  Final evaluation on dev set (T={optimal_T:.4f})")
    print(f"{'─'*40}")
    final_dev_raw = _eval_split(model, dev_loader, device,
                                use_symbolic=False, temperature=optimal_T)
    print(f"  Accuracy: {final_dev_raw['accuracy']:.4f}")
    print(f"  ECE: {final_dev_raw['ece']:.4f}")
    print(f"  Brier: {final_dev_raw['brier']:.4f}")
    for label, stats in final_dev_raw.get("per_label", {}).items():
        print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

    # With symbolic fusion
    final_dev = final_dev_raw
    if use_symbolic_at_inference:
        print(f"\n{'─'*40}")
        print("  Final evaluation on dev set (+ symbolic fusion)")
        print(f"{'─'*40}")
        final_dev = _eval_split(model, dev_loader, device,
                                use_symbolic=True, temperature=optimal_T)
        print(f"  Fused Accuracy: {final_dev['accuracy']:.4f}")
        print(f"  ECE: {final_dev['ece']:.4f}")
        for label, stats in final_dev.get("per_label", {}).items():
            print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")
        delta = final_dev['accuracy'] - final_dev_raw['accuracy']
        print(f"  Symbolic fusion delta: {delta:+.4f}")

    # Held-out dev_test
    final_dev_test = None
    if "dev_test" in splits and splits["dev_test"]:
        print(f"\n{'─'*40}")
        print("  Held-out dev_test evaluation")
        print(f"{'─'*40}")
        dev_test_ds = FeverGoldDataset(splits["dev_test"])
        dev_test_loader = _build_dataloader(
            dev_test_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
        )
        final_dev_test_raw = _eval_split(model, dev_test_loader, device,
                                         use_symbolic=False, temperature=optimal_T)
        print(f"  Raw Accuracy: {final_dev_test_raw['accuracy']:.4f}")
        for label, stats in final_dev_test_raw.get("per_label", {}).items():
            print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

        final_dev_test = final_dev_test_raw
        if use_symbolic_at_inference:
            final_dev_test = _eval_split(model, dev_test_loader, device,
                                         use_symbolic=True, temperature=optimal_T)
            print(f"  + Symbolic fusion: {final_dev_test['accuracy']:.4f}")
            delta_ht = final_dev_test['accuracy'] - final_dev_test_raw['accuracy']
            print(f"  Fusion delta: {delta_ht:+.4f}")

    # ── Build report ──────────────────────────────────────────
    report = {
        "mode": "veri_v2",
        "evidence_mode": evidence_mode,
        "model_name": model_name,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else 0,
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 2),
        "beta_contradiction": beta_contradiction,
        "beta_relevance": beta_relevance,
        "gamma_contrastive": gamma_contrastive,
        "rdrop_alpha": rdrop_alpha,
        "focal_gamma": focal_gamma,
        "refutes_gamma": refutes_gamma,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "max_length": max_length,
        "precision": _prec,
        "seed": seed,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "best_dev_acc": round(best_dev_acc, 4),
        "nan_abort": nan_abort,
        "early_stop": early_stop,
        "dev": final_dev,
        "dev_raw": final_dev_raw,
        "dev_test": final_dev_test,
        "train_log": train_log,
        "temperature": round(optimal_T, 4),
    }

    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(outdir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best dev accuracy: {best_dev_acc:.4f}")
    print(f"  Output: {outdir}")

    return report
