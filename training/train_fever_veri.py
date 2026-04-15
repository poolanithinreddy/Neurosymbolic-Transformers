"""NST-VERI training loop: multi-phase verification-enhanced FEVER training.

This is the flagship training loop for the Neurosymbolic Transformers project.
It implements the full NST-VERI pipeline with 3-phase training:

  Phase 1 (warm-up):     Pure NLI + auxiliary verification heads
  Phase 2 (contrastive): + supervised contrastive loss on high-confidence examples
  Phase 3 (constrained): + adaptive constraint loss with curriculum warmup

Key innovations:
  - Verification heads bridge neural and symbolic reasoning
  - Adaptive per-sample lambda learns when to trust constraints
  - Focal loss option for hard-example mining (REFUTES/NEI boundary)
  - Curriculum constraint scheduling prevents early interference
  - Post-hoc temperature scaling for calibration

This module is separate from train_fever_nst.py to keep the simpler modes
clean while giving NST-VERI the dedicated complexity it needs.
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
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from data.fever_dataset import (
    LABEL2ID, ID2LABEL, NUM_LABELS, FEVER_LABELS,
    load_fever_splits, FeverGoldDataset, FeverPipelineDataset,
    fever_collate_fn, print_fever_stats,
)
from models.nst_veri import NSTVeriModel
from models.fever_nli import build_fever_model
from symbolic.constraints_v2 import ConstraintEngineV2
from training.adaptive_lambda import AdaptiveLambdaModule
from eval.calibration_metrics import expected_calibration_error, brier_score
from training.config_validation import load_and_validate_config

logger = logging.getLogger("train_fever_veri")


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in FEVER.

    Reduces loss for well-classified examples, focusing training
    on hard REFUTES/NEI boundary cases.

    L_focal = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t is the probability of the correct class.

    Args:
        alpha: class weights (tensor of size num_classes).
        gamma: focusing parameter (0 = standard CE, 2 = strong focusing).
        label_smoothing: optional label smoothing.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        if self.label_smoothing > 0:
            n_classes = logits.size(-1)
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + \
                              self.label_smoothing / n_classes

        # p_t: probability of the ground truth class
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        # Cross entropy
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(targets_one_hot * log_probs).sum(dim=-1)

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce
        return loss.mean()


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


def _eval_split(model, dataloader, device):
    """Evaluate model on a split. Returns accuracy, ECE, Brier, per-label metrics."""
    model.eval()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            result = model.predict(input_ids, attention_mask)
            probs = result["probs"]
            preds = probs.argmax(dim=-1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

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

    model.train()
    return {
        "accuracy": round(accuracy, 4),
        "ece": ece,
        "brier": bs,
        "per_label": per_label,
        "n_samples": len(all_labels),
    }


def _get_phase(epoch: int, total_epochs: int) -> int:
    """Determine training phase from epoch number.

    Phase 1: first 20% of epochs (NLI + aux)
    Phase 2: next 20% (+ contrastive)
    Phase 3: remaining 60% (+ constraints with warmup)
    """
    if total_epochs <= 2:
        # Short training: simplified phases
        if epoch == 0:
            return 1
        return 3
    frac = epoch / total_epochs
    if frac < 0.2:
        return 1
    elif frac < 0.4:
        return 2
    return 3


def _constraint_warmup(epoch: int, total_epochs: int) -> float:
    """Constraint weight warmup schedule for Phase 3.

    Returns multiplier in [0, 1] that ramps up constraint influence.
    This prevents early constraint interference when the model
    hasn't learned basic NLI yet.
    """
    phase = _get_phase(epoch, total_epochs)
    if phase < 3:
        return 0.0
    # Linear warmup within phase 3
    phase3_start = int(0.4 * total_epochs)
    if total_epochs <= 2:
        phase3_start = 1
    progress = (epoch - phase3_start) / max(1, total_epochs - phase3_start)
    return min(1.0, max(0.0, progress))


def train_fever_veri(
    config_path: str,
    outdir_override: str | None = None,
    config_overrides: dict | None = None,
) -> dict:
    """Train NST-VERI: the flagship verification-enhanced FEVER model.

    This implements the full 3-phase training protocol:
      Phase 1: NLI head + auxiliary verification heads
      Phase 2: + contrastive prototype loss
      Phase 3: + adaptive constraint loss with curriculum warmup

    Returns:
        Report dict with train/dev metrics and detailed diagnostics.
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
    outdir = outdir_override or io_cfg.get("out_dir", "outputs_fever_veri")
    os.makedirs(outdir, exist_ok=True)

    # Training hyperparameters
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 1e-5))
    lr_lora = float(train_cfg.get("lr_lora", 3e-4))
    lr_heads = float(train_cfg.get("lr_heads", 5e-4))
    lr_gate = float(train_cfg.get("lr_gate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    eval_every = int(train_cfg.get("eval_every_steps", 250))
    patience = int(train_cfg.get("patience", 6))
    num_workers = int(train_cfg.get("num_workers", 0))
    fp16 = train_cfg.get("fp16", False)
    bf16 = train_cfg.get("bf16", device == "cuda")
    if bf16:
        fp16 = False
    use_compile = train_cfg.get("torch_compile", False)
    enable_tf32 = train_cfg.get("tf32", False)
    use_fused = train_cfg.get("fused_optimizer", False)
    use_focal = train_cfg.get("focal_loss", False)
    focal_gamma = float(train_cfg.get("focal_gamma", 2.0))

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

    # VERI-specific config
    veri_cfg = cfg.get("veri", {})
    n_constraints = int(veri_cfg.get("n_constraints", 6))
    lambda_max = float(veri_cfg.get("lambda_max", 0.3))
    contrastive_temp = float(veri_cfg.get("contrastive_temperature", 0.07))
    beta = float(veri_cfg.get("beta_aux", 1.0))
    gamma = float(veri_cfg.get("gamma_contrastive", 0.1))

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

    if evidence_mode == "gold":
        train_ds = FeverGoldDataset(splits["train"])
        dev_ds = FeverGoldDataset(splits["dev"])
        logger.info("Using GOLD EVIDENCE mode (Setting A)")
    else:
        raise ValueError("NST-VERI currently requires gold evidence mode")

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

    # Determine hidden dim from model config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_name)
    hidden_dim = hf_config.hidden_size

    # Compute class weights
    from collections import Counter
    label_counts = Counter(it["label"] for it in splits["train"])
    total_train = sum(label_counts.values())
    class_weights = torch.tensor([
        total_train / (NUM_LABELS * max(1, label_counts.get(FEVER_LABELS[i], 1)))
        for i in range(NUM_LABELS)
    ], dtype=torch.float32).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Build NST-VERI model
    model = NSTVeriModel(
        backbone=base_model,
        hidden_dim=hidden_dim,
        n_constraints=n_constraints,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        dropout=dropout,
        contrastive_temperature=contrastive_temp,
    ).to(device)

    # Build adaptive lambda module
    adaptive_lambda_mod = AdaptiveLambdaModule(
        n_constraints=n_constraints,
        lambda_max=lambda_max,
    ).to(device)

    # Build constraint engine
    constraint_engine = ConstraintEngineV2()

    # Optional: focal loss
    focal_loss_fn = None
    if use_focal:
        focal_loss_fn = FocalLoss(
            alpha=class_weights, gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        logger.info(f"Focal loss enabled: gamma={focal_gamma}")

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
        from torch.utils.data import Subset
        rng = random.Random(seed)
        indices = rng.sample(range(len(dev_ds)), dev_sample)
        dev_sample_ds = Subset(dev_ds, indices)
        dev_sample_loader = _build_dataloader(
            dev_sample_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
        )

    # ── torch.compile ─────────────────────────────────────────
    _compiled = False
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            _compiled = True
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # ── Optimizer ─────────────────────────────────────────────
    backbone_params, lora_params, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params.append(p)
        elif any(h in name for h in ["verification", "contrastive", "residual"]):
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
    # Adaptive lambda module
    param_groups.append({
        "params": list(adaptive_lambda_mod.parameters()),
        "lr": lr_gate, "name": "adaptive_lambda",
    })

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    al_params = sum(p.numel() for p in adaptive_lambda_mod.parameters())

    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        logger.info(f"  {pg['name']}: {n_p/1e6:.2f}M params, lr={pg['lr']}")

    fused = use_fused and torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    total_steps = epochs * math.ceil(len(train_ds) / batch_size / grad_accum_steps)
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

    # ── Training ──────────────────────────────────────────────
    best_dev_acc = 0.0
    patience_counter = 0
    train_log = []
    global_step = 0
    nan_abort = False

    _prec = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
    print(f"\n{'='*65}")
    print(f"  NST-VERI Training: Verification-Enhanced FEVER")
    print(f"  Model: {model_name} + LoRA r={lora_rank}" if use_lora else f"  Model: {model_name}")
    print(f"  Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M")
    print(f"  Adaptive lambda: {al_params} params, lambda_max={lambda_max}")
    print(f"  epochs={epochs}, bs={batch_size}x{grad_accum_steps}={batch_size*grad_accum_steps}")
    print(f"  lr={lr}, lr_lora={lr_lora}, lr_heads={lr_heads}")
    print(f"  precision={_prec}, compile={_compiled}")
    print(f"  Focal loss: {'yes (γ='+str(focal_gamma)+')' if use_focal else 'no'}")
    print(f"  Phases: 1→NLI+aux, 2→+contrastive, 3→+constraints")
    print(f"  total_steps={total_steps}, warmup={warmup_steps}")
    print(f"{'='*65}\n")

    t0 = time.time()

    for epoch in range(epochs):
        if nan_abort:
            break

        phase = _get_phase(epoch, epochs)
        sched_mult = _constraint_warmup(epoch, epochs)

        # Phase-dependent loss weights
        beta_cur = beta if phase >= 1 else 0.0
        gamma_cur = gamma if phase >= 2 else 0.0

        print(f"  Phase {phase} | Epoch {epoch+1}/{epochs} | "
              f"β={beta_cur:.2f} γ={gamma_cur:.2f} sched={sched_mult:.2f}")

        model.train()
        adaptive_lambda_mod.train()
        epoch_loss = 0.0
        epoch_nli_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_constraint_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            claims = batch["claims"]
            evidences = batch["evidences"]

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                # Extract constraint signals (CPU, no grad needed)
                with torch.no_grad():
                    constraint_signals = constraint_engine.evaluate_batch(claims, evidences)

                # Adaptive lambda computation (differentiable)
                fires = constraint_signals["fires"].to(device).float()
                confidence = constraint_signals["confidence"].to(device)

                # Forward pass through NST-VERI
                result = model(
                    input_ids, attention_mask, labels,
                    constraint_signals=constraint_signals,
                    phase=phase,
                    beta=beta_cur,
                    gamma=gamma_cur,
                )

                # Compute adaptive lambda if in phase 3
                adaptive_out = None
                if phase >= 3 and sched_mult > 0:
                    probs = result["probs"].detach()  # Detach to prevent double backprop
                    adaptive_out = adaptive_lambda_mod(
                        fires, confidence, probs,
                        schedule_multiplier=sched_mult,
                    )
                    # Recompute forward with constraint loss
                    result = model(
                        input_ids, attention_mask, labels,
                        constraint_signals=constraint_signals,
                        phase=phase,
                        beta=beta_cur,
                        gamma=gamma_cur,
                        adaptive_lambda=adaptive_out,
                    )

                # Use focal loss if enabled (replaces the NLI loss component)
                if use_focal and focal_loss_fn is not None:
                    loss_focal = focal_loss_fn(result["logits"], labels)
                    loss = loss_focal + beta_cur * result.get("loss_aux", 0.0) + \
                           gamma_cur * result.get("loss_contrastive", 0.0) + \
                           result.get("loss_constraint", 0.0)
                else:
                    loss = result["loss"]

                loss = loss / grad_accum_steps

            # Backward
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at epoch {epoch}, step {step}")
                nan_abort = True
                break

            # Gradient step
            if (step + 1) % grad_accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(adaptive_lambda_mod.parameters()),
                        max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(adaptive_lambda_mod.parameters()),
                        max_grad_norm,
                    )
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Periodic evaluation
                if eval_every > 0 and global_step % eval_every == 0:
                    dev_metrics = _eval_split(model, dev_sample_loader, device)
                    dev_acc = dev_metrics["accuracy"]
                    res_scale = result.get("residual_scale", 0.0)
                    if isinstance(res_scale, torch.Tensor):
                        res_scale = res_scale.item()
                    print(
                        f"    Step {global_step}: loss={loss.item()*grad_accum_steps:.4f} "
                        f"nli={result.get('loss_nli', torch.tensor(0.0)).item():.4f} "
                        f"aux={result.get('loss_aux', torch.tensor(0.0)).item():.4f} "
                        f"con={result.get('loss_contrastive', torch.tensor(0.0)).item():.4f} "
                        f"cst={result.get('loss_constraint', torch.tensor(0.0)).item():.4f} "
                        f"| dev_acc={dev_acc:.4f} ECE={dev_metrics['ece']:.4f} "
                        f"res_scale={res_scale:.4f}"
                    )
                    train_log.append({
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "phase": phase,
                        "train_loss": round(loss.item() * grad_accum_steps, 4),
                        "loss_nli": round(result.get("loss_nli", torch.tensor(0.0)).item(), 4),
                        "loss_aux": round(result.get("loss_aux", torch.tensor(0.0)).item(), 4),
                        "loss_contrastive": round(result.get("loss_contrastive", torch.tensor(0.0)).item(), 4),
                        "loss_constraint": round(result.get("loss_constraint", torch.tensor(0.0)).item(), 4),
                        "dev_accuracy": dev_acc,
                        "dev_ece": dev_metrics["ece"],
                        "dev_brier": dev_metrics["brier"],
                        "residual_scale": round(res_scale, 4),
                        "schedule_multiplier": sched_mult,
                    })

                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        patience_counter = 0
                        ckpt_dir = os.path.join(outdir, "ckpt")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        _unwrap(model).backbone.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        torch.save({
                            "global_step": global_step,
                            "best_dev_acc": best_dev_acc,
                            "phase": phase,
                            "verification_heads": _unwrap(model).verification.state_dict(),
                            "contrastive_head": _unwrap(model).contrastive.state_dict(),
                            "adaptive_lambda": adaptive_lambda_mod.state_dict(),
                        }, os.path.join(ckpt_dir, "nst_veri_state.pt"))
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at step {global_step}")
                            nan_abort = True
                            break
                    model.train()
                    adaptive_lambda_mod.train()

            total_loss_val = loss.item() * grad_accum_steps
            epoch_loss += total_loss_val
            _nli = result.get("loss_nli", torch.tensor(0.0))
            epoch_nli_loss += _nli.item() if isinstance(_nli, torch.Tensor) else _nli
            _aux = result.get("loss_aux", torch.tensor(0.0))
            epoch_aux_loss += _aux.item() if isinstance(_aux, torch.Tensor) else _aux
            _con = result.get("loss_contrastive", torch.tensor(0.0))
            epoch_contrastive_loss += _con.item() if isinstance(_con, torch.Tensor) else _con
            _cst = result.get("loss_constraint", torch.tensor(0.0))
            epoch_constraint_loss += _cst.item() if isinstance(_cst, torch.Tensor) else _cst
            n_steps += 1

        if nan_abort:
            break

        avg_loss = epoch_loss / max(1, n_steps)
        # End-of-epoch evaluation on full dev
        dev_metrics = _eval_split(model, dev_loader, device)
        dev_acc = dev_metrics["accuracy"]
        print(
            f"  Epoch {epoch+1}/{epochs} (phase {phase}): "
            f"loss={avg_loss:.4f} nli={epoch_nli_loss/max(1,n_steps):.4f} "
            f"aux={epoch_aux_loss/max(1,n_steps):.4f} "
            f"con={epoch_contrastive_loss/max(1,n_steps):.4f} "
            f"cst={epoch_constraint_loss/max(1,n_steps):.4f} "
            f"| dev_acc={dev_acc:.4f} ECE={dev_metrics['ece']:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            ckpt_dir = os.path.join(outdir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            _unwrap(model).backbone.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save({
                "epoch": epoch, "best_dev_acc": best_dev_acc, "phase": phase,
                "verification_heads": _unwrap(model).verification.state_dict(),
                "contrastive_head": _unwrap(model).contrastive.state_dict(),
                "adaptive_lambda": adaptive_lambda_mod.state_dict(),
            }, os.path.join(ckpt_dir, "nst_veri_state.pt"))
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

    # ── Final evaluation ──────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Final evaluation on dev set")
    print(f"{'─'*40}")
    final_dev = _eval_split(model, dev_loader, device)
    print(f"  Label Accuracy ({evidence_mode.upper()} evidence): {final_dev['accuracy']:.4f}")
    print(f"  ECE: {final_dev['ece']:.4f}")
    print(f"  Brier: {final_dev['brier']:.4f}")
    for label, stats in final_dev.get("per_label", {}).items():
        print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

    # Held-out dev_test
    final_dev_test = None
    if "dev_test" in splits and splits["dev_test"]:
        print(f"\n{'─'*40}")
        print("  Final evaluation on held-out dev_test")
        print(f"{'─'*40}")
        dev_test_ds = FeverGoldDataset(splits["dev_test"])
        dev_test_loader = _build_dataloader(
            dev_test_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
        )
        final_dev_test = _eval_split(model, dev_test_loader, device)
        print(f"  Label Accuracy: {final_dev_test['accuracy']:.4f}")
        print(f"  ECE: {final_dev_test['ece']:.4f}")
        for label, stats in final_dev_test.get("per_label", {}).items():
            print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

    # ── Build report ──────────────────────────────────────────
    report = {
        "mode": "veri",
        "evidence_mode": evidence_mode,
        "model_name": model_name,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else 0,
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 2),
        "adaptive_lambda_params": al_params,
        "n_constraints": n_constraints,
        "lambda_max": lambda_max,
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
        "focal_loss": use_focal,
        "dev": final_dev,
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
