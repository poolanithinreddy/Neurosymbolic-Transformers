"""NST v3 training loop: Focal loss + R-Drop + symbolic inference boost.

Key design:
  - Same architecture as neural baseline (DeBERTa + LoRA + linear head)
  - Focal loss with per-class gamma (REFUTES γ=3.0, others γ=1.5)
  - R-Drop consistency regularization (two forward passes, symmetric KL)
  - Class weights (inverse frequency)
  - Symbolic contradiction detection at inference time (rule-based logit boost)
  - NO auxiliary heads, NO RecalibrationNetwork, NO multi-task overhead
  - NO temperature scaling at final eval (fair comparison with baseline)

Why this works when v2 didn't:
  - v2 added 5 loss terms and 3 extra heads → gradient noise killed REFUTES
  - v3 keeps one loss (focal CE), one regularizer (R-Drop), one model
  - Symbolic reasoning is at inference only → can't hurt training
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
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
from models.fever_nli import build_fever_model, FeverNLIWrapper
from eval.calibration_metrics import expected_calibration_error, brier_score
from training.config_validation import load_and_validate_config

logger = logging.getLogger("train_fever_v3")

# ── Focal Loss ────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    class_gamma: dict[int, float] | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal cross-entropy loss with optional per-class gamma.

    Args:
        logits: [B, C] raw logits
        labels: [B] integer labels
        weight: [C] class weights (inverse frequency)
        gamma: default focal gamma
        class_gamma: {class_id: gamma} for per-class gamma
        label_smoothing: standard label smoothing
    """
    ce = F.cross_entropy(logits, labels, weight=weight,
                         label_smoothing=label_smoothing, reduction='none')
    pt = torch.exp(-ce)

    if class_gamma is not None:
        g = torch.full_like(ce, gamma)
        for cls_id, cls_g in class_gamma.items():
            g[labels == cls_id] = cls_g
        focal_weight = (1 - pt) ** g
    else:
        focal_weight = (1 - pt) ** gamma

    return (focal_weight * ce).mean()


# ── R-Drop ────────────────────────────────────────────────────

def rdrop_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two forward pass logits."""
    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    p_soft = F.softmax(logits1.detach(), dim=-1)
    q_soft = F.softmax(logits2.detach(), dim=-1)
    kl_pq = F.kl_div(q, p_soft, reduction='batchmean')
    kl_qp = F.kl_div(p, q_soft, reduction='batchmean')
    return 0.5 * (kl_pq + kl_qp)


# ── Symbolic Contradiction Detection ─────────────────────────

_NEGATION_WORDS = frozenset({
    "not", "never", "no", "neither", "nor", "nothing", "nobody", "nowhere",
    "n't", "didn't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "won't", "wouldn't", "can't", "couldn't", "shouldn't", "isn't", "aren't",
    "don't", "doesn't", "none", "without", "fail", "failed", "lacks",
    "unable", "impossible", "deny", "denied", "denies", "reject", "rejected",
})

_NUMBER_RE = re.compile(r'\b\d+(?:,\d+)*(?:\.\d+)?\b')


def contradiction_score(claim: str, evidence: str) -> float:
    """Rule-based contradiction score between claim and evidence.

    Returns a float in [0, 1] indicating likelihood of contradiction.
    Higher = more likely contradictory.
    """
    c_lower = claim.lower()
    e_lower = evidence.lower()
    c_tokens = set(c_lower.split())
    e_tokens = set(e_lower.split())

    score = 0.0

    # 1. Asymmetric negation: one side has negation, other doesn't
    c_neg = c_tokens & _NEGATION_WORDS
    e_neg = e_tokens & _NEGATION_WORDS
    if bool(c_neg) != bool(e_neg):
        score += 0.4
    # Both have negation but different ones → potential double negation
    elif c_neg and e_neg and c_neg != e_neg:
        score += 0.2

    # 2. Number mismatch (same topic, different numbers)
    c_nums = set(_NUMBER_RE.findall(claim))
    e_nums = set(_NUMBER_RE.findall(evidence))
    if c_nums and e_nums and not c_nums.intersection(e_nums):
        score += 0.35

    # 3. Antonym-like indicators
    antonym_pairs = [
        ("increase", "decrease"), ("more", "less"), ("more", "fewer"),
        ("before", "after"), ("first", "last"), ("true", "false"),
        ("win", "lose"), ("won", "lost"), ("alive", "dead"),
        ("success", "failure"), ("positive", "negative"),
        ("include", "exclude"), ("accept", "reject"),
        ("start", "end"), ("begin", "finish"), ("open", "close"),
        ("male", "female"), ("man", "woman"),
    ]
    for w1, w2 in antonym_pairs:
        if (w1 in c_tokens and w2 in e_tokens) or (w2 in c_tokens and w1 in e_tokens):
            score += 0.25
            break  # Only count once

    return min(score, 1.0)


def apply_symbolic_boost(
    logits: torch.Tensor,
    claims: list[str],
    evidences: list[str],
    boost_strength: float = 1.0,
    min_refutes_prob: float = 0.15,
    max_refutes_prob: float = 0.60,
) -> torch.Tensor:
    """Apply symbolic contradiction boost to REFUTES logit.

    Only boosts when:
    1. Contradiction signals are detected (score > 0.3)
    2. Model is uncertain about REFUTES (prob in [min, max] range)

    This prevents overriding confident predictions while helping uncertain ones.
    """
    refutes_id = LABEL2ID["REFUTES"]
    probs = F.softmax(logits, dim=-1)
    boosted = logits.clone()

    for i, (claim, evidence) in enumerate(zip(claims, evidences)):
        cscore = contradiction_score(claim, evidence)
        refutes_prob = probs[i, refutes_id].item()

        if cscore > 0.3 and min_refutes_prob < refutes_prob < max_refutes_prob:
            boosted[i, refutes_id] += boost_strength * cscore

    return boosted


# ── Helpers ───────────────────────────────────────────────────

def _auto_device(preferred: str | None = None) -> str:
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def _build_dataloader(
    dataset, tokenizer, batch_size, max_length=384,
    shuffle=True, num_workers=0, pin_memory=False, seed=42,
) -> DataLoader:
    collate = partial(fever_collate_fn, tokenizer=tokenizer, max_length=max_length)
    gen = torch.Generator()
    gen.manual_seed(seed)
    use_persistent = num_workers > 0
    pf = 4 if num_workers > 0 else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory,
        drop_last=shuffle, generator=gen if shuffle else None,
        persistent_workers=use_persistent, prefetch_factor=pf,
    )


def _eval_split(
    model: FeverNLIWrapper,
    dataloader: DataLoader,
    device: str,
    symbolic_boost: bool = False,
    boost_strength: float = 1.0,
) -> dict:
    """Evaluate model on a split. NO temperature scaling."""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, attention_mask)
            logits = out["logits"]

            if symbolic_boost and "claims" in batch and "evidences" in batch:
                logits = apply_symbolic_boost(
                    logits, batch["claims"], batch["evidences"],
                    boost_strength=boost_strength,
                )

            probs = F.softmax(logits, dim=-1)
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


def _unwrap(model):
    return getattr(model, '_orig_mod', model)


# ── Main Training Function ──────────────────────────────────

def train_fever_v3(
    config_path: str,
    outdir_override: str | None = None,
) -> dict:
    """Train FEVER NLI with focal loss + R-Drop + symbolic inference boost.

    Same architecture as neural baseline. Only changes:
    1. Focal loss (per-class gamma, emphasizing REFUTES)
    2. R-Drop consistency regularization
    3. Symbolic logit boost at inference time
    """
    # ── Load config ──────────────────────────────────────────
    cfg = load_and_validate_config(config_path)
    seed = int(cfg.get("seed", 42))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = _auto_device(cfg.get("device", "auto"))

    io_cfg = cfg.get("io", {})
    outdir = outdir_override or io_cfg.get("out_dir", "outputs_v3")
    os.makedirs(outdir, exist_ok=True)

    # Training params
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 8))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 2e-5))
    lr_lora = float(train_cfg.get("lr_lora", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 2))
    eval_every = int(train_cfg.get("eval_every_steps", 100))
    patience = int(train_cfg.get("patience", 10))
    num_workers = int(train_cfg.get("num_workers", 4))
    enable_tf32 = train_cfg.get("tf32", True)

    # Model params
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "microsoft/deberta-v3-base")
    label_smoothing = float(model_cfg.get("label_smoothing", 0.02))
    dropout = float(model_cfg.get("dropout", 0.1))
    max_length = int(model_cfg.get("max_length", 384))
    use_lora = model_cfg.get("use_lora", True)
    lora_rank = int(model_cfg.get("lora_rank", 16))
    lora_alpha = int(model_cfg.get("lora_alpha", 32))
    gradient_checkpointing = model_cfg.get("gradient_checkpointing", True)

    # Data params
    data_cfg = cfg.get("data", {})
    max_train = data_cfg.get("max_train", None)
    max_dev = data_cfg.get("max_dev", None)
    dev_test_ratio = float(data_cfg.get("dev_test_ratio", 0.0))

    # V3-specific params
    v3_cfg = cfg.get("v3", {})
    focal_gamma = float(v3_cfg.get("focal_gamma", 2.0))
    refutes_gamma = float(v3_cfg.get("refutes_gamma", 3.0))
    supports_gamma = float(v3_cfg.get("supports_gamma", 1.5))
    nei_gamma = float(v3_cfg.get("nei_gamma", 0.5))
    rdrop_alpha = float(v3_cfg.get("rdrop_alpha", 1.0))
    rdrop_warmup_epochs = int(v3_cfg.get("rdrop_warmup_epochs", 1))
    symbolic_boost_strength = float(v3_cfg.get("symbolic_boost_strength", 1.0))
    use_symbolic_boost = v3_cfg.get("use_symbolic_boost", True)
    refutes_weight_mult = float(v3_cfg.get("refutes_weight_mult", 1.5))

    # Per-class gamma
    class_gamma = {
        LABEL2ID["SUPPORTS"]: supports_gamma,
        LABEL2ID["REFUTES"]: refutes_gamma,
        LABEL2ID["NOT ENOUGH INFO"]: nei_gamma,
    }

    # ── TF32 ─────────────────────────────────────────────────
    if enable_tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading FEVER dataset...")
    splits = load_fever_splits(
        cache_dir=None, max_train=max_train, max_dev=max_dev,
        dev_test_ratio=dev_test_ratio, seed=seed,
    )
    print_fever_stats(splits)

    train_ds = FeverGoldDataset(splits["train"])
    dev_ds = FeverGoldDataset(splits["dev"])

    # ── Build model (SAME as neural baseline) ────────────────
    tokenizer, base_model = build_fever_model(
        model_name=model_name,
        label_smoothing=label_smoothing,
        dropout=dropout,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Class weights (inverse frequency) with REFUTES boost
    label_counts = Counter(it["label"] for it in splits["train"])
    total_train = sum(label_counts.values())
    class_weights = torch.tensor([
        total_train / (NUM_LABELS * max(1, label_counts.get(FEVER_LABELS[i], 1)))
        for i in range(NUM_LABELS)
    ], dtype=torch.float32)
    # Extra boost for REFUTES
    refutes_id = LABEL2ID["REFUTES"]
    class_weights[refutes_id] *= refutes_weight_mult
    class_weights = class_weights.to(device)
    print(f"  Class weights: {class_weights.tolist()}")

    # Model wrapper (class_weights=None since we use focal loss in loop)
    model = FeverNLIWrapper(
        base_model, label_smoothing=0.0, class_weights=None,
    ).to(device)

    # ── DataLoaders ──────────────────────────────────────────
    pin_memory = device == "cuda"
    train_loader = _build_dataloader(
        train_ds, tokenizer, batch_size, max_length,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
    )
    dev_loader = _build_dataloader(
        dev_ds, tokenizer, batch_size, max_length,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory, seed=seed,
    )

    # ── Optimizer + Scheduler ────────────────────────────────
    backbone_params, lora_params, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params.append(p)
        elif "classifier" in name.lower() or "pooler" in name.lower():
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_lora, "name": "head"})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "name": "lora"})

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    total_steps = epochs * math.floor(len(train_ds) / batch_size) // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Mixed precision ──────────────────────────────────────
    use_amp = device == "cuda"
    amp_dtype = torch.float32  # Use TF32 via matmul flags, not AMP
    scaler = None

    # ── Print config ─────────────────────────────────────────
    _lora_str = f" + LoRA r={lora_rank}" if use_lora else ""
    print(f"\n{'='*60}")
    print(f"  NST v3: Focal Loss + R-Drop + Symbolic Boost")
    print(f"  Model: {model_name}{_lora_str}")
    print(f"  Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M")
    print(f"  Focal gamma: S={supports_gamma} R={refutes_gamma} N={nei_gamma}")
    print(f"  R-Drop α={rdrop_alpha} (warmup {rdrop_warmup_epochs} epochs)")
    print(f"  Symbolic boost: {use_symbolic_boost} (strength={symbolic_boost_strength})")
    print(f"  epochs={epochs}, bs={batch_size}x{grad_accum_steps}, lr={lr}")
    print(f"  total_steps={total_steps}, warmup={warmup_steps}")
    print(f"{'='*60}\n")

    # ── Training loop ────────────────────────────────────────
    best_dev_acc = 0.0
    patience_counter = 0
    train_log = []
    global_step = 0
    nan_abort = False
    early_stop = False
    best_state = None  # Will hold state_dict of best model

    t0 = time.time()

    for epoch in range(epochs):
        if nan_abort or early_stop:
            break

        model.train()
        epoch_loss = 0.0
        epoch_focal = 0.0
        epoch_rdrop = 0.0
        n_steps = 0
        optimizer.zero_grad()

        # R-Drop warmup: no R-Drop in first epoch(s)
        use_rdrop = epoch >= rdrop_warmup_epochs and rdrop_alpha > 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass 1
            result1 = model(input_ids, attention_mask, labels)
            logits1 = result1["logits"]

            # Focal loss (pass 1)
            loss_focal_1 = focal_loss(
                logits1, labels,
                weight=class_weights.to(logits1.dtype),
                gamma=focal_gamma,
                class_gamma=class_gamma,
                label_smoothing=label_smoothing,
            )

            if use_rdrop:
                # Forward pass 2 (with different dropout)
                result2 = model(input_ids, attention_mask, labels)
                logits2 = result2["logits"]

                loss_focal_2 = focal_loss(
                    logits2, labels,
                    weight=class_weights.to(logits2.dtype),
                    gamma=focal_gamma,
                    class_gamma=class_gamma,
                    label_smoothing=label_smoothing,
                )

                # Symmetric R-Drop KL
                loss_rdrop = rdrop_kl_loss(logits1, logits2)

                # Average focal losses + R-Drop
                loss = 0.5 * (loss_focal_1 + loss_focal_2) + rdrop_alpha * loss_rdrop
            else:
                loss = loss_focal_1
                loss_rdrop = torch.tensor(0.0)

            loss = loss / grad_accum_steps

            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at epoch {epoch}, step {step}")
                nan_abort = True
                break

            loss.backward()

            # Gradient step
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Step-level eval
                if eval_every > 0 and global_step % eval_every == 0:
                    dev_metrics = _eval_split(model, dev_loader, device)
                    dev_acc = dev_metrics["accuracy"]
                    print(
                        f"  E{epoch+1} S{global_step}: "
                        f"focal={loss_focal_1.item():.4f} "
                        f"rdrop={loss_rdrop.item():.4f} "
                        f"| dev={dev_acc:.4f} "
                        f"ECE={dev_metrics['ece']:.4f}"
                    )
                    train_log.append({
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "focal_loss": round(loss_focal_1.item(), 4),
                        "rdrop_loss": round(loss_rdrop.item(), 4),
                        "dev_accuracy": dev_acc,
                        "dev_ece": dev_metrics["ece"],
                    })
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        patience_counter = 0
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        ckpt_dir = os.path.join(outdir, "ckpt")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        _unwrap(model).model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        torch.save(
                            {"global_step": global_step, "best_dev_acc": best_dev_acc},
                            os.path.join(ckpt_dir, "training_state.pt"),
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at step {global_step}")
                            early_stop = True
                            break
                    model.train()

            epoch_loss += loss.item() * grad_accum_steps
            epoch_focal += loss_focal_1.item()
            epoch_rdrop += loss_rdrop.item()
            n_steps += 1

        avg_loss = epoch_loss / max(1, n_steps)
        avg_focal = epoch_focal / max(1, n_steps)
        avg_rdrop = epoch_rdrop / max(1, n_steps)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} focal={avg_focal:.4f} rdrop={avg_rdrop:.4f}")

    elapsed = time.time() - t0

    # ── Load best checkpoint ─────────────────────────────────
    if best_state is not None:
        print(f"\n  Restoring best checkpoint (dev_acc={best_dev_acc:.4f})...")
        model.load_state_dict(best_state)

    # ── Final evaluation (NO temperature scaling) ────────────
    print(f"\n{'─'*40}")
    print("  Final evaluation — raw model (no temp scaling)")
    print(f"{'─'*40}")
    final_dev_raw = _eval_split(model, dev_loader, device, symbolic_boost=False)
    print(f"  Accuracy (raw): {final_dev_raw['accuracy']:.4f}")
    print(f"  ECE: {final_dev_raw['ece']:.4f}")
    for label, stats in final_dev_raw.get("per_label", {}).items():
        print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

    # With symbolic boost
    final_dev = final_dev_raw
    if use_symbolic_boost:
        print(f"\n{'─'*40}")
        print(f"  Final evaluation — + symbolic contradiction boost")
        print(f"{'─'*40}")
        final_dev = _eval_split(
            model, dev_loader, device,
            symbolic_boost=True,
            boost_strength=symbolic_boost_strength,
        )
        print(f"  Accuracy (+ symbolic): {final_dev['accuracy']:.4f}")
        print(f"  ECE: {final_dev['ece']:.4f}")
        for label, stats in final_dev.get("per_label", {}).items():
            print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")
        delta = final_dev['accuracy'] - final_dev_raw['accuracy']
        print(f"  Symbolic boost delta: {delta:+.4f}")

    # Held-out
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
        final_dev_test_raw = _eval_split(model, dev_test_loader, device, symbolic_boost=False)
        print(f"  Raw accuracy: {final_dev_test_raw['accuracy']:.4f}")
        for label, stats in final_dev_test_raw.get("per_label", {}).items():
            print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

        final_dev_test = final_dev_test_raw
        if use_symbolic_boost:
            final_dev_test = _eval_split(
                model, dev_test_loader, device,
                symbolic_boost=True, boost_strength=symbolic_boost_strength,
            )
            print(f"  + Symbolic: {final_dev_test['accuracy']:.4f}")
            for label, stats in final_dev_test.get("per_label", {}).items():
                print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

    # ── Build report ─────────────────────────────────────────
    report = {
        "method": "nst_v3",
        "model_name": model_name,
        "use_lora": use_lora,
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 2),
        "focal_gamma": focal_gamma,
        "refutes_gamma": refutes_gamma,
        "rdrop_alpha": rdrop_alpha,
        "symbolic_boost": use_symbolic_boost,
        "seed": seed,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "best_dev_acc": round(best_dev_acc, 4),
        "nan_abort": nan_abort,
        "early_stop": early_stop,
        "dev_raw": final_dev_raw,
        "dev": final_dev,
        "dev_test": final_dev_test,
        "train_log": train_log,
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
