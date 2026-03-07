"""NST-VERI Training Loop: Multi-phase neurosymbolic training.

Implements the full NST-VERI training procedure:

Phase 1 (Epoch 0):     β=1.0, γ=0.0, λ_max=0.0   [Pure NLI + aux tasks]
Phase 2 (Epoch 1):     β=1.0, γ→0.1, λ_max=0.0   [Add contrastive]
Phase 3 (Epochs 2-N):  β=1.0, γ=0.1, λ_max→0.3   [Warmup constraints]

Supports all 6 modes:
  1. neural:     pure DeBERTa cross-entropy (strong baseline)
  2. soft:       CE + fixed-weight constraint loss (v2 constraints)
  3. lagrangian: CE + adaptive Lagrangian constraint loss
  4. cegis:      Lagrangian + counterexample-guided outer loop
  5. gated:      ECCG — Evidence-Conditioned Constraint Gating
  6. veri:       ★ NST-VERI (flagship) — full method

Features:
  - A100-optimized (fused optimizer, grad checkpointing, TF32, SDPA)
  - Multi-phase constraint warmup
  - Per-sample adaptive lambda
  - Verification heads with auxiliary supervision
  - Contrastive constraint loss for representation shaping
  - Comprehensive logging (per-class, calibration, constraint stats)
  - No data leaks (train/dev/dev_test strictly separated)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

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
from symbolic.constraints_v2 import ConstraintEngineV2
from training.adaptive_lambda import AdaptiveLambdaModule
from training.model_setup import (
    build_nst_veri_model, create_optimizer, create_dataloader,
    setup_gpu_optimizations, detect_gpu_tier, build_tokenizer,
)
from eval.calibration_metrics import expected_calibration_error, brier_score
from training.config_validation import load_and_validate_config

logger = logging.getLogger("train_veri")


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


def _get_phase_params(epoch: int, total_epochs: int) -> dict:
    """Determine training phase parameters for the current epoch.

    Phase 1 (epoch 0):     Pure NLI + auxiliary verification tasks
    Phase 2 (epoch 1):     Add contrastive loss
    Phase 3 (epoch 2+):    Add constraint loss with warmup

    Returns dict with: phase, beta, gamma, lambda_schedule_mult
    """
    if epoch == 0:
        return {"phase": 1, "beta": 1.0, "gamma": 0.0, "lambda_schedule": 0.0}
    elif epoch == 1:
        return {"phase": 2, "beta": 1.0, "gamma": 0.1, "lambda_schedule": 0.0}
    else:
        # Phase 3: linear warmup of lambda from 0 to 1 over remaining epochs
        remaining = total_epochs - 2
        progress = min(1.0, (epoch - 2) / max(1, remaining - 1))
        return {"phase": 3, "beta": 1.0, "gamma": 0.1, "lambda_schedule": progress}


def _eval_split(
    model: NSTVeriModel,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model on a split. Returns accuracy, ECE, Brier, per-label metrics."""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

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

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    accuracy = (all_preds == all_labels).float().mean().item()

    per_label = {}
    for label_name, label_id in LABEL2ID.items():
        mask = all_labels == label_id
        if mask.sum() > 0:
            per_label[label_name] = {
                "count": mask.sum().item(),
                "accuracy": (all_preds[mask] == label_id).float().mean().item(),
                "precision": (
                    (all_preds[all_preds == label_id] == all_labels[all_preds == label_id])
                    .float().mean().item()
                    if (all_preds == label_id).sum() > 0 else 0.0
                ),
            }

    ece, _ = expected_calibration_error(all_probs, all_labels)
    bs = brier_score(all_probs, all_labels)

    # Confusion matrix
    confusion = {}
    for gold_name, gold_id in LABEL2ID.items():
        confusion[gold_name] = {}
        gold_mask = all_labels == gold_id
        for pred_name, pred_id in LABEL2ID.items():
            confusion[gold_name][pred_name] = (all_preds[gold_mask] == pred_id).sum().item()

    model.train()
    return {
        "accuracy": round(accuracy, 4),
        "ece": ece,
        "brier": bs,
        "per_label": per_label,
        "confusion": confusion,
        "n_samples": len(all_labels),
    }


def train_nst_veri(
    config_path: str,
    outdir_override: str | None = None,
    config_overrides: dict | None = None,
) -> dict:
    """Train NST-VERI: the flagship neurosymbolic model.

    This is the main training entry point for NST-VERI. It handles:
      - Config loading and GPU auto-detection
      - Model building with LoRA + verification heads
      - Multi-phase training with constraint warmup
      - Per-sample adaptive lambda
      - Comprehensive evaluation and logging

    Args:
        config_path: Path to YAML config file.
        outdir_override: Override output directory.
        config_overrides: Dict of config overrides (e.g., from notebook GPU detection).

    Returns:
        Report dict with all metrics.
    """
    # ── Load config ──────────────────────────────────────────
    cfg = load_and_validate_config(config_path)
    if config_overrides:
        _deep_update(cfg, config_overrides)

    mode = cfg.get("mode", "veri")
    seed = int(cfg.get("seed", 42))

    # ── Reproducibility ──────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _auto_device(cfg.get("device", "auto"))

    # ── GPU optimizations ────────────────────────────────────
    gpu_opts = setup_gpu_optimizations(device)

    # ── Parse config sections ────────────────────────────────
    io_cfg = cfg.get("io", {})
    outdir = outdir_override or io_cfg.get("out_dir", f"outputs_fever_{mode}")
    os.makedirs(outdir, exist_ok=True)

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-5))
    lr_lora = float(train_cfg.get("lr_lora", 3e-4))
    lr_heads = float(train_cfg.get("lr_heads", 5e-4))
    lr_gate = float(train_cfg.get("lr_gate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 2))
    eval_every = int(train_cfg.get("eval_every", train_cfg.get("eval_every_steps", 500)))
    eval_strategy = train_cfg.get("eval_strategy", "steps")
    if "eval_every_steps" in train_cfg and "eval_strategy" not in train_cfg:
        eval_strategy = "steps"
    patience = int(train_cfg.get("patience", 5))
    num_workers = int(train_cfg.get("num_workers", 4))
    fp16 = train_cfg.get("fp16", device == "cuda")
    bf16 = train_cfg.get("bf16", False)
    if bf16:
        fp16 = False
    use_compile = train_cfg.get("torch_compile", False)
    compile_mode = train_cfg.get("compile_mode", "reduce-overhead")
    use_fused = train_cfg.get("fused_optimizer", True)

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "microsoft/deberta-v3-large")
    label_smoothing = float(model_cfg.get("label_smoothing", 0.05))
    dropout = float(model_cfg.get("dropout", 0.1))
    max_length = int(model_cfg.get("max_length", train_cfg.get("max_length", 384)))
    use_lora = model_cfg.get("use_lora", True)
    lora_rank = int(model_cfg.get("lora_rank", 16))
    lora_alpha = int(model_cfg.get("lora_alpha", 32))
    gradient_checkpointing = model_cfg.get("gradient_checkpointing", True)
    pretrained_path = model_cfg.get("pretrained_path", None)

    data_cfg = cfg.get("data", {})
    max_train = data_cfg.get("max_train", None)
    max_dev = data_cfg.get("max_dev", None)
    evidence_mode = data_cfg.get("evidence_mode", cfg.get("evidence_mode", "gold"))
    cache_dir = data_cfg.get("cache_dir", None)
    dev_test_ratio = float(data_cfg.get("dev_test_ratio", 0.1))

    # NST-VERI specific config
    veri_cfg = cfg.get("veri", {})
    n_constraints = int(veri_cfg.get("n_constraints", 6))
    lambda_max = float(veri_cfg.get("lambda_max", 0.3))
    contrastive_temp = float(veri_cfg.get("contrastive_temperature", 0.07))

    # Legacy constraint config for backward compatibility
    logic_cfg = cfg.get("logic", cfg.get("constraints", {}))
    constraint_lambda = float(logic_cfg.get("lambda", 0.1))

    # ── Load data ────────────────────────────────────────────
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
        from retrieval.bm25_retriever import BM25Retriever, build_synthetic_sentence_store
        logger.info("Using FULL PIPELINE mode (Setting B)")
        retriever_path = data_cfg.get("retriever_index", None)
        if retriever_path and os.path.exists(retriever_path):
            retriever = BM25Retriever.load_index(retriever_path)
        else:
            store = build_synthetic_sentence_store(
                splits["train"] + splits["dev"], noise_sentences=50,
            )
            retriever = BM25Retriever(sentence_store=store, top_k_sents=5)
        train_evidence = retriever.retrieve_all_cached(splits["train"], "fever_train")
        dev_evidence = retriever.retrieve_all_cached(splits["dev"], "fever_dev")
        train_ds = FeverPipelineDataset(splits["train"], train_evidence)
        dev_ds = FeverPipelineDataset(splits["dev"], dev_evidence)

    # ── Compute class weights ────────────────────────────────
    label_counts = Counter(it["label"] for it in splits["train"])
    total_train = sum(label_counts.values())
    class_weights = None
    if total_train > 0:
        class_weights = torch.tensor([
            total_train / (NUM_LABELS * max(1, label_counts.get(FEVER_LABELS[i], 1)))
            for i in range(NUM_LABELS)
        ], dtype=torch.float32).to(device)
        logger.info(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    # ── Build model ──────────────────────────────────────────
    tokenizer, model = build_nst_veri_model(
        model_name=model_name,
        label_smoothing=label_smoothing,
        dropout=dropout,
        n_constraints=n_constraints,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gradient_checkpointing=gradient_checkpointing,
        class_weights=class_weights,
        contrastive_temperature=contrastive_temp,
        pretrained_path=pretrained_path,
    )
    model = model.to(device)

    # ── Constraint engine ────────────────────────────────────
    constraint_engine = ConstraintEngineV2()
    logger.info(f"Constraint engine: {constraint_engine.n_constraints} constraints "
                f"({', '.join(constraint_engine.constraint_names)})")

    # ── Adaptive lambda module ───────────────────────────────
    adaptive_lambda = None
    if mode == "veri":
        adaptive_lambda = AdaptiveLambdaModule(
            n_constraints=n_constraints,
            lambda_max=lambda_max,
        ).to(device)
        logger.info(f"Adaptive lambda: λ_max={lambda_max}")

    # ── DataLoaders ──────────────────────────────────────────
    pin_memory = device == "cuda"
    train_loader = create_dataloader(
        train_ds, tokenizer, batch_size, max_length,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        seed=seed,
    )
    dev_loader = create_dataloader(
        dev_ds, tokenizer, batch_size, max_length,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        seed=seed,
    )

    # Dev sample for fast periodic evaluation
    dev_sample = data_cfg.get("dev_sample", 2000)
    if dev_sample and isinstance(dev_sample, int) and dev_sample < len(dev_ds):
        from torch.utils.data import Subset
        rng = random.Random(seed)
        dev_sample_indices = rng.sample(range(len(dev_ds)), dev_sample)
        dev_sample_ds = Subset(dev_ds, dev_sample_indices)
        dev_sample_loader = create_dataloader(
            dev_sample_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
            seed=seed,
        )
    else:
        dev_sample_loader = dev_loader

    # ── Optimizer ────────────────────────────────────────────
    optimizer = create_optimizer(
        model,
        lr_backbone=lr,
        lr_lora=lr_lora,
        lr_heads=lr_heads,
        lr_gate=lr_gate,
        weight_decay=weight_decay,
        use_fused=use_fused,
        adaptive_lambda_module=adaptive_lambda,
    )

    # ── LR Scheduler ─────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps = epochs * math.ceil(steps_per_epoch / grad_accum_steps)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── torch.compile ────────────────────────────────────────
    _compiled = False
    if use_compile and hasattr(torch, "compile") and device == "cuda":
        try:
            model = torch.compile(model, mode=compile_mode or "reduce-overhead")
            if adaptive_lambda is not None:
                adaptive_lambda = torch.compile(adaptive_lambda)
            _compiled = True
            logger.info(f"torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # ── Mixed precision ──────────────────────────────────────
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

    # ── Training state ───────────────────────────────────────
    best_dev_acc = 0.0
    patience_counter = 0
    train_log = []
    global_step = 0
    nan_abort = False

    _prec = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
    eff_bs = batch_size * grad_accum_steps
    print(f"\n{'='*65}")
    print(f"  NST-VERI Training: mode={mode}")
    print(f"  Model: {model_name}" + (f" + LoRA r={lora_rank}" if use_lora else ""))
    print(f"  Epochs: {epochs}, Batch: {batch_size}×{grad_accum_steps}={eff_bs}")
    print(f"  LR: backbone={lr}, LoRA={lr_lora}, heads={lr_heads}")
    print(f"  Precision: {_prec}, Compile: {_compiled}, Device: {device}")
    print(f"  Constraints: {n_constraints}, λ_max={lambda_max}")
    print(f"  Evidence: {evidence_mode}, Max seq: {max_length}")
    print(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"{'='*65}\n")

    t0 = time.time()

    # ── Lagrangian state for non-veri modes ──────────────────
    from symbolic.lagrangian import LagrangianState, lagrangian_loss, update_dual_variable
    lag_state = LagrangianState(
        lam=constraint_lambda if mode == "soft" else 0.0,
        epsilon=float(cfg.get("lagrangian", {}).get("epsilon", 0.05)),
        alpha=float(cfg.get("lagrangian", {}).get("alpha", 0.01)),
        rho=float(cfg.get("lagrangian", {}).get("rho", 1.0)),
        lam_max=float(cfg.get("lagrangian", {}).get("lam_max", 10.0)),
    )

    # For CEGIS
    max_rounds = int(cfg.get("cegis", {}).get("max_rounds", 5))
    max_ce = int(cfg.get("cegis", {}).get("max_counterexamples", 500))
    n_rounds = max_rounds if mode == "cegis" else 1
    cegis_log = [] if mode == "cegis" else None

    # Legacy imports for backward-compat modes
    if mode in ("soft", "lagrangian", "cegis", "gated"):
        from symbolic.fever_constraints import extract_batch_facts
        from symbolic.fever_constraint_loss import fever_constraint_loss
    if mode == "gated":
        from symbolic.constraint_gating import (
            ConstraintGate, gated_fever_constraint_loss,
        )
        gate_cfg_data = cfg.get("gate", cfg.get("eccg", {}))
        constraint_gate = ConstraintGate(
            hidden_dim=int(gate_cfg_data.get("hidden_dim", 16)),
        ).to(device)

    # ── Main training loop ───────────────────────────────────
    for cegis_round in range(n_rounds):
        if nan_abort:
            break

        round_label = f"R{cegis_round}/" if mode == "cegis" else ""

        for epoch in range(epochs):
            if nan_abort:
                break

            # Phase parameters (for veri mode)
            phase_params = _get_phase_params(epoch, epochs) if mode == "veri" else {
                "phase": 3, "beta": 0.0, "gamma": 0.0, "lambda_schedule": 1.0
            }

            model.train()
            if adaptive_lambda is not None:
                _unwrap(adaptive_lambda).train()

            epoch_loss = 0.0
            epoch_nli_loss = 0.0
            epoch_aux_loss = 0.0
            epoch_contrast_loss = 0.0
            epoch_constraint_loss = 0.0
            n_steps = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):

                    if mode == "veri":
                        # ── NST-VERI forward pass ──
                        # Get constraint signals
                        c_signals = constraint_engine.evaluate_batch(
                            batch["claims"], batch["evidences"]
                        )

                        # Adaptive lambda
                        ada_lambda_out = None
                        if adaptive_lambda is not None and phase_params["phase"] >= 3:
                            # First get model probs for lambda estimation
                            with torch.no_grad():
                                quick_out = model.predict(input_ids, attention_mask)
                                model_probs = quick_out["probs"]

                            ada_lambda_out = _unwrap(adaptive_lambda)(
                                c_signals["fires"].to(device),
                                c_signals["confidence"].to(device),
                                model_probs,
                                schedule_multiplier=phase_params["lambda_schedule"],
                            )

                        # Main forward
                        result = model(
                            input_ids, attention_mask, labels,
                            constraint_signals=c_signals,
                            phase=phase_params["phase"],
                            beta=phase_params["beta"],
                            gamma=phase_params["gamma"],
                            adaptive_lambda=ada_lambda_out,
                        )
                        loss = result["loss"] / grad_accum_steps

                        epoch_nli_loss += result["loss_nli"].item()
                        epoch_aux_loss += result["loss_aux"].item()
                        epoch_contrast_loss += result["loss_contrastive"].item()
                        epoch_constraint_loss += result["loss_constraint"].item()

                    elif mode == "neural":
                        # ── Pure neural baseline ──
                        result = model(input_ids, attention_mask, labels,
                                       phase=1, beta=0.0, gamma=0.0)
                        loss = result["loss_nli"] / grad_accum_steps

                    elif mode in ("soft", "lagrangian", "cegis"):
                        # ── Legacy constraint modes ──
                        result = model(input_ids, attention_mask, labels,
                                       phase=1, beta=0.0, gamma=0.0)
                        loss_task = result["loss_nli"]
                        probs = result["probs"]

                        facts = extract_batch_facts(batch["claims"], batch["evidences"])
                        loss_c, _ = fever_constraint_loss(
                            probs[:, LABEL2ID["SUPPORTS"]],
                            probs[:, LABEL2ID["REFUTES"]],
                            probs[:, LABEL2ID["NOT ENOUGH INFO"]],
                            facts,
                        )
                        epoch_constraint_loss += loss_c.item()

                        if mode == "soft":
                            loss = (loss_task + lag_state.lam * loss_c) / grad_accum_steps
                        else:
                            loss = lagrangian_loss(loss_task, loss_c, lag_state) / grad_accum_steps

                    elif mode == "gated":
                        # ── ECCG legacy mode ──
                        result = model(input_ids, attention_mask, labels,
                                       phase=1, beta=0.0, gamma=0.0)
                        loss_task = result["loss_nli"]
                        probs = result["probs"]

                        facts = extract_batch_facts(batch["claims"], batch["evidences"])
                        loss_c, _ = gated_fever_constraint_loss(
                            probs[:, LABEL2ID["SUPPORTS"]],
                            probs[:, LABEL2ID["REFUTES"]],
                            probs[:, LABEL2ID["NOT ENOUGH INFO"]],
                            facts, gate=constraint_gate,
                        )
                        epoch_constraint_loss += loss_c.item()
                        loss = lagrangian_loss(loss_task, loss_c, lag_state) / grad_accum_steps

                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                # Backward
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # NaN guard
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf at epoch {epoch}, step {step}")
                    nan_abort = True
                    break

                # Gradient step
                if (step + 1) % grad_accum_steps == 0:
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(model.parameters()) +
                            (list(adaptive_lambda.parameters()) if adaptive_lambda else []),
                            max_grad_norm,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            list(model.parameters()) +
                            (list(adaptive_lambda.parameters()) if adaptive_lambda else []),
                            max_grad_norm,
                        )
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Step-level evaluation
                    if (eval_strategy == "steps"
                            and eval_every > 0
                            and global_step % eval_every == 0):
                        dev_metrics = _eval_split(model, dev_sample_loader, device)
                        dev_acc = dev_metrics["accuracy"]

                        phase_str = f" P{phase_params['phase']}" if mode == "veri" else ""
                        print(
                            f"  {round_label}Step {global_step}{phase_str}: "
                            f"loss={loss.item()*grad_accum_steps:.4f} | "
                            f"dev_acc={dev_acc:.4f} ECE={dev_metrics['ece']:.4f}"
                        )

                        train_log.append({
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "phase": phase_params.get("phase", 0),
                            "train_loss": round(loss.item() * grad_accum_steps, 4),
                            "dev_accuracy": dev_acc,
                            "dev_ece": dev_metrics["ece"],
                            "dev_brier": dev_metrics["brier"],
                        })

                        if dev_acc > best_dev_acc:
                            best_dev_acc = dev_acc
                            patience_counter = 0
                            _save_checkpoint(model, tokenizer, adaptive_lambda,
                                             outdir, global_step, best_dev_acc, mode)
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                logger.info(f"Early stopping at step {global_step}")
                                nan_abort = True
                                break
                        model.train()

                epoch_loss += loss.item() * grad_accum_steps
                n_steps += 1

            if nan_abort:
                break

            avg_loss = epoch_loss / max(1, n_steps)

            # Update Lagrangian for legacy modes
            if mode in ("lagrangian", "cegis", "gated"):
                avg_c = epoch_constraint_loss / max(1, n_steps)
                update_dual_variable(lag_state, avg_c, step=epoch)

            # Epoch-level evaluation
            do_eval = (
                eval_strategy == "epoch"
                or epoch == epochs - 1
            )
            if do_eval:
                dev_metrics = _eval_split(model, dev_loader, device)
                dev_acc = dev_metrics["accuracy"]

                phase_str = f" P{phase_params['phase']}" if mode == "veri" else ""
                nli_str = f" nli={epoch_nli_loss/max(1,n_steps):.4f}" if mode == "veri" else ""
                aux_str = f" aux={epoch_aux_loss/max(1,n_steps):.4f}" if mode == "veri" else ""
                con_str = f" ctr={epoch_contrast_loss/max(1,n_steps):.4f}" if mode == "veri" else ""
                cst_str = f" cst={epoch_constraint_loss/max(1,n_steps):.4f}" if mode != "neural" else ""

                print(
                    f"  {round_label}Epoch {epoch+1}/{epochs}{phase_str}: "
                    f"loss={avg_loss:.4f}{nli_str}{aux_str}{con_str}{cst_str} | "
                    f"dev_acc={dev_acc:.4f} ECE={dev_metrics['ece']:.4f}"
                )

                # Per-class breakdown
                for label, stats in dev_metrics.get("per_label", {}).items():
                    print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

                train_log.append({
                    "epoch": epoch + 1,
                    "phase": phase_params.get("phase", 0),
                    "train_loss": round(avg_loss, 4),
                    "dev_accuracy": dev_acc,
                    "dev_ece": dev_metrics["ece"],
                    "dev_brier": dev_metrics["brier"],
                    "per_label": dev_metrics["per_label"],
                })

                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    patience_counter = 0
                    _save_checkpoint(model, tokenizer, adaptive_lambda,
                                     outdir, global_step, best_dev_acc, mode)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

    elapsed = time.time() - t0

    # ── Post-hoc temperature scaling ───────────────────────
    from eval.temperature_scaling import learn_temperature
    print(f"\n{'─'*50}")
    print("  Post-hoc temperature scaling")
    print(f"{'─'*50}")
    optimal_T = learn_temperature(model, dev_loader, device)

    # ── Final evaluation ─────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Final evaluation on dev set")
    print(f"{'─'*50}")
    final_dev = _eval_split(model, dev_loader, device)
    _print_eval_results(final_dev, evidence_mode)

    # ── Held-out dev_test ────────────────────────────────────
    final_dev_test = None
    if "dev_test" in splits and splits["dev_test"]:
        print(f"\n{'─'*50}")
        print("  Final evaluation on held-out dev_test")
        print(f"{'─'*50}")
        if evidence_mode == "gold":
            dev_test_ds = FeverGoldDataset(splits["dev_test"])
        else:
            dev_test_ds = FeverPipelineDataset(splits["dev_test"], dev_evidence)
        dev_test_loader = create_dataloader(
            dev_test_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
            seed=seed,
        )
        final_dev_test = _eval_split(model, dev_test_loader, device)
        _print_eval_results(final_dev_test, evidence_mode, split_name="dev_test")

    # ── Constraint calibration report ────────────────────────
    if mode == "veri":
        print(f"\n{'─'*50}")
        print("  Constraint calibration (on training data sample)")
        print(f"{'─'*50}")
        train_claims = [it["claim"] for it in splits["train"][:5000]]
        train_evs = [it["gold_evidence_text"] for it in splits["train"][:5000]]
        train_labels = [it["label_id"] for it in splits["train"][:5000]]
        from symbolic.constraints_v2 import calibrate_constraints
        cal_stats = calibrate_constraints(
            constraint_engine, train_claims, train_evs, train_labels, n_samples=5000
        )
        for name, stats in cal_stats.items():
            print(f"    {name}: precision={stats['precision']:.3f}, "
                  f"fire_rate={stats['fire_rate']:.3f}")

    # ── Build report ─────────────────────────────────────────
    report = {
        "mode": mode,
        "method": "NST-VERI" if mode == "veri" else mode.upper(),
        "evidence_mode": evidence_mode,
        "model_name": model_name,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else 0,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "lr": lr,
        "max_length": max_length,
        "elapsed_s": round(elapsed, 1),
        "best_dev_acc": round(best_dev_acc, 4),
        "nan_abort": nan_abort,
        "dev": final_dev,
        "dev_test": final_dev_test,
        "train_log": train_log,
        "temperature": round(optimal_T, 4),
        "gpu_optimizations": gpu_opts,
    }

    # Save everything
    _save_report(report, outdir, train_log)

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best dev accuracy: {best_dev_acc:.4f}")
    if final_dev_test:
        print(f"  Dev_test accuracy: {final_dev_test['accuracy']:.4f}")
    print(f"  Output: {outdir}")

    return report


def _save_checkpoint(model, tokenizer, adaptive_lambda, outdir, step, acc, mode):
    """Save model checkpoint."""
    ckpt_dir = os.path.join(outdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    unwrapped = _unwrap(model)
    # Save backbone
    if hasattr(unwrapped, 'backbone'):
        unwrapped.backbone.save_pretrained(ckpt_dir)
    elif hasattr(unwrapped, 'model'):
        unwrapped.model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Save verification heads + contrastive head
    torch.save({
        "step": step,
        "best_dev_acc": acc,
        "mode": mode,
        "verification_state": unwrapped.verification.state_dict() if hasattr(unwrapped, 'verification') else None,
        "contrastive_state": unwrapped.contrastive.state_dict() if hasattr(unwrapped, 'contrastive') else None,
    }, os.path.join(ckpt_dir, "nst_veri_state.pt"))

    # Save adaptive lambda
    if adaptive_lambda is not None:
        torch.save(
            _unwrap(adaptive_lambda).state_dict(),
            os.path.join(ckpt_dir, "adaptive_lambda.pt"),
        )


def _save_report(report, outdir, train_log):
    """Save report and training log."""
    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(os.path.join(outdir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)


def _print_eval_results(metrics, evidence_mode, split_name="dev"):
    """Pretty-print evaluation results."""
    print(f"  Label Accuracy ({evidence_mode.upper()} evidence, {split_name}): "
          f"{metrics['accuracy']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")
    for label, stats in metrics.get("per_label", {}).items():
        prec = stats.get("precision", "N/A")
        prec_str = f" prec={prec:.4f}" if isinstance(prec, float) else ""
        print(f"    {label}: acc={stats['accuracy']:.4f}{prec_str} (n={stats['count']})")

    if "confusion" in metrics:
        print("  Confusion matrix:")
        labels = list(metrics["confusion"].keys())
        header = "  " + " " * 22 + "  ".join(f"{l[:6]:>8}" for l in labels)
        print(header)
        for gold in labels:
            row = f"  {gold:<20}"
            for pred in labels:
                row += f"  {metrics['confusion'][gold][pred]:>8}"
            print(row)
