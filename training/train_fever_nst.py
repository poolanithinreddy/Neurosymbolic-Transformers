"""FEVER training loop with NST constraint integration.

Supports five modes (matching multi-digit/kinship pattern):
  1. neural:     pure DeBERTa cross-entropy (baseline)
  2. soft:       cross-entropy + fixed-weight constraint loss
  3. lagrangian: cross-entropy + adaptive Lagrangian constraint loss
  4. cegis:      Lagrangian + counterexample-guided outer loop
  5. gated:      ECCG — Evidence-Conditioned Constraint Gating (novel)

Training features:
  - Mixed precision (fp16/bf16) via torch.amp
  - Gradient accumulation for large effective batch size
  - Linear warmup + cosine decay scheduler
  - Early stopping on dev label accuracy
  - NaN fail-fast with debug dump
  - Periodic evaluation with ECE/Brier metrics
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
from models.fever_nli import build_fever_model, FeverNLIWrapper
from symbolic.fever_constraints import extract_batch_facts
from symbolic.fever_constraint_loss import fever_constraint_loss, verify_fever_constraints
from symbolic.constraint_gating import (
    ConstraintGate, gated_fever_constraint_loss, facts_to_gate_features,
)
from symbolic.lagrangian import (
    LagrangianState, lagrangian_loss, update_dual_variable,
    save_lambda_trajectory,
)
from eval.calibration_metrics import expected_calibration_error, brier_score
from training.config_validation import load_and_validate_config

logger = logging.getLogger("train_fever")


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* config dict (in-place)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _unwrap(model):
    """Unwrap torch.compile'd model to access original module."""
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
    dataset,
    tokenizer,
    batch_size: int,
    max_length: int = 384,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build DataLoader with tokenizing collate function.

    Uses a seeded ``torch.Generator`` for reproducible shuffle order.
    Optimized for GPU training with persistent workers and prefetch.
    """
    collate = partial(fever_collate_fn, tokenizer=tokenizer, max_length=max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_persistent = num_workers > 0
    pf = 4 if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,  # Even batch sizes for stability
        generator=generator if shuffle else None,
        persistent_workers=use_persistent,
        prefetch_factor=pf,
    )


def _eval_split(
    model: FeverNLIWrapper,
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

            out = model(input_ids, attention_mask)
            probs = out["probs"]
            preds = probs.argmax(dim=-1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # Overall accuracy
    accuracy = (all_preds == all_labels).float().mean().item()

    # Per-label accuracy
    per_label = {}
    for label_name, label_id in LABEL2ID.items():
        mask = all_labels == label_id
        if mask.sum() > 0:
            per_label[label_name] = {
                "count": mask.sum().item(),
                "accuracy": (all_preds[mask] == label_id).float().mean().item(),
            }

    # Calibration
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


def _mine_counterexamples(
    model: FeverNLIWrapper,
    dataloader: DataLoader,
    device: str,
    max_ce: int = 500,
) -> list[dict]:
    """Mine counterexamples: samples where model predictions violate constraints.

    Returns list of dicts with claim, evidence, label_id, violation info.
    """
    model.eval()
    counterexamples = []

    with torch.no_grad():
        for batch in dataloader:
            if len(counterexamples) >= max_ce:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            result = model.predict(input_ids, attention_mask)
            pred_labels = result["pred_labels"]
            claims = batch["claims"]
            evidences = batch["evidences"]

            # Extract facts and check constraints
            facts_batch = extract_batch_facts(claims, evidences)
            violations, _ = verify_fever_constraints(pred_labels, facts_batch)

            for i, violated in enumerate(violations):
                if violated and len(counterexamples) < max_ce:
                    counterexamples.append({
                        "claim": claims[i],
                        "evidence": evidences[i],
                        "label_id": labels[i].item(),
                        "pred_label": pred_labels[i],
                        "gold_label": ID2LABEL[labels[i].item()],
                    })

    model.train()
    return counterexamples


def train_fever_nst(
    config_path: str,
    outdir_override: str | None = None,
    config_overrides: dict | None = None,
) -> dict:
    """Train FEVER NLI model with NST constraints.

    Modes (set in config):
      - neural: pure cross-entropy
      - soft: CE + fixed λ * constraint_loss
      - lagrangian: CE + adaptive Lagrangian
      - cegis: Lagrangian + counterexample-guided outer loop
      - gated: ECCG — learned per-sample, per-constraint gates

    Returns:
        Report dict with train/dev metrics.
    """
    # ── Load config ──────────────────────────────────────────
    cfg = load_and_validate_config(config_path)
    if config_overrides:
        _deep_update(cfg, config_overrides)
    mode = cfg.get("mode", "neural")
    seed = int(cfg.get("seed", 42))

    # ── Full reproducibility ─────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms — slight perf hit but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = _auto_device(cfg.get("device", "auto"))

    # IO config — support both flat and nested
    io_cfg = cfg.get("io", {})
    outdir = outdir_override or io_cfg.get("out_dir", cfg.get("outdir", f"outputs_fever_{mode}"))
    os.makedirs(outdir, exist_ok=True)

    # Training hyperparameters — support both nested 'train' and flat keys
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", cfg.get("epochs", 3)))
    batch_size = int(train_cfg.get("batch_size", cfg.get("batch_size", 16)))
    lr = float(train_cfg.get("lr", cfg.get("lr", 2e-5)))
    weight_decay = float(train_cfg.get("weight_decay", cfg.get("weight_decay", 0.01)))
    warmup_ratio = float(train_cfg.get("warmup_ratio", cfg.get("warmup_ratio", 0.1)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", train_cfg.get("grad_clip", 1.0)))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    eval_every = int(train_cfg.get("eval_every", train_cfg.get("eval_every_steps", 1)))
    eval_strategy = train_cfg.get("eval_strategy", "epoch")  # "epoch" or "steps"
    # If key is explicitly 'eval_every_steps', treat as step-level
    if "eval_every_steps" in train_cfg and "eval_strategy" not in train_cfg:
        eval_strategy = "steps"
    patience = int(train_cfg.get("patience", 5))
    num_workers = int(train_cfg.get("num_workers", 0))
    fp16 = train_cfg.get("fp16", device == "cuda")
    bf16 = train_cfg.get("bf16", False)
    if bf16:
        fp16 = False  # bf16 takes priority over fp16
    use_compile = train_cfg.get("torch_compile", False)
    enable_tf32 = train_cfg.get("tf32", False)
    enable_benchmark = train_cfg.get("benchmark", False)

    # Model config — support nested 'model' section
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", cfg.get("model_name", "microsoft/deberta-v3-base"))
    label_smoothing = float(model_cfg.get("label_smoothing", cfg.get("label_smoothing", 0.0)))
    dropout = float(model_cfg.get("dropout", cfg.get("dropout", 0.1)))
    max_length = int(model_cfg.get("max_length", train_cfg.get("max_length", 384)))
    use_lora = model_cfg.get("use_lora", False)
    lora_rank = int(model_cfg.get("lora_rank", 16))
    lora_alpha = int(model_cfg.get("lora_alpha", 32))
    gradient_checkpointing = model_cfg.get("gradient_checkpointing", False)
    lr_lora = float(train_cfg.get("lr_lora", 3e-4))
    use_fused = train_cfg.get("fused_optimizer", False)

    # Data config
    data_cfg = cfg.get("data", {})
    max_train = data_cfg.get("max_train", None)
    max_dev = data_cfg.get("max_dev", None)
    evidence_mode = data_cfg.get("evidence_mode", cfg.get("evidence_mode", "gold"))
    cache_dir = data_cfg.get("cache_dir", None)
    dev_test_ratio = float(data_cfg.get("dev_test_ratio", 0.0))

    # Constraint config — support both 'logic' and 'constraints' sections
    logic_cfg = cfg.get("logic", cfg.get("constraints", {}))
    constraint_lambda = float(logic_cfg.get("lambda", logic_cfg.get("lambda_constraint", 0.1)))
    constraint_weights = logic_cfg.get("constraint_weights", logic_cfg.get("weights", {}))

    # Lagrangian config
    lag_cfg = cfg.get("lagrangian", {})
    lag_epsilon = float(lag_cfg.get("epsilon", logic_cfg.get("epsilon", 0.05)))
    lag_alpha = float(lag_cfg.get("alpha", logic_cfg.get("lambda_lr", 0.01)))
    lag_rho = float(lag_cfg.get("rho", 1.0))
    lag_lam_max = float(lag_cfg.get("lam_max", logic_cfg.get("lambda_max", 10.0)))

    # CEGIS config
    cegis_cfg = cfg.get("cegis", {})
    max_rounds = int(cegis_cfg.get("max_rounds", 5))
    max_counterexamples = int(cegis_cfg.get("max_counterexamples",
                                             cegis_cfg.get("mine_top_k", 500)))
    ce_oversample = int(cegis_cfg.get("ce_oversample", cegis_cfg.get("oversample", 3)))

    # Gated (ECCG) config
    gate_cfg = cfg.get("gate", cfg.get("eccg", {}))
    gate_hidden = int(gate_cfg.get("hidden_dim", 16))
    gate_dropout = float(gate_cfg.get("dropout", 0.1))
    gate_init_bias = float(gate_cfg.get("init_bias", 0.5))
    gate_lr_mult = float(gate_cfg.get("lr_multiplier", 10.0))  # Gate learns faster than backbone

    # ── Performance tuning (Ampere+ / Hopper GPUs) ────────
    if enable_tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul and cuDNN")
    if enable_benchmark and device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("cuDNN benchmark mode enabled")

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
        # Pipeline mode — requires retriever
        from retrieval.bm25_retriever import BM25Retriever, build_synthetic_sentence_store
        logger.info("Using FULL PIPELINE mode (Setting B)")

        # Build or load retriever
        retriever_path = data_cfg.get("retriever_index", None)
        if retriever_path and os.path.exists(retriever_path):
            retriever = BM25Retriever.load_index(retriever_path)
        else:
            logger.warning(
                "No Wikipedia dump available. Building synthetic sentence store "
                "for smoke-testing. RESULTS ARE NOT LEGITIMATE for paper claims."
            )
            store = build_synthetic_sentence_store(
                splits["train"] + splits["dev"],
                noise_sentences=50,
            )
            retriever = BM25Retriever(sentence_store=store, top_k_sents=5)

        # Retrieve evidence
        train_evidence = retriever.retrieve_all_cached(splits["train"], "fever_train")
        dev_evidence = retriever.retrieve_all_cached(splits["dev"], "fever_dev")

        train_ds = FeverPipelineDataset(splits["train"], train_evidence)
        dev_ds = FeverPipelineDataset(splits["dev"], dev_evidence)

    # ── Build model ──────────────────────────────────────────
    tokenizer, base_model = build_fever_model(
        model_name=model_name,
        label_smoothing=label_smoothing,
        dropout=dropout,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Compute class weights for imbalanced FEVER labels
    from collections import Counter
    label_counts = Counter(it["label"] for it in splits["train"])
    total_train = sum(label_counts.values())
    n_classes = NUM_LABELS
    class_weights = None
    if total_train > 0:
        class_weights = torch.tensor([
            total_train / (n_classes * max(1, label_counts.get(FEVER_LABELS[i], 1)))
            for i in range(n_classes)
        ], dtype=torch.float32).to(device)
        logger.info(f"Class weights: {class_weights.tolist()}")

    model = FeverNLIWrapper(
        base_model,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
    ).to(device)

    # ── DataLoaders ──────────────────────────────────────────
    pin_memory = device == "cuda"
    train_loader = _build_dataloader(
        train_ds, tokenizer, batch_size, max_length,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        seed=seed,
    )
    dev_loader = _build_dataloader(
        dev_ds, tokenizer, batch_size, max_length,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        seed=seed,
    )

    # Optional: smaller dev subset for fast periodic evaluation
    dev_sample = data_cfg.get("dev_sample", None)
    if dev_sample and isinstance(dev_sample, int) and dev_sample < len(dev_ds):
        from torch.utils.data import Subset
        rng = random.Random(seed)
        dev_sample_indices = rng.sample(range(len(dev_ds)), dev_sample)
        dev_sample_ds = Subset(dev_ds, dev_sample_indices)
        dev_sample_loader = _build_dataloader(
            dev_sample_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
            seed=seed,
        )
        logger.info(f"Periodic eval uses dev subset: {dev_sample}/{len(dev_ds)} examples")
    else:
        dev_sample_loader = dev_loader

    # ── Optimizer + Scheduler ────────────────────────────────
    # Build constraint gate for 'gated' mode
    constraint_gate = None
    if mode == "gated":
        constraint_gate = ConstraintGate(
            hidden_dim=gate_hidden,
            dropout=gate_dropout,
            init_bias=gate_init_bias,
        ).to(device)
        logger.info(f"ECCG gate params: {sum(p.numel() for p in constraint_gate.parameters())}")

    # ── torch.compile (Ampere+ / Hopper GPUs) ────────────────
    _compiled = False
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            if constraint_gate is not None:
                constraint_gate = torch.compile(constraint_gate)
            _compiled = True
            logger.info("torch.compile enabled — first batch will be slow (compilation)")
        except Exception as e:
            logger.warning(f"torch.compile failed, using eager mode: {e}")

    # Separate param groups: backbone at base lr, LoRA at higher lr,
    # classifier/pooler at LoRA lr (randomly initialized, needs fast learning)
    backbone_params = []
    lora_params = []
    head_params = []  # classifier + pooler (from modules_to_save)
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
        # Randomly initialized heads need higher LR (same as LoRA or higher)
        param_groups.append({"params": head_params, "lr": lr_lora, "name": "head"})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "name": "lora"})
    elif not backbone_params and not head_params:
        # Fallback: all params at base lr
        param_groups = [{"params": list(model.parameters()), "lr": lr}]
    if constraint_gate is not None:
        param_groups.append({
            "params": list(constraint_gate.parameters()),
            "lr": lr * gate_lr_mult,
            "name": "gate",
        })

    # Log param groups
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.2f}M total")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        logger.info(f"  {pg.get('name', '?')}: {n_p/1e6:.2f}M params, lr={pg['lr']}")

    fused = use_fused and torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay, fused=fused,
        )
        if fused:
            logger.info("Using fused AdamW")
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay,
        )
    total_steps = epochs * math.ceil(len(train_ds) / batch_size / grad_accum_steps)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Lagrangian state ─────────────────────────────────────
    lag_state = LagrangianState(
        lam=constraint_lambda if mode == "soft" else 0.0,
        epsilon=lag_epsilon,
        alpha=lag_alpha,
        rho=lag_rho,
        lam_max=lag_lam_max,
    )

    # ── Mixed precision ──────────────────────────────────────
    use_amp = (fp16 or bf16) and device == "cuda"
    if bf16 and device == "cuda":
        amp_dtype = torch.bfloat16
        scaler = None  # BF16 has sufficient dynamic range — no GradScaler needed
    elif fp16 and device == "cuda":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    else:
        amp_dtype = torch.float32
        scaler = None

    # ── Training loop ────────────────────────────────────────
    best_dev_acc = 0.0
    patience_counter = 0
    train_log = []
    nan_abort = False
    ce_buffer_ds = None  # For CEGIS
    global_step = 0  # Tracks optimizer steps across epochs/rounds

    n_rounds = max_rounds if mode == "cegis" else 1
    cegis_log = [] if mode == "cegis" else None

    _prec = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
    _lora_str = f" + LoRA r={lora_rank}" if use_lora else " (full FT)"
    print(f"\n{'='*60}")
    print(f"  FEVER Training: mode={mode}")
    print(f"  Model: {model_name}{_lora_str}")
    print(f"  Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")
    print(f"  epochs={epochs}, bs={batch_size}x{grad_accum_steps}={batch_size*grad_accum_steps}, lr={lr}" + (f", lr_lora={lr_lora}" if lora_params else ""))
    print(f"  evidence_mode={evidence_mode}, precision={_prec}, compile={_compiled}")
    if mode == "gated":
        print(f"  ECCG gate: hidden={gate_hidden}, lr_mult={gate_lr_mult}")
    print(f"  total_steps={total_steps}, warmup={warmup_steps}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for cegis_round in range(n_rounds):
        if nan_abort:
            break

        # CEGIS: augment training data with counterexamples
        if mode == "cegis" and cegis_round > 0:
            # IMPORTANT: Mine counterexamples from TRAINING set, NOT dev set.
            # Mining from dev would constitute training on dev data = leakage.
            counterexamples = _mine_counterexamples(
                model, train_loader, device, max_ce=max_counterexamples,
            )
            n_ce = len(counterexamples)
            logger.info(f"CEGIS round {cegis_round}: {n_ce} counterexamples found")

            if n_ce == 0:
                logger.info("CEGIS converged: no counterexamples")
                if cegis_log is not None:
                    cegis_log.append({
                        "round": cegis_round,
                        "n_counterexamples": 0,
                        "converged": True,
                    })
                break

            # Create CE dataset
            ce_items = []
            for ce in counterexamples * ce_oversample:
                ce_items.append({
                    "id": -1,
                    "claim": ce["claim"],
                    "evidence": ce["evidence"],
                    "label_id": ce["label_id"],
                    "label": ce["gold_label"],
                })
            ce_ds = FeverGoldDataset(ce_items)

            # Augment training loader
            combined_ds = ConcatDataset([train_ds, ce_ds])
            train_loader = _build_dataloader(
                combined_ds, tokenizer, batch_size, max_length,
                shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                seed=seed + cegis_round,
            )

            if cegis_log is not None:
                cegis_log.append({
                    "round": cegis_round,
                    "n_counterexamples": n_ce,
                    "converged": False,
                })

        round_label = f"R{cegis_round}/" if mode == "cegis" else ""

        # Inner training loop
        for epoch in range(epochs):
            if nan_abort:
                break

            model.train()
            epoch_loss = 0.0
            epoch_constraint_loss = 0.0
            n_steps = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    result = model(input_ids, attention_mask, labels)
                    loss_task = result["loss"]

                    # Constraint loss (for soft/lagrangian/cegis/gated modes)
                    loss_constraint = torch.tensor(0.0, device=device)
                    constraint_info = {}
                    if mode in ("soft", "lagrangian", "cegis"):
                        probs = result["probs"]
                        facts = extract_batch_facts(
                            batch["claims"], batch["evidences"],
                        )
                        loss_constraint, constraint_info = fever_constraint_loss(
                            probs[:, LABEL2ID["SUPPORTS"]],
                            probs[:, LABEL2ID["REFUTES"]],
                            probs[:, LABEL2ID["NOT ENOUGH INFO"]],
                            facts,
                            weights=constraint_weights,
                        )
                    elif mode == "gated":
                        probs = result["probs"]
                        facts = extract_batch_facts(
                            batch["claims"], batch["evidences"],
                        )
                        loss_constraint, constraint_info = gated_fever_constraint_loss(
                            probs[:, LABEL2ID["SUPPORTS"]],
                            probs[:, LABEL2ID["REFUTES"]],
                            probs[:, LABEL2ID["NOT ENOUGH INFO"]],
                            facts,
                            gate=constraint_gate,
                            base_weights=constraint_weights,
                        )

                    # Combine losses
                    if mode == "neural":
                        loss = loss_task
                    elif mode == "soft":
                        loss = loss_task + lag_state.lam * loss_constraint
                    elif mode == "gated":
                        # ECCG: gated constraint loss with Lagrangian weighting
                        loss = lagrangian_loss(loss_task, loss_constraint, lag_state)
                    elif mode in ("lagrangian", "cegis"):
                        loss = lagrangian_loss(loss_task, loss_constraint, lag_state)
                    else:
                        loss = loss_task

                    loss = loss / grad_accum_steps

                # Backward
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # NaN guard
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss at epoch {epoch}, step {step}")
                    nan_abort = True
                    break

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

                    # Step-level evaluation (uses dev_sample_loader for speed)
                    if (eval_strategy == "steps"
                            and eval_every > 0
                            and global_step % eval_every == 0):
                        dev_metrics = _eval_split(model, dev_sample_loader, device)
                        dev_acc = dev_metrics["accuracy"]
                        lam_str = f" λ={lag_state.lam:.4f}" if mode != "neural" else ""
                        print(
                            f"  {round_label}Step {global_step}: "
                            f"loss={loss.item()*grad_accum_steps:.4f}"
                            f"{lam_str} | dev_acc={dev_acc:.4f} "
                            f"ECE={dev_metrics['ece']:.4f}"
                        )
                        train_log.append({
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "cegis_round": cegis_round if mode == "cegis" else None,
                            "train_loss": round(loss.item() * grad_accum_steps, 4),
                            "constraint_loss": round(loss_constraint.item(), 4),
                            "lambda": round(lag_state.lam, 4),
                            "dev_accuracy": dev_acc,
                            "dev_ece": dev_metrics["ece"],
                            "dev_brier": dev_metrics["brier"],
                        })
                        if dev_acc > best_dev_acc:
                            best_dev_acc = dev_acc
                            patience_counter = 0
                            ckpt_dir = os.path.join(outdir, "ckpt")
                            os.makedirs(ckpt_dir, exist_ok=True)
                            _unwrap(model).model.save_pretrained(ckpt_dir)
                            tokenizer.save_pretrained(ckpt_dir)
                            torch.save(
                                {"global_step": global_step, "best_dev_acc": best_dev_acc, "mode": mode},
                                os.path.join(ckpt_dir, "training_state.pt"),
                            )
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                logger.info(f"Early stopping at step {global_step}")
                                nan_abort = True  # reuse flag to break outer loops
                                break
                        model.train()

                epoch_loss += loss.item() * grad_accum_steps
                epoch_constraint_loss += loss_constraint.item()
                n_steps += 1

            if nan_abort:
                break

            avg_loss = epoch_loss / max(1, n_steps)
            avg_constraint = epoch_constraint_loss / max(1, n_steps)

            # Update Lagrangian
            if mode in ("lagrangian", "cegis", "gated"):
                update_dual_variable(
                    lag_state, avg_constraint,
                    step=epoch + cegis_round * epochs,
                    loss_task=avg_loss,
                )

            # Evaluation (epoch-level)
            do_epoch_eval = (
                eval_strategy == "epoch"
                and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
            )
            # Also eval at end of epoch for step-strategy (last epoch only)
            if eval_strategy == "steps" and epoch == epochs - 1:
                do_epoch_eval = True
            if do_epoch_eval:
                dev_metrics = _eval_split(model, dev_loader, device)
                dev_acc = dev_metrics["accuracy"]

                lam_str = f" λ={lag_state.lam:.4f}" if mode != "neural" else ""
                print(
                    f"  {round_label}Epoch {epoch+1}/{epochs}: "
                    f"loss={avg_loss:.4f} constraint={avg_constraint:.4f}"
                    f"{lam_str} | dev_acc={dev_acc:.4f} ECE={dev_metrics['ece']:.4f}"
                )

                train_log.append({
                    "epoch": epoch + 1,
                    "cegis_round": cegis_round if mode == "cegis" else None,
                    "train_loss": round(avg_loss, 4),
                    "constraint_loss": round(avg_constraint, 4),
                    "lambda": round(lag_state.lam, 4),
                    "dev_accuracy": dev_acc,
                    "dev_ece": dev_metrics["ece"],
                    "dev_brier": dev_metrics["brier"],
                })

                # Early stopping
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    patience_counter = 0
                    # Save best model
                    ckpt_dir = os.path.join(outdir, "ckpt")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    _unwrap(model).model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save(
                        {"epoch": epoch, "best_dev_acc": best_dev_acc, "mode": mode},
                        os.path.join(ckpt_dir, "training_state.pt"),
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(
                    f"  {round_label}Epoch {epoch+1}/{epochs}: "
                    f"loss={avg_loss:.4f} constraint={avg_constraint:.4f}"
                )

    elapsed = time.time() - t0

    # ── Post-hoc temperature scaling ───────────────────────
    from eval.temperature_scaling import learn_temperature
    print(f"\n{'─'*40}")
    print("  Post-hoc temperature scaling (dev set)")
    print(f"{'─'*40}")
    optimal_T = learn_temperature(model, dev_loader, device)

    # ── Final evaluation ─────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Final evaluation on dev set")
    print(f"{'─'*40}")
    final_dev = _eval_split(model, dev_loader, device)
    print(f"  Label Accuracy ({evidence_mode.upper()} evidence): {final_dev['accuracy']:.4f}")
    print(f"  ECE: {final_dev['ece']:.4f}")
    print(f"  Brier: {final_dev['brier']:.4f}")
    for label, stats in final_dev.get("per_label", {}).items():
        print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

    # ── Held-out dev_test evaluation (if split exists) ───────
    final_dev_test = None
    if "dev_test" in splits and splits["dev_test"]:
        print(f"\n{'─'*40}")
        print("  Final evaluation on held-out dev_test")
        print(f"{'─'*40}")
        if evidence_mode == "gold":
            dev_test_ds = FeverGoldDataset(splits["dev_test"])
        else:
            dev_test_ds = FeverPipelineDataset(splits["dev_test"], dev_evidence)
        dev_test_loader = _build_dataloader(
            dev_test_ds, tokenizer, batch_size, max_length,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
            seed=seed,
        )
        final_dev_test = _eval_split(model, dev_test_loader, device)
        print(f"  Label Accuracy ({evidence_mode.upper()} evidence): {final_dev_test['accuracy']:.4f}")
        print(f"  ECE: {final_dev_test['ece']:.4f}")
        print(f"  Brier: {final_dev_test['brier']:.4f}")
        for label, stats in final_dev_test.get("per_label", {}).items():
            print(f"    {label}: acc={stats['accuracy']:.4f} (n={stats['count']})")

    # ── Build report ─────────────────────────────────────────
    report = {
        "mode": mode,
        "evidence_mode": evidence_mode,
        "model_name": model_name,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else 0,
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 2),
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
        "dev": final_dev,
        "dev_test": final_dev_test,
        "train_log": train_log,
        "final_lambda": round(lag_state.lam, 6),
        "temperature": round(optimal_T, 4),
    }
    if cegis_log is not None:
        report["cegis"] = {
            "rounds": cegis_log,
            "total_rounds": len(cegis_log),
            "converged": any(r.get("converged", False) for r in cegis_log),
        }

    # Save report
    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Save λ trajectory
    if mode in ("lagrangian", "cegis", "gated"):
        save_lambda_trajectory(lag_state, os.path.join(outdir, "lambda_trajectory.json"))

    # Save gate weights
    if constraint_gate is not None:
        gate_path = os.path.join(outdir, "ckpt", "constraint_gate.pt")
        os.makedirs(os.path.dirname(gate_path), exist_ok=True)
        torch.save(_unwrap(constraint_gate).state_dict(), gate_path)
        logger.info(f"Constraint gate saved to {gate_path}")

    # Save training log
    with open(os.path.join(outdir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best dev accuracy: {best_dev_acc:.4f}")
    print(f"  Output: {outdir}")

    return report
