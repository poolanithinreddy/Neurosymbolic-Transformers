"""FEVER training loop with NST constraint integration.

Supports four modes (matching multi-digit/kinship pattern):
  1. neural:     pure DeBERTa cross-entropy (baseline)
  2. soft:       cross-entropy + fixed-weight constraint loss
  3. lagrangian: cross-entropy + adaptive Lagrangian constraint loss
  4. cegis:      Lagrangian + counterexample-guided outer loop

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
from symbolic.lagrangian import (
    LagrangianState, lagrangian_loss, update_dual_variable,
    save_lambda_trajectory,
)
from eval.calibration_metrics import expected_calibration_error, brier_score
from training.config_validation import load_and_validate_config

logger = logging.getLogger("train_fever")


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
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build DataLoader with tokenizing collate function."""
    collate = partial(fever_collate_fn, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
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
) -> dict:
    """Train FEVER NLI model with NST constraints.

    Modes (set in config):
      - neural: pure cross-entropy
      - soft: CE + fixed λ * constraint_loss
      - lagrangian: CE + adaptive Lagrangian
      - cegis: Lagrangian + counterexample-guided outer loop

    Returns:
        Report dict with train/dev metrics.
    """
    # ── Load config ──────────────────────────────────────────
    cfg = load_and_validate_config(config_path)
    mode = cfg.get("mode", "neural")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _auto_device(cfg.get("device", "auto"))
    outdir = outdir_override or cfg.get("outdir", f"outputs_fever_{mode}")
    os.makedirs(outdir, exist_ok=True)

    # Training hyperparameters
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 3))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    max_length = int(train_cfg.get("max_length", 256))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    eval_every = int(train_cfg.get("eval_every", 1))
    patience = int(train_cfg.get("patience", 5))
    num_workers = int(train_cfg.get("num_workers", 0))
    fp16 = train_cfg.get("fp16", device == "cuda")

    # Model
    model_name = cfg.get("model_name", "microsoft/deberta-v3-base")

    # Data
    data_cfg = cfg.get("data", {})
    max_train = data_cfg.get("max_train", None)
    max_dev = data_cfg.get("max_dev", None)
    evidence_mode = data_cfg.get("evidence_mode", "gold")  # "gold" or "pipeline"
    cache_dir = data_cfg.get("cache_dir", None)

    # Constraint config
    constraint_cfg = cfg.get("constraints", {})
    constraint_lambda = float(constraint_cfg.get("lambda", 0.1))
    constraint_weights = constraint_cfg.get("weights", {})

    # Lagrangian config
    lag_cfg = cfg.get("lagrangian", {})
    lag_epsilon = float(lag_cfg.get("epsilon", 0.05))
    lag_alpha = float(lag_cfg.get("alpha", 0.01))
    lag_rho = float(lag_cfg.get("rho", 1.0))
    lag_lam_max = float(lag_cfg.get("lam_max", 10.0))

    # CEGIS config
    cegis_cfg = cfg.get("cegis", {})
    max_rounds = int(cegis_cfg.get("max_rounds", 5))
    max_counterexamples = int(cegis_cfg.get("max_counterexamples", 500))
    ce_oversample = int(cegis_cfg.get("ce_oversample", 3))

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading FEVER dataset...")
    splits = load_fever_splits(cache_dir=cache_dir, max_train=max_train, max_dev=max_dev)
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
    )
    model = FeverNLIWrapper(base_model, label_smoothing=label_smoothing).to(device)

    # ── DataLoaders ──────────────────────────────────────────
    pin_memory = device == "cuda"
    train_loader = _build_dataloader(
        train_ds, tokenizer, batch_size, max_length,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
    )
    dev_loader = _build_dataloader(
        dev_ds, tokenizer, batch_size, max_length,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    # ── Optimizer + Scheduler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
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
    use_amp = fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype = torch.float16 if use_amp else torch.float32

    # ── Training loop ────────────────────────────────────────
    best_dev_acc = 0.0
    patience_counter = 0
    train_log = []
    nan_abort = False
    ce_buffer_ds = None  # For CEGIS

    n_rounds = max_rounds if mode == "cegis" else 1
    cegis_log = [] if mode == "cegis" else None

    print(f"\n{'='*60}")
    print(f"  FEVER Training: mode={mode}, model={model_name}")
    print(f"  epochs={epochs}, bs={batch_size}, lr={lr}, device={device}")
    print(f"  evidence_mode={evidence_mode}, fp16={use_amp}")
    print(f"  total_steps={total_steps}, warmup={warmup_steps}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for cegis_round in range(n_rounds):
        if nan_abort:
            break

        # CEGIS: augment training data with counterexamples
        if mode == "cegis" and cegis_round > 0:
            counterexamples = _mine_counterexamples(
                model, dev_loader, device, max_ce=max_counterexamples,
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
            )

            if cegis_log is not None:
                cegis_log.append({
                    "round": cegis_round,
                    "n_counterexamples": n_ce,
                    "converged": False,
                })

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

                    # Constraint loss (for soft/lagrangian/cegis modes)
                    loss_constraint = torch.tensor(0.0, device=device)
                    if mode in ("soft", "lagrangian", "cegis"):
                        probs = result["probs"]
                        facts = extract_batch_facts(
                            batch["claims"], batch["evidences"],
                        )
                        loss_constraint, _ = fever_constraint_loss(
                            probs[:, LABEL2ID["SUPPORTS"]],
                            probs[:, LABEL2ID["REFUTES"]],
                            probs[:, LABEL2ID["NOT ENOUGH INFO"]],
                            facts,
                            weights=constraint_weights,
                        )

                    # Combine losses
                    if mode == "neural":
                        loss = loss_task
                    elif mode == "soft":
                        loss = loss_task + lag_state.lam * loss_constraint
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * grad_accum_steps
                epoch_constraint_loss += loss_constraint.item()
                n_steps += 1

            if nan_abort:
                break

            avg_loss = epoch_loss / max(1, n_steps)
            avg_constraint = epoch_constraint_loss / max(1, n_steps)

            # Update Lagrangian
            if mode in ("lagrangian", "cegis"):
                update_dual_variable(
                    lag_state, avg_constraint,
                    step=epoch + cegis_round * epochs,
                    loss_task=avg_loss,
                )

            # Evaluation
            round_label = f"R{cegis_round}/" if mode == "cegis" else ""
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
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
                    model.model.save_pretrained(ckpt_dir)
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

    # ── Build report ─────────────────────────────────────────
    report = {
        "mode": mode,
        "evidence_mode": evidence_mode,
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "best_dev_acc": round(best_dev_acc, 4),
        "nan_abort": nan_abort,
        "dev": final_dev,
        "train_log": train_log,
        "final_lambda": round(lag_state.lam, 6),
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
    if mode in ("lagrangian", "cegis"):
        save_lambda_trajectory(lag_state, os.path.join(outdir, "lambda_trajectory.json"))

    # Save training log
    with open(os.path.join(outdir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best dev accuracy: {best_dev_acc:.4f}")
    print(f"  Output: {outdir}")

    return report
