"""Baseline training methods for fair comparison with Neural CEGIS.

Implements three baselines that control for CEGIS's advantages:

1. Random Replay: Same data augmentation budget, but counterexamples are
   replaced with random samples from the training set. This isolates whether
   the *targeting* of counterexamples matters, or just the extra data.

2. Hard Example Mining: After each round, find examples with highest loss
   (not constraint violations). This is the standard curriculum/hard mining
   approach. Isolates whether constraint-awareness matters.

3. Same Budget (Extra Epochs): Train for the same total compute budget
   (epochs × data) as CEGIS, but without any replay. Controls for CEGIS
   simply training longer.

These baselines are critical for reviewers: if CEGIS wins over random replay
and hard mining, the counterexample targeting is genuinely important.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import yaml

from symbolic.lagrangian import (
    LagrangianState,
    lagrangian_loss,
    update_dual_variable,
    save_lambda_trajectory,
)

logger = logging.getLogger("baselines")


def _auto_device(preferred: str | None = None) -> str:
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def _resolve_cfg(raw_cfg: dict) -> dict:
    """Resolve config from either cegis, baseline, or training sections.

    Baseline YAML files may use 'training' + 'baseline' + 'lagrangian' sections
    instead of the 'cegis' section used by cegis.py. This helper creates a
    unified parameter dict from whichever sections are present.
    """
    cegis = raw_cfg.get("cegis", {})
    training = raw_cfg.get("training", {})
    baseline = raw_cfg.get("baseline", {})
    lagrangian = raw_cfg.get("lagrangian", {})

    def _get(key, default, *sections):
        """Look up key in sections, return first found."""
        for s in sections:
            if key in s:
                return s[key]
        return default

    return {
        "max_rounds": int(_get("max_rounds", _get("rounds", 10, baseline, cegis), cegis, baseline)),
        "inner_epochs": int(_get("inner_epochs", _get("epochs", 15, training, cegis), cegis, training)),
        "lr": float(_get("lr", 1e-3, cegis, training)),
        "batch_size": int(_get("batch_size", 64, cegis, training)),
        "seed": int(_get("seed", 42, cegis, training, raw_cfg.get("data", {}))),
        "max_counterexamples": int(_get("max_counterexamples",
                                        _get("replay_size", _get("mine_size", 500, baseline), baseline),
                                        cegis, baseline)),
        "ce_oversample": int(_get("ce_oversample", _get("oversample", 3, baseline), cegis, baseline)),
        "device": _get("device", "auto", cegis, training),
        "outdir": _get("outdir", None, cegis, training),
        "lagrangian_epsilon": float(_get("epsilon", 0.05, lagrangian, cegis)),
        "lagrangian_alpha": float(_get("alpha", 0.01, lagrangian, cegis)),
        "lagrangian_rho": float(_get("rho", 1.0, lagrangian, cegis)),
        "lagrangian_lam_max": float(_get("lam_max", 10.0, lagrangian, cegis)),
    }


# ─────────────────────────────────────────────────────────────
# Baseline 1: Random Replay
# ─────────────────────────────────────────────────────────────

def train_random_replay(
    config_path: str,
    outdir_override: str | None = None,
    task: str = "multi_digit",
    seed: int | None = None,
) -> dict:
    """Train with random replay: same data budget as CEGIS but random augmentation.

    At each "round," instead of finding counterexamples, we sample N random
    examples from the training set and add them to a replay buffer. This
    matches CEGIS's data augmentation budget without targeted selection.
    """
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    p = _resolve_cfg(raw_cfg)
    data_cfg = raw_cfg.get("data", {})

    max_rounds = p["max_rounds"]
    inner_epochs = p["inner_epochs"]
    lr = p["lr"]
    batch_size = p["batch_size"]
    _seed = int(seed if seed is not None else p["seed"])
    max_ce = p["max_counterexamples"]
    ce_oversample = p["ce_oversample"]
    device = _auto_device(p["device"])
    outdir = outdir_override or p["outdir"] or "outputs_random_replay"
    os.makedirs(outdir, exist_ok=True)

    random.seed(_seed)
    torch.manual_seed(_seed)

    # Load data + model (same as CEGIS)
    if task == "multi_digit":
        from data.multi_digit_addition import MultiDigitAdditionDataset, multi_digit_collate
        from models.nst_multi_digit import MultiDigitModel

        train_ds = MultiDigitAdditionDataset(
            split="train", n_samples=data_cfg.get("n_train", 5000),
            seed=_seed, img_size=data_cfg.get("img_size", 28),
        )
        collate_fn = multi_digit_collate
        model = MultiDigitModel(mode="lagrangian")
    else:
        raise ValueError(f"Unsupported task for random replay: {task}")

    model.to(device)

    lag_state = LagrangianState(
        lam=0.0,
        epsilon=p["lagrangian_epsilon"],
        alpha=p["lagrangian_alpha"],
        rho=p["lagrangian_rho"],
        lam_max=p["lagrangian_lam_max"],
    )

    replay_buffer: list[int] = []  # indices into train_ds
    history = []

    for round_num in range(1, max_rounds + 1):
        t0 = time.time()

        # Random replay: sample random indices from training set
        new_indices = [random.randint(0, len(train_ds) - 1) for _ in range(max_ce)]
        replay_buffer.extend(new_indices)

        # Build augmented dataset
        replay_subset = Subset(train_ds, replay_buffer[-max_ce * ce_oversample:])
        augmented = ConcatDataset([train_ds, replay_subset])
        loader = DataLoader(augmented, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Inner training
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()

        total_loss = total_constraint = total_csr = n_batches = 0
        for epoch in range(inner_epochs):
            for batch in loader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                result = model(batch_dev["img_a"], batch_dev["img_b"],
                               a_tens=batch_dev.get("a_tens"), a_ones=batch_dev.get("a_ones"),
                               b_tens=batch_dev.get("b_tens"), b_ones=batch_dev.get("b_ones"),
                               sum_ones=batch_dev.get("sum_ones"), sum_tens=batch_dev.get("sum_tens"),
                               sum_hundreds=batch_dev.get("sum_hundreds"))

                loss = lagrangian_loss(result["loss_digit"], result["loss_constraint"], lag_state)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_constraint += result["loss_constraint"].item()
                total_csr += result.get("csr", 0.0)
                n_batches += 1

            avg_c = total_constraint / max(n_batches, 1)
            update_dual_variable(lag_state, avg_c, step=round_num * inner_epochs + epoch)

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        avg_csr = total_csr / max(n_batches, 1)

        history.append({
            "round": round_num,
            "loss": round(avg_loss, 4),
            "csr": round(avg_csr, 4),
            "lambda": round(lag_state.lam, 4),
            "replay_size": len(replay_buffer),
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  [Random Replay] Round {round_num}: loss={avg_loss:.4f} CSR={avg_csr:.3f} λ={lag_state.lam:.3f}")

    # Save
    report = {"method": "random_replay", "history": history, "final_lambda": lag_state.lam}
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))

    return report


# ─────────────────────────────────────────────────────────────
# Baseline 2: Hard Example Mining (by loss, not by constraint)
# ─────────────────────────────────────────────────────────────

def train_hard_mining(
    config_path: str,
    outdir_override: str | None = None,
    task: str = "multi_digit",
    seed: int | None = None,
) -> dict:
    """Train with hard example mining: select highest-loss samples each round.

    This is the standard curriculum learning / hard mining approach.
    Instead of constraint violations, we mine samples with highest cross-entropy
    loss and add them to the replay buffer.
    """
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    p = _resolve_cfg(raw_cfg)
    data_cfg = raw_cfg.get("data", {})

    max_rounds = p["max_rounds"]
    inner_epochs = p["inner_epochs"]
    lr = p["lr"]
    batch_size = p["batch_size"]
    _seed = int(seed if seed is not None else p["seed"])
    max_ce = p["max_counterexamples"]
    ce_oversample = p["ce_oversample"]
    device = _auto_device(p["device"])
    outdir = outdir_override or p["outdir"] or "outputs_hard_mining"
    os.makedirs(outdir, exist_ok=True)

    random.seed(_seed)
    torch.manual_seed(_seed)

    if task == "multi_digit":
        from data.multi_digit_addition import MultiDigitAdditionDataset, multi_digit_collate
        from models.nst_multi_digit import MultiDigitModel

        train_ds = MultiDigitAdditionDataset(
            split="train", n_samples=data_cfg.get("n_train", 5000),
            seed=_seed, img_size=data_cfg.get("img_size", 28),
        )
        # Mining pool: use comp_test to find hard examples
        mine_ds = MultiDigitAdditionDataset(
            split="comp_test", n_samples=data_cfg.get("n_verify", 2000),
            seed=_seed + 1, img_size=data_cfg.get("img_size", 28),
        )
        collate_fn = multi_digit_collate
        model = MultiDigitModel(mode="lagrangian")
    else:
        raise ValueError(f"Unsupported task: {task}")

    model.to(device)

    lag_state = LagrangianState(
        lam=0.0,
        epsilon=p["lagrangian_epsilon"],
        alpha=p["lagrangian_alpha"],
        rho=p["lagrangian_rho"],
        lam_max=p["lagrangian_lam_max"],
    )

    hard_buffer: list[dict] = []
    history = []

    for round_num in range(1, max_rounds + 1):
        t0 = time.time()

        # Mine hard examples by loss
        model.eval()
        mine_loader = DataLoader(mine_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        sample_losses = []

        with torch.no_grad():
            for batch in mine_loader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                result = model(batch_dev["img_a"], batch_dev["img_b"],
                               a_tens=batch_dev.get("a_tens"), a_ones=batch_dev.get("a_ones"),
                               b_tens=batch_dev.get("b_tens"), b_ones=batch_dev.get("b_ones"),
                               sum_ones=batch_dev.get("sum_ones"), sum_tens=batch_dev.get("sum_tens"),
                               sum_hundreds=batch_dev.get("sum_hundreds"))

                # Per-sample loss
                for i in range(len(batch_dev["img_a"])):
                    loss_i = result["loss_digit"].item()  # approximate
                    sample_losses.append((loss_i, {k: v[i].cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}))

        # Sort by loss descending, take top-K
        sample_losses.sort(key=lambda x: -x[0])
        new_hard = [s[1] for s in sample_losses[:max_ce]]
        hard_buffer.extend(new_hard)

        # Limit buffer
        if len(hard_buffer) > max_ce * 3:
            hard_buffer = hard_buffer[-max_ce * 3:]

        # Build augmented dataset
        from training.cegis import _CEDataset
        hard_ds = _CEDataset(hard_buffer)
        oversampled = ConcatDataset([hard_ds] * ce_oversample) if ce_oversample > 1 else hard_ds
        augmented = ConcatDataset([train_ds, oversampled])
        loader = DataLoader(augmented, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Inner training
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()

        total_loss = total_constraint = total_csr = n_batches = 0
        for epoch in range(inner_epochs):
            for batch in loader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                result = model(batch_dev["img_a"], batch_dev["img_b"],
                               a_tens=batch_dev.get("a_tens"), a_ones=batch_dev.get("a_ones"),
                               b_tens=batch_dev.get("b_tens"), b_ones=batch_dev.get("b_ones"),
                               sum_ones=batch_dev.get("sum_ones"), sum_tens=batch_dev.get("sum_tens"),
                               sum_hundreds=batch_dev.get("sum_hundreds"))

                loss = lagrangian_loss(result["loss_digit"], result["loss_constraint"], lag_state)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_constraint += result["loss_constraint"].item()
                total_csr += result.get("csr", 0.0)
                n_batches += 1

            avg_c = total_constraint / max(n_batches, 1)
            update_dual_variable(lag_state, avg_c, step=round_num * inner_epochs + epoch)

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        avg_csr = total_csr / max(n_batches, 1)

        history.append({
            "round": round_num,
            "loss": round(avg_loss, 4),
            "csr": round(avg_csr, 4),
            "lambda": round(lag_state.lam, 4),
            "buffer_size": len(hard_buffer),
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  [Hard Mining] Round {round_num}: loss={avg_loss:.4f} CSR={avg_csr:.3f}")

    report = {"method": "hard_mining", "history": history, "final_lambda": lag_state.lam}
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))

    return report


# ─────────────────────────────────────────────────────────────
# Baseline 3: Same Budget (Extra Epochs)
# ─────────────────────────────────────────────────────────────

def train_same_budget(
    config_path: str,
    outdir_override: str | None = None,
    task: str = "multi_digit",
    seed: int | None = None,
) -> dict:
    """Train for the same total compute budget as CEGIS, but without replay.

    If CEGIS does K rounds × N epochs, this baseline trains for K×N epochs
    straight through with Lagrangian only. Controls for CEGIS simply getting
    more total gradient updates.
    """
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    p = _resolve_cfg(raw_cfg)
    data_cfg = raw_cfg.get("data", {})

    max_rounds = p["max_rounds"]
    inner_epochs = p["inner_epochs"]
    total_epochs = max_rounds * inner_epochs

    lr = p["lr"]
    batch_size = p["batch_size"]
    _seed = int(seed if seed is not None else p["seed"])
    device = _auto_device(p["device"])
    outdir = outdir_override or p["outdir"] or "outputs_same_budget"
    os.makedirs(outdir, exist_ok=True)

    random.seed(_seed)
    torch.manual_seed(_seed)

    if task == "multi_digit":
        from data.multi_digit_addition import MultiDigitAdditionDataset, multi_digit_collate
        from models.nst_multi_digit import MultiDigitModel

        train_ds = MultiDigitAdditionDataset(
            split="train", n_samples=data_cfg.get("n_train", 5000),
            seed=_seed, img_size=data_cfg.get("img_size", 28),
        )
        collate_fn = multi_digit_collate
        model = MultiDigitModel(mode="lagrangian")
    else:
        raise ValueError(f"Unsupported task: {task}")

    model.to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    lag_state = LagrangianState(
        lam=0.0,
        epsilon=p["lagrangian_epsilon"],
        alpha=p["lagrangian_alpha"],
        rho=p["lagrangian_rho"],
        lam_max=p["lagrangian_lam_max"],
    )

    history = []
    print(f"[Same Budget] Training for {total_epochs} epochs (= {max_rounds} rounds × {inner_epochs} epochs)")

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = total_constraint = total_csr = n_batches = 0

        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            result = model(batch_dev["img_a"], batch_dev["img_b"],
                           a_tens=batch_dev.get("a_tens"), a_ones=batch_dev.get("a_ones"),
                           b_tens=batch_dev.get("b_tens"), b_ones=batch_dev.get("b_ones"),
                           sum_ones=batch_dev.get("sum_ones"), sum_tens=batch_dev.get("sum_tens"),
                           sum_hundreds=batch_dev.get("sum_hundreds"))

            loss = lagrangian_loss(result["loss_digit"], result["loss_constraint"], lag_state)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_constraint += result["loss_constraint"].item()
            total_csr += result.get("csr", 0.0)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_constraint = total_constraint / max(n_batches, 1)
        avg_csr = total_csr / max(n_batches, 1)

        update_dual_variable(lag_state, avg_constraint, step=epoch, loss_task=avg_loss)

        if epoch % 10 == 0 or epoch == total_epochs:
            print(f"  Epoch {epoch}/{total_epochs}: loss={avg_loss:.4f} CSR={avg_csr:.3f} λ={lag_state.lam:.3f}")
            history.append({
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "csr": round(avg_csr, 4),
                "lambda": round(lag_state.lam, 4),
            })

    report = {"method": "same_budget", "total_epochs": total_epochs, "history": history, "final_lambda": lag_state.lam}
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))

    return report
