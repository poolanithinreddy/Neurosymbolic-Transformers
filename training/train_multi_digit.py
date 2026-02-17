"""Training loop for multi-digit addition (neural / soft / lagrangian baselines).

This is the standard training loop WITHOUT CEGIS, used for baselines:
  - neural: pure MLP, no constraints
  - soft: fixed-λ differentiable carry constraint
  - lagrangian: adaptive-λ augmented Lagrangian
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from data.multi_digit_addition import MultiDigitAdditionDataset, multi_digit_collate
from models.nst_multi_digit import MultiDigitModel
from symbolic.lagrangian import (
    LagrangianState,
    lagrangian_loss,
    save_lambda_trajectory,
    update_dual_variable,
)


def _auto_device(preferred: str | None = None) -> str:
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def train_multi_digit(config_path: str, outdir_override: str | None = None) -> dict:
    """Train multi-digit addition model (non-CEGIS baselines)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    mode = model_cfg.get("mode", "neural")
    epochs = train_cfg.get("epochs", 30)
    lr = train_cfg.get("lr", 1e-3)
    batch_size = train_cfg.get("batch_size", 64)
    seed = train_cfg.get("seed", 42)
    device = _auto_device(train_cfg.get("device", "auto"))
    outdir = outdir_override or train_cfg.get("outdir", f"outputs_multi_digit_{mode}")

    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(outdir, exist_ok=True)

    # Data
    img_size = data_cfg.get("img_size", 28)
    train_ds = MultiDigitAdditionDataset(
        split="train", n_samples=data_cfg.get("n_train", 5000),
        seed=seed, img_size=img_size,
    )
    iid_ds = MultiDigitAdditionDataset(
        split="iid_test", n_samples=data_cfg.get("n_test", 2000),
        seed=seed + 1, img_size=img_size,
    )
    comp_ds = MultiDigitAdditionDataset(
        split="comp_test", n_samples=data_cfg.get("n_test", 2000),
        seed=seed + 2, img_size=img_size,
    )
    hard_ds = MultiDigitAdditionDataset(
        split="hard_test", n_samples=data_cfg.get("n_test", 2000),
        seed=seed + 3, img_size=img_size,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=multi_digit_collate)

    # Model
    model = MultiDigitModel(mode=mode)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Lagrangian state (only used for lagrangian mode)
    lag_state = LagrangianState(
        lam=0.0,
        epsilon=train_cfg.get("lagrangian_epsilon", 0.05),
        alpha=train_cfg.get("lagrangian_alpha", 0.01),
        rho=train_cfg.get("lagrangian_rho", 1.0),
        lam_max=train_cfg.get("lagrangian_lam_max", 10.0),
    )

    lambda_constraint = train_cfg.get("lambda_constraint", 0.5)

    print(f"Training multi-digit addition [{mode}] on {device}")
    print(f"  Train: {len(train_ds)}, IID test: {len(iid_ds)}, Comp test: {len(comp_ds)}, Hard test: {len(hard_ds)}")

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_constraint = 0.0
        epoch_csr = 0.0
        n_batches = 0

        for batch in train_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            result = model(
                batch_dev["img_a"], batch_dev["img_b"],
                a_tens=batch_dev["a_tens"], a_ones=batch_dev["a_ones"],
                b_tens=batch_dev["b_tens"], b_ones=batch_dev["b_ones"],
                sum_ones=batch_dev["sum_ones"], sum_tens=batch_dev["sum_tens"],
                sum_hundreds=batch_dev["sum_hundreds"],
            )

            if mode == "lagrangian":
                loss = lagrangian_loss(result["loss_digit"], result["loss_constraint"], lag_state)
            elif mode == "soft":
                loss = result["loss_digit"] + lambda_constraint * result["loss_constraint"]
            else:
                loss = result["loss_total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_constraint += result.get("loss_constraint", torch.tensor(0.0)).item()
            epoch_csr += result.get("csr", 0.0)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_constraint = epoch_constraint / max(n_batches, 1)
        avg_csr = epoch_csr / max(n_batches, 1)

        if mode == "lagrangian":
            update_dual_variable(lag_state, avg_constraint, step=epoch, loss_task=avg_loss)

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            comp_metrics = _eval_split(model, comp_ds, device, batch_size)
            lam_str = f"λ={lag_state.lam:.3f}" if mode == "lagrangian" else ""
            print(
                f"  Epoch {epoch:>3}/{epochs}: loss={avg_loss:.4f} "
                f"CSR={avg_csr:.3f} comp_sum_acc={comp_metrics['sum_acc']:.3f} "
                f"comp_csr={comp_metrics['csr']:.3f} {lam_str}"
            )

        history.append({"epoch": epoch, "loss": avg_loss, "constraint": avg_constraint, "csr": avg_csr})

    # Final evaluation
    report = {"mode": mode, "training_history": history}
    for split_name, ds in [("iid_test", iid_ds), ("comp_test", comp_ds), ("hard_test", hard_ds)]:
        metrics = _eval_split(model, ds, device, batch_size)
        report[split_name] = metrics
        print(f"  {split_name}: sum_acc={metrics['sum_acc']:.4f}, CSR={metrics['csr']:.4f}")

    if mode == "lagrangian":
        report["final_lambda"] = lag_state.lam
        save_lambda_trajectory(lag_state, os.path.join(outdir, "lambda_trajectory.json"))

    # Save
    ckpt_dir = os.path.join(outdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


@torch.no_grad()
def _eval_split(model, ds, device, batch_size):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=multi_digit_collate, shuffle=False)
    correct_sum = 0
    total = 0
    csr_total = 0.0

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        result = model(batch_dev["img_a"], batch_dev["img_b"])

        pred_s = (
            result["probs_s_hund"].argmax(-1) * 100 +
            result["probs_s_tens"].argmax(-1) * 10 +
            result["probs_s_ones"].argmax(-1)
        )
        target_s = batch_dev["sum_hundreds"] * 100 + batch_dev["sum_tens"] * 10 + batch_dev["sum_ones"]
        correct_sum += (pred_s == target_s).sum().item()
        csr_total += result["csr"] * len(batch_dev["img_a"])
        total += len(batch_dev["img_a"])

    return {"sum_acc": correct_sum / max(total, 1), "csr": csr_total / max(total, 1), "n_samples": total}
