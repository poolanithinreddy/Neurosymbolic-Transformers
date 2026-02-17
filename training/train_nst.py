"""Training loop for the neuro-symbolic digit addition experiment.

Supports four ablation modes:
- "neural":      pure cross-entropy training (digit + sum MLP).
- "soft":        cross-entropy + differentiable arithmetic constraint loss (λ-warmup).
- "hard":        same as soft during training; Z3 repair at inference.
- "lagrangian":  augmented Lagrangian with learned dual variable λ.

Also supports CAGrad multi-task gradient balancing (optional).
"""

import json
import math
import os
import random
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

# Ensure project root is importable
_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from data.digit_addition import DigitAdditionDataset
from models.nst_model import NSTDigitAddModel
from symbolic.constraint_solver import constraint_satisfaction_rate
from symbolic.lagrangian import (
    LagrangianState,
    lagrangian_loss,
    save_lambda_trajectory,
    update_dual_variable,
)
from training.cagrad import cagrad as cagrad_merge


def set_seed(s: int):
    """Set random seeds for reproducibility."""
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _auto_device(preferred: str | None = None) -> str:
    """Auto-select best available device."""
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def _lambda_warmup(step: int, warmup_steps: int, target_lambda: float) -> float:
    """Linear warmup schedule for constraint loss weight."""
    if warmup_steps <= 0:
        return target_lambda
    if step >= warmup_steps:
        return target_lambda
    return target_lambda * (step / warmup_steps)


def flatten_grads(params):
    """Flatten parameter gradients into a single vector."""
    grads = []
    shapes = []
    for p in params:
        if p.grad is None:
            shapes.append(None)
            grads.append(torch.zeros(0, device=p.device))
            continue
        g = p.grad.detach().reshape(-1)
        shapes.append(p.grad.shape)
        grads.append(g)
    if grads:
        return torch.cat([g for g in grads if g.numel() > 0], dim=0), shapes
    return torch.tensor([], device=params[0].device), shapes


def set_grads(params, flat_grad, shapes):
    """Restore flat gradient vector back to parameter .grad attributes."""
    idx = 0
    for p, shp in zip(params, shapes):
        if shp is None:
            continue
        n = math.prod(shp)
        g = flat_grad[idx : idx + n].reshape(shp)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        idx += n


def collate_fn(batch):
    """Custom collate for DigitAdditionDataset."""
    return {
        "img_a": torch.stack([b["img_a"] for b in batch]),
        "img_b": torch.stack([b["img_b"] for b in batch]),
        "digit_a": torch.stack([b["digit_a"] for b in batch]),
        "digit_b": torch.stack([b["digit_b"] for b in batch]),
        "sum": torch.stack([b["sum"] for b in batch]),
    }


def evaluate_model(model, dataloader, device, use_hard=False):
    """Evaluate model on a dataloader.

    Returns:
        Dict with digit_acc_a, digit_acc_b, sum_acc, csr.
    """
    model.eval()
    correct_a = correct_b = correct_sum = total = 0
    csr_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            img_a = batch["img_a"].to(device)
            img_b = batch["img_b"].to(device)
            digit_a = batch["digit_a"].to(device)
            digit_b = batch["digit_b"].to(device)
            sum_target = batch["sum"].to(device)

            preds = model.predict(img_a, img_b, use_hard_constraints=use_hard)

            correct_a += (preds["pred_a"] == digit_a).sum().item()
            correct_b += (preds["pred_b"] == digit_b).sum().item()
            correct_sum += (preds["pred_sum"] == sum_target).sum().item()
            total += digit_a.size(0)
            csr_total += preds["csr"]
            n_batches += 1

    model.train()
    return {
        "digit_acc_a": correct_a / max(1, total),
        "digit_acc_b": correct_b / max(1, total),
        "sum_acc": correct_sum / max(1, total),
        "csr": csr_total / max(1, n_batches),
        "n_samples": total,
    }


def train(cfg_path: str, outdir_override: str | None = None):
    """Main training function.

    Args:
        cfg_path: path to YAML config file.
        outdir_override: override for output directory.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = _auto_device(cfg.get("device", "auto"))
    print(f"[train_nst] Device: {device}, Seed: {seed}")

    # Mode: neural / soft / hard / lagrangian
    mode = cfg.get("mode", "soft")
    print(f"[train_nst] Mode: {mode}")

    # Data
    data_cfg = cfg.get("data", {})
    n_train = int(data_cfg.get("n_train", 10000))
    n_test = int(data_cfg.get("n_test", 2000))
    comp_threshold = int(data_cfg.get("comp_threshold", 9))
    img_size = int(data_cfg.get("img_size", 28))
    noise_std = float(data_cfg.get("noise_std", 0.15))

    train_ds = DigitAdditionDataset(
        "train", n_samples=n_train, comp_threshold=comp_threshold,
        img_size=img_size, noise_std=noise_std, seed=seed,
    )
    iid_ds = DigitAdditionDataset(
        "iid_test", n_samples=n_test, comp_threshold=comp_threshold,
        img_size=img_size, noise_std=noise_std, seed=seed + 1,
    )
    comp_ds = DigitAdditionDataset(
        "comp_test", n_samples=n_test, comp_threshold=comp_threshold,
        img_size=img_size, noise_std=noise_std, seed=seed + 2,
    )

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-3))
    warmup_steps = int(train_cfg.get("warmup_steps", 200))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    iid_loader = DataLoader(iid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    comp_loader = DataLoader(comp_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model_mode = mode
    if mode == "hard":
        model_mode = "soft"
    elif mode == "lagrangian":
        model_mode = "lagrangian"
    model = NSTDigitAddModel(mode=model_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Logic config
    logic_cfg = cfg.get("logic", {})
    target_lambda = float(logic_cfg.get("lambda", 0.5))
    cagrad_c = float(logic_cfg.get("cagrad_c", 0.0))

    # Lagrangian config
    lagrangian_cfg = cfg.get("lagrangian", {})
    lag_state = None
    if mode == "lagrangian":
        lag_state = LagrangianState(
            lam=float(lagrangian_cfg.get("lambda_init", 0.0)),
            epsilon=float(lagrangian_cfg.get("epsilon", 0.05)),
            alpha=float(lagrangian_cfg.get("alpha", 0.01)),
            rho=float(lagrangian_cfg.get("rho", 1.0)),
            lam_max=float(lagrangian_cfg.get("lambda_max", 10.0)),
        )

    # Training loop
    outdir = outdir_override or cfg.get("io", {}).get("out_dir", "outputs_digit_add")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "train_log.jsonl")

    global_step = 0
    best_sum_acc = 0.0
    start_time = time.time()

    print(f"[train_nst] Training for {epochs} epochs, {len(train_loader)} batches/epoch")
    if mode == "lagrangian":
        print(f"[train_nst] Lagrangian: ε={lag_state.epsilon}, α={lag_state.alpha}, ρ={lag_state.rho}")
    else:
        print(f"[train_nst] Lambda target: {target_lambda}, CAGrad c: {cagrad_c}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_digit_loss = 0.0
        epoch_constraint_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            img_a = batch["img_a"].to(device)
            img_b = batch["img_b"].to(device)
            digit_a = batch["digit_a"].to(device)
            digit_b = batch["digit_b"].to(device)
            sum_target = batch["sum"].to(device)

            result = model(img_a, img_b, digit_a, digit_b, sum_target)

            loss_digit = result["loss_digit"]
            loss_constraint = result.get("loss_constraint", torch.tensor(0.0, device=device))

            if mode == "lagrangian":
                # Augmented Lagrangian: learned dual variable
                lam = lag_state.lam
                total_loss = lagrangian_loss(loss_digit, loss_constraint, lag_state)
            elif mode == "neural":
                lam = 0.0
                total_loss = result["loss_total"]
            else:
                lam = _lambda_warmup(global_step, warmup_steps, target_lambda)
                total_loss = loss_digit + lam * loss_constraint

            if cagrad_c > 0 and mode != "neural":
                # CAGrad: balance digit loss vs constraint loss
                params = [p for p in model.parameters() if p.requires_grad]

                optimizer.zero_grad()
                loss_digit.backward(retain_graph=True)
                g_task, shapes = flatten_grads(params)

                optimizer.zero_grad()
                (lam * loss_constraint).backward(retain_graph=False)
                g_logic, _ = flatten_grads(params)

                g_comb = cagrad_merge([g_task, g_logic], c=cagrad_c)
                optimizer.zero_grad()
                set_grads(params, g_comb, shapes)
                optimizer.step()
            else:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            epoch_loss += total_loss.item()
            epoch_digit_loss += loss_digit.item()
            epoch_constraint_loss += loss_constraint.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_digit = epoch_digit_loss / max(1, n_batches)
        avg_constraint = epoch_constraint_loss / max(1, n_batches)

        # Lagrangian dual variable update (per epoch)
        if mode == "lagrangian" and lag_state is not None:
            update_dual_variable(
                lag_state, avg_constraint,
                step=epoch + 1, loss_task=avg_digit,
            )

        # Evaluate
        use_hard = mode == "hard"
        iid_metrics = evaluate_model(model, iid_loader, device, use_hard=use_hard)
        comp_metrics = evaluate_model(model, comp_loader, device, use_hard=use_hard)

        elapsed = time.time() - start_time
        log_entry = {
            "epoch": epoch + 1,
            "step": global_step,
            "loss": round(avg_loss, 4),
            "digit_loss": round(avg_digit, 4),
            "constraint_loss": round(avg_constraint, 4),
            "lambda": round(lam, 4),
            "iid": {k: round(v, 4) for k, v in iid_metrics.items()},
            "comp": {k: round(v, 4) for k, v in comp_metrics.items()},
            "elapsed_s": round(elapsed, 1),
        }

        print(
            f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
            f"IID sum_acc={iid_metrics['sum_acc']:.3f} CSR={iid_metrics['csr']:.3f} | "
            f"COMP sum_acc={comp_metrics['sum_acc']:.3f} CSR={comp_metrics['csr']:.3f}"
        )

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save best model
        if iid_metrics["sum_acc"] > best_sum_acc:
            best_sum_acc = iid_metrics["sum_acc"]
            ckpt_dir = os.path.join(outdir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_data = {
                "model_state_dict": model.state_dict(),
                "mode": mode,
                "epoch": epoch + 1,
                "step": global_step,
                "iid_metrics": iid_metrics,
                "comp_metrics": comp_metrics,
                "config": cfg,
            }
            if lag_state is not None:
                ckpt_data["lagrangian_state"] = lag_state.to_dict()
            torch.save(ckpt_data, os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  → Saved best model (IID sum_acc={best_sum_acc:.4f})")

    # Save final model
    final_dir = os.path.join(outdir, "ckpt")
    os.makedirs(final_dir, exist_ok=True)
    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "mode": mode,
        "epoch": epochs,
        "step": global_step,
        "config": cfg,
    }
    if lag_state is not None:
        final_ckpt["lagrangian_state"] = lag_state.to_dict()
    torch.save(final_ckpt, os.path.join(final_dir, "final_model.pt"))

    # Save λ trajectory if Lagrangian
    if lag_state is not None:
        save_lambda_trajectory(lag_state, os.path.join(outdir, "lambda_trajectory.json"))

    # Final evaluation report
    final_report = {
        "mode": mode,
        "epochs": epochs,
        "best_iid_sum_acc": round(best_sum_acc, 4),
        "final_iid": {k: round(v, 4) for k, v in iid_metrics.items()},
        "final_comp": {k: round(v, 4) for k, v in comp_metrics.items()},
        "elapsed_s": round(time.time() - start_time, 1),
    }
    if lag_state is not None:
        final_report["final_lambda"] = round(lag_state.lam, 6)
        final_report["price_of_logic"] = round(lag_state.lam, 6)
    report_path = os.path.join(outdir, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[train_nst] Final report saved to {report_path}")
    print(json.dumps(final_report, indent=2))

    return final_report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train NST digit-addition model")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--outdir", default=None, help="Override output directory")
    args = ap.parse_args()
    train(args.config, outdir_override=args.outdir)
