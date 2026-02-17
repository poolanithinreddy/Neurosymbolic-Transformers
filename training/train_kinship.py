"""Training loop for the kinship relational reasoning experiment.

Supports modes: neural, soft (fixed-λ), lagrangian.
"""

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

from data.kinship import KinshipDataset, kinship_collate_fn
from models.nst_kinship import KinshipTransformer
from symbolic.lagrangian import (
    LagrangianState,
    lagrangian_loss,
    save_lambda_trajectory,
    update_dual_variable,
)


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _auto_device(preferred=None):
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def evaluate_kinship(model, dataloader, device):
    """Evaluate kinship model."""
    model.eval()
    correct = total = 0
    csr_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            chain_lengths = batch["chain_lengths"]

            result = model(input_ids, labels=labels, chain_lengths=chain_lengths)
            preds = result["probs"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            csr_total += result.get("csr", 0.0)
            n_batches += 1

    model.train()
    return {
        "accuracy": correct / max(1, total),
        "csr": csr_total / max(1, n_batches),
        "n_samples": total,
    }


def train_kinship(cfg_path: str, outdir_override: str | None = None):
    """Main kinship training function."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = _auto_device(cfg.get("device", "auto"))
    mode = cfg.get("mode", "soft")
    print(f"[train_kinship] Device: {device}, Mode: {mode}, Seed: {seed}")

    # Data
    data_cfg = cfg.get("data", {})
    n_train = int(data_cfg.get("n_train", 5000))
    n_test = int(data_cfg.get("n_test", 1000))
    max_train_depth = int(data_cfg.get("max_train_depth", 3))
    max_test_depth = int(data_cfg.get("max_test_depth", 5))
    max_seq_len = int(data_cfg.get("max_seq_len", 256))

    train_ds = KinshipDataset(
        "train", n_train, max_train_depth, max_test_depth,
        max_seq_len=max_seq_len, seed=seed,
    )
    iid_ds = KinshipDataset(
        "iid_test", n_test, max_train_depth, max_test_depth,
        max_seq_len=max_seq_len, seed=seed + 1,
    )
    comp_ds = KinshipDataset(
        "comp_test", n_test, max_train_depth, max_test_depth,
        max_seq_len=max_seq_len, seed=seed + 2,
    )

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 15))
    lr = float(train_cfg.get("lr", 3e-4))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=kinship_collate_fn
    )
    iid_loader = DataLoader(
        iid_ds, batch_size=batch_size, shuffle=False, collate_fn=kinship_collate_fn
    )
    comp_loader = DataLoader(
        comp_ds, batch_size=batch_size, shuffle=False, collate_fn=kinship_collate_fn
    )

    # Model
    model_cfg = cfg.get("model", {})
    model = KinshipTransformer(
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        d_ff=int(model_cfg.get("d_ff", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_seq_len=max_seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Logic config
    logic_cfg = cfg.get("logic", {})
    target_lambda = float(logic_cfg.get("lambda", 0.5))

    # Lagrangian
    lag_state = None
    if mode == "lagrangian":
        lagrangian_cfg = cfg.get("lagrangian", {})
        lag_state = LagrangianState(
            lam=float(lagrangian_cfg.get("lambda_init", 0.0)),
            epsilon=float(lagrangian_cfg.get("epsilon", 0.05)),
            alpha=float(lagrangian_cfg.get("alpha", 0.01)),
            rho=float(lagrangian_cfg.get("rho", 1.0)),
            lam_max=float(lagrangian_cfg.get("lambda_max", 10.0)),
        )

    # Output
    outdir = outdir_override or cfg.get("io", {}).get("out_dir", "outputs_kinship")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "train_log.jsonl")

    best_acc = 0.0
    start_time = time.time()

    print(f"[train_kinship] {epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_constraint_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            chain_lengths = batch["chain_lengths"]

            result = model(input_ids, labels=labels, chain_lengths=chain_lengths)
            loss_task = result["loss_task"]
            loss_constraint = result.get("loss_constraint", torch.tensor(0.0, device=device))

            if mode == "neural":
                total_loss = loss_task
            elif mode == "lagrangian" and lag_state is not None:
                total_loss = lagrangian_loss(loss_task, loss_constraint, lag_state)
            else:
                total_loss = loss_task + target_lambda * loss_constraint

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_task_loss += loss_task.item()
            epoch_constraint_loss += loss_constraint.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_task = epoch_task_loss / max(1, n_batches)
        avg_constraint = epoch_constraint_loss / max(1, n_batches)

        # Lagrangian dual update
        if mode == "lagrangian" and lag_state is not None:
            update_dual_variable(lag_state, avg_constraint, step=epoch + 1, loss_task=avg_task)

        # Evaluate
        iid_m = evaluate_kinship(model, iid_loader, device)
        comp_m = evaluate_kinship(model, comp_loader, device)

        lam_val = lag_state.lam if lag_state else target_lambda if mode != "neural" else 0.0

        log_entry = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "task_loss": round(avg_task, 4),
            "constraint_loss": round(avg_constraint, 4),
            "lambda": round(lam_val, 4),
            "iid": {k: round(v, 4) for k, v in iid_m.items()},
            "comp": {k: round(v, 4) for k, v in comp_m.items()},
        }

        print(
            f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} λ={lam_val:.3f} | "
            f"IID acc={iid_m['accuracy']:.3f} CSR={iid_m['csr']:.3f} | "
            f"COMP acc={comp_m['accuracy']:.3f} CSR={comp_m['csr']:.3f}"
        )

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if iid_m["accuracy"] > best_acc:
            best_acc = iid_m["accuracy"]
            ckpt_dir = os.path.join(outdir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_data = {
                "model_state_dict": model.state_dict(),
                "mode": mode,
                "epoch": epoch + 1,
                "iid_metrics": iid_m,
                "comp_metrics": comp_m,
                "config": cfg,
            }
            if lag_state is not None:
                ckpt_data["lagrangian_state"] = lag_state.to_dict()
            torch.save(ckpt_data, os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  → Saved best (acc={best_acc:.4f})")

    # Save λ trajectory
    if lag_state is not None:
        save_lambda_trajectory(lag_state, os.path.join(outdir, "lambda_trajectory.json"))

    # Final report
    final_report = {
        "mode": mode,
        "epochs": epochs,
        "best_iid_acc": round(best_acc, 4),
        "final_iid": {k: round(v, 4) for k, v in iid_m.items()},
        "final_comp": {k: round(v, 4) for k, v in comp_m.items()},
        "elapsed_s": round(time.time() - start_time, 1),
    }
    if lag_state is not None:
        final_report["final_lambda"] = round(lag_state.lam, 6)

    report_path = os.path.join(outdir, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[train_kinship] Report: {report_path}")
    print(json.dumps(final_report, indent=2))

    return final_report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    train_kinship(args.config, args.outdir)
