"""Evaluation module for neuro-symbolic digit addition experiments.

Loads a trained model checkpoint and evaluates on IID and compositional
test splits, reporting accuracy, constraint satisfaction rate, and
compositional gap.  Supports hard-constraint inference (Z3 repair).
"""

import json
import os
import sys

import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from data.digit_addition import DigitAdditionDataset
from models.nst_model import NSTDigitAddModel
from training.train_nst import collate_fn


def _auto_device(preferred: str | None = None) -> str:
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def load_model(ckpt_path: str, device: str = "cpu") -> NSTDigitAddModel:
    """Load model from checkpoint.

    Args:
        ckpt_path: path to .pt checkpoint file.
        device: target device.

    Returns:
        Loaded NSTDigitAddModel.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "soft")
    # For "hard" mode, the training mode is "soft"
    model_mode = "soft" if mode == "hard" else mode
    model = NSTDigitAddModel(mode=model_mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_split(
    model: NSTDigitAddModel,
    dataloader: DataLoader,
    device: str,
    use_hard: bool = False,
) -> dict:
    """Evaluate model on a single data split.

    Returns:
        Dict with digit_acc_a, digit_acc_b, sum_acc, csr, repair_rate, n.
    """
    correct_a = correct_b = correct_sum = total = 0
    csr_sum = 0.0
    repair_sum = 0.0
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
            csr_sum += preds["csr"]
            repair_sum += preds.get("repair_rate", 0.0)
            n_batches += 1

    return {
        "digit_acc_a": round(correct_a / max(1, total), 4),
        "digit_acc_b": round(correct_b / max(1, total), 4),
        "sum_acc": round(correct_sum / max(1, total), 4),
        "csr": round(csr_sum / max(1, n_batches), 4),
        "repair_rate": round(repair_sum / max(1, n_batches), 4),
        "n_samples": total,
    }


def evaluate_full(
    ckpt_path: str,
    device: str = "cpu",
    n_test: int = 2000,
    comp_threshold: int = 9,
    seed: int = 42,
    use_hard: bool = False,
    batch_size: int = 64,
) -> dict:
    """Full evaluation: IID + compositional + ablation comparison.

    Args:
        ckpt_path: path to model checkpoint.
        device: computation device.
        n_test: number of test samples per split.
        comp_threshold: compositional split threshold.
        seed: random seed for data generation.
        use_hard: whether to apply Z3 hard constraints.
        batch_size: evaluation batch size.

    Returns:
        Dict with iid, comp, and compositional_gap results.
    """
    model = load_model(ckpt_path, device)

    iid_ds = DigitAdditionDataset(
        "iid_test", n_samples=n_test, comp_threshold=comp_threshold, seed=seed + 1
    )
    comp_ds = DigitAdditionDataset(
        "comp_test", n_samples=n_test, comp_threshold=comp_threshold, seed=seed + 2
    )

    iid_loader = DataLoader(iid_ds, batch_size=batch_size, collate_fn=collate_fn)
    comp_loader = DataLoader(comp_ds, batch_size=batch_size, collate_fn=collate_fn)

    iid_results = evaluate_split(model, iid_loader, device, use_hard=use_hard)
    comp_results = evaluate_split(model, comp_loader, device, use_hard=use_hard)

    # Compositional gap
    gap = iid_results["sum_acc"] - comp_results["sum_acc"]

    report = {
        "ckpt": ckpt_path,
        "use_hard_constraints": use_hard,
        "iid": iid_results,
        "comp": comp_results,
        "compositional_gap": round(gap, 4),
    }

    return report


def print_results_table(reports: dict[str, dict]):
    """Print a formatted comparison table.

    Args:
        reports: dict mapping model name to evaluation report.
    """
    header = (
        f"{'Model':<16} | {'Digit Acc(IID)':>14} | {'Sum Acc(IID)':>12} | "
        f"{'CSR(IID)':>9} | {'Sum Acc(Comp)':>13} | {'CSR(Comp)':>10} | "
        f"{'Gap':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for name, rep in reports.items():
        iid = rep["iid"]
        comp = rep["comp"]
        da = (iid["digit_acc_a"] + iid["digit_acc_b"]) / 2
        print(
            f"{name:<16} | {da:>14.4f} | {iid['sum_acc']:>12.4f} | "
            f"{iid['csr']:>9.4f} | {comp['sum_acc']:>13.4f} | "
            f"{comp['csr']:>10.4f} | {rep['compositional_gap']:>6.4f}"
        )
    print(sep)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate NST digit-addition model")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    ap.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    ap.add_argument("--n_test", type=int, default=2000, help="Number of test samples")
    ap.add_argument("--threshold", type=int, default=9, help="Compositional split threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard", action="store_true", help="Apply Z3 hard constraints")
    ap.add_argument("--report", default=None, help="Path to save JSON report")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = _auto_device(args.device)
    report = evaluate_full(
        args.ckpt, device=device, n_test=args.n_test,
        comp_threshold=args.threshold, seed=args.seed,
        use_hard=args.hard, batch_size=args.batch_size,
    )

    print_results_table({"Model": report})

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.report}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
