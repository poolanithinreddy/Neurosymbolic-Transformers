"""Calibration evaluation CLI.

Loads a trained model checkpoint, collects predicted probabilities,
and computes ECE + Brier score using calibration_metrics.
"""

import argparse
import json
import os
import sys

import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from eval.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)


def evaluate_calibration(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """Compute full calibration report from probabilities and labels.

    Args:
        probs: [N, C] predicted probabilities (post-softmax).
        labels: [N] ground-truth class indices.
        n_bins: number of ECE bins.

    Returns:
        Dict with ECE, Brier, reliability diagram data.
    """
    ece, bin_data = expected_calibration_error(probs, labels, n_bins)
    bs = brier_score(probs, labels)
    rel_data = reliability_diagram_data(probs, labels, n_bins)

    return {
        "ece": ece,
        "brier": bs,
        "n_bins": n_bins,
        "n_samples": len(labels),
        "bin_data": bin_data,
        "reliability": rel_data,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate model calibration")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--report", required=True, help="Output report path")
    ap.add_argument("--task", default="digit_add", choices=["digit_add", "kinship"])
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # Load checkpoint and evaluate — requires a trained model
    # For now, save the config as a placeholder since model loading
    # depends on the task-specific model class
    rep = {
        "status": "requires-checkpoint",
        "task": args.task,
        "ckpt": args.ckpt,
        "n_bins": args.n_bins,
    }

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if "calibration" in ckpt:
            rep = ckpt["calibration"]
            rep["status"] = "loaded-from-checkpoint"

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "bin_data"}, indent=2))


if __name__ == "__main__":
    main()
