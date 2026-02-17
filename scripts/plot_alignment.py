#!/usr/bin/env python3
"""Alignment phase plot: the "Price of Logic" figure.

Generates a publication-ready dual-axis plot showing:
  - Left Y-axis: task accuracy (or sum accuracy) over epochs/rounds.
  - Right Y-axis: constraint violation rate (1 − CSR) or λ trajectory.
  - X-axis: training epochs (inner loop) or CEGIS rounds.
  - Shaded region marks the "alignment phase" where λ rises and CSR improves.

Reads from:
  - training logs (train_log.jsonl or report.json)
  - lambda_trajectory.json
  - cegis_log.json

Usage:
    python scripts/plot_alignment.py --logdir outputs_multi_digit_lagrangian
    python scripts/plot_alignment.py --logdir outputs_multi_digit_cegis --cegis
    python scripts/plot_alignment.py --logdir outputs_multi_digit_lagrangian --pdf figures/alignment.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("matplotlib is required for plotting. Install with: pip install matplotlib")
    sys.exit(1)

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


def load_lagrangian_log(logdir: str) -> dict | None:
    """Load λ trajectory from a Lagrangian training run."""
    path = os.path.join(logdir, "lambda_trajectory.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_training_log(logdir: str) -> list[dict] | None:
    """Load per-epoch training log."""
    # Try JSONL first
    jsonl_path = os.path.join(logdir, "train_log.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    # Try report.json with embedded history
    report_path = os.path.join(logdir, "report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            data = json.load(f)
        if "training_history" in data:
            return data["training_history"]

    return None


def load_cegis_log(logdir: str) -> dict | None:
    """Load CEGIS round-by-round log."""
    path = os.path.join(logdir, "cegis_log.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def find_alignment_phase(lambdas: list[float], threshold: float = 0.1) -> tuple[int, int]:
    """Identify the alignment phase: the interval where λ is rising fastest.

    Returns (start_epoch, end_epoch) of the steepest λ increase region.
    """
    if len(lambdas) < 3:
        return 0, len(lambdas) - 1

    diffs = np.diff(lambdas)
    # Find the region where λ is consistently increasing
    rising = diffs > threshold * max(abs(d) for d in diffs if d != 0)

    start = 0
    for i, r in enumerate(rising):
        if r:
            start = i
            break

    end = len(lambdas) - 1
    for i in range(len(rising) - 1, -1, -1):
        if rising[i]:
            end = i + 1
            break

    return start, end


def plot_lagrangian_alignment(
    logdir: str,
    save_png: str | None = None,
    save_pdf: str | None = None,
    title: str | None = None,
):
    """Plot accuracy + λ trajectory for a Lagrangian training run."""
    lam_log = load_lagrangian_log(logdir)
    train_log = load_training_log(logdir)

    if lam_log is None:
        print(f"No lambda_trajectory.json found in {logdir}")
        return
    if train_log is None:
        print(f"No training log found in {logdir}")
        return

    trajectory = lam_log.get("trajectory", [])
    epochs = [t.get("step", i + 1) for i, t in enumerate(trajectory)]
    lambdas = [t["lambda"] for t in trajectory]
    constraint_losses = [t.get("loss_logic", 0) for t in trajectory]

    # Extract CSR from training log
    csrs = [t.get("csr", 0) for t in train_log]
    # Align lengths
    n = min(len(epochs), len(csrs))
    epochs = epochs[:n]
    lambdas = lambdas[:n]
    csrs = csrs[:n]
    violation_rates = [1.0 - c for c in csrs]

    # Find alignment phase
    start, end = find_alignment_phase(lambdas)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Left axis: CSR (accuracy proxy)
    color_acc = "#2196F3"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Constraint Satisfaction Rate (CSR)", color=color_acc)
    ax1.plot(epochs, csrs, color=color_acc, linewidth=2, label="CSR", zorder=3)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(-0.05, 1.05)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Right axis: λ (price of logic)
    ax2 = ax1.twinx()
    color_lam = "#E91E63"
    ax2.set_ylabel("λ (Price of Logic)", color=color_lam)
    ax2.plot(epochs, lambdas, color=color_lam, linewidth=2, linestyle="--",
             label="λ", zorder=3)
    ax2.tick_params(axis="y", labelcolor=color_lam)

    # Shade alignment phase
    if start < end:
        ax1.axvspan(epochs[start], epochs[min(end, n - 1)],
                     alpha=0.12, color="#FFC107", label="Alignment phase", zorder=1)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(title or "Price of Logic: Alignment Phase")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_png:
        os.makedirs(os.path.dirname(save_png) or ".", exist_ok=True)
        fig.savefig(save_png, dpi=300)
        print(f"Saved PNG: {save_png}")
    if save_pdf:
        os.makedirs(os.path.dirname(save_pdf) or ".", exist_ok=True)
        fig.savefig(save_pdf)
        print(f"Saved PDF: {save_pdf}")

    plt.close(fig)


def plot_cegis_convergence(
    logdir: str,
    save_png: str | None = None,
    save_pdf: str | None = None,
    title: str | None = None,
):
    """Plot counterexample count + CSR + λ across CEGIS rounds."""
    cegis_log = load_cegis_log(logdir)
    if cegis_log is None:
        print(f"No cegis_log.json found in {logdir}")
        return

    rounds = cegis_log.get("rounds", [])
    if not rounds:
        print("Empty CEGIS log")
        return

    round_nums = [r["round"] for r in rounds]
    ce_counts = [r["n_counterexamples"] for r in rounds]
    csrs = [r["csr"] for r in rounds]
    lambdas = [r["lambda"] for r in rounds]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Left axis: counterexample count (bar chart)
    color_ce = "#FF7043"
    ax1.bar(round_nums, ce_counts, color=color_ce, alpha=0.6,
            label="Counterexamples", zorder=2)
    ax1.set_xlabel("CEGIS Round")
    ax1.set_ylabel("Counterexample Count", color=color_ce)
    ax1.tick_params(axis="y", labelcolor=color_ce)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Right axis: CSR
    ax2 = ax1.twinx()
    color_csr = "#2196F3"
    ax2.plot(round_nums, csrs, color=color_csr, linewidth=2, marker="o",
             markersize=5, label="CSR", zorder=3)
    ax2.set_ylabel("CSR / λ", color=color_csr)
    ax2.tick_params(axis="y", labelcolor=color_csr)
    ax2.set_ylim(-0.05, max(max(csrs) + 0.1, max(lambdas) + 0.5))

    # Also plot λ on right axis
    color_lam = "#E91E63"
    ax2.plot(round_nums, lambdas, color=color_lam, linewidth=2, linestyle="--",
             marker="s", markersize=4, label="λ", zorder=3)

    # Mark convergence
    if cegis_log.get("converged"):
        conv_round = cegis_log["convergence_round"]
        ax1.axvline(x=conv_round, color="#4CAF50", linestyle=":", linewidth=2,
                     label=f"Converged (round {conv_round})")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(title or "Neural CEGIS Convergence")
    ax1.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_png:
        os.makedirs(os.path.dirname(save_png) or ".", exist_ok=True)
        fig.savefig(save_png, dpi=300)
        print(f"Saved PNG: {save_png}")
    if save_pdf:
        os.makedirs(os.path.dirname(save_pdf) or ".", exist_ok=True)
        fig.savefig(save_pdf)
        print(f"Saved PDF: {save_pdf}")

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate alignment/convergence plots")
    ap.add_argument("--logdir", required=True, help="Output directory with training logs")
    ap.add_argument("--cegis", action="store_true", help="Plot CEGIS convergence instead")
    ap.add_argument("--png", default=None, help="Save PNG to this path")
    ap.add_argument("--pdf", default=None, help="Save PDF to this path")
    ap.add_argument("--outdir", default=None, help="Output directory (auto-generates PNG+PDF)")
    ap.add_argument("--title", default=None, help="Custom plot title")
    args = ap.parse_args()

    # --outdir sets both png and pdf with standard names
    base = args.outdir or args.logdir

    if args.cegis:
        png = args.png or os.path.join(base, "cegis_convergence.png")
        pdf = args.pdf or os.path.join(base, "cegis_convergence.pdf")
        plot_cegis_convergence(args.logdir, save_png=png, save_pdf=pdf, title=args.title)
    else:
        png = args.png or os.path.join(base, "alignment_phase.png")
        pdf = args.pdf or os.path.join(base, "alignment_phase.pdf")
        plot_lagrangian_alignment(args.logdir, save_png=png, save_pdf=pdf, title=args.title)


if __name__ == "__main__":
    main()
