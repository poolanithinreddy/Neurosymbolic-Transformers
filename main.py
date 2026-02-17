#!/usr/bin/env python3
"""NST: Neuro-Symbolic Transformers — Unified CLI entry point.

Usage:
    # Train digit-addition (soft constraints)
    python main.py train --config configs/digit_add_soft.yaml

    # Train digit-addition (Lagrangian adaptive)
    python main.py train --config configs/digit_add_lagrangian.yaml

    # Evaluate a checkpoint
    python main.py eval --ckpt outputs_digit_add_soft/ckpt/best_model.pt --hard

    # Generate dataset statistics
    python main.py data-stats --threshold 9

    # Train FEVER (original pipeline)
    python main.py train-fever --config configs/mac_quick.yaml

    # Train kinship relational reasoning
    python main.py train-kinship --config configs/kinship_lagrangian.yaml

    # Run all ablations
    python main.py ablation

    # Generate results tables
    python main.py results --format markdown
"""

import argparse
import json
import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def cmd_train(args):
    """Train digit-addition model."""
    from training.train_nst import train
    train(args.config, outdir_override=args.outdir)


def cmd_eval(args):
    """Evaluate digit-addition model."""
    from eval.eval_nst import evaluate_full, print_results_table, _auto_device
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
        print(f"Report saved to {args.report}")


def cmd_data_stats(args):
    """Print dataset statistics."""
    from data.digit_addition import generate_stats
    generate_stats(seed=args.seed, n_train=args.n_train, n_test=args.n_test, threshold=args.threshold)


def cmd_train_fever(args):
    """Train FEVER model (original pipeline)."""
    from training.train import train_one
    train_one(args.config, task="fever", outdir_override=args.outdir)


def cmd_train_kinship(args):
    """Train kinship relational-reasoning model."""
    from training.train_kinship import train_kinship
    train_kinship(args.config, outdir_override=args.outdir)


def cmd_train_multi_digit(args):
    """Train multi-digit addition model (neural/soft/lagrangian)."""
    from training.train_multi_digit import train_multi_digit
    train_multi_digit(args.config, outdir_override=args.outdir)


def cmd_train_cegis(args):
    """Train multi-digit addition model with Neural CEGIS."""
    from training.cegis import train_multi_digit_cegis
    train_multi_digit_cegis(args.config, outdir_override=args.outdir)


def cmd_multi_digit_stats(args):
    """Print multi-digit addition dataset statistics."""
    from data.multi_digit_addition import generate_stats
    generate_stats(seed=args.seed, n_train=args.n_train, n_test=args.n_test)


def cmd_kinship_stats(args):
    """Print kinship dataset statistics."""
    from data.kinship import KinshipDataset
    ds_train = KinshipDataset(split="train", n_samples=args.n_train, seed=args.seed)
    ds_test  = KinshipDataset(split="comp_test", n_samples=args.n_test, seed=args.seed)
    print("=== Kinship Dataset Statistics ===")
    print(f"  Train samples : {len(ds_train)}")
    print(f"  Comp-test     : {len(ds_test)}")
    print(f"  Train chains  : depth 1–3")
    print(f"  Comp chains   : depth 4–5")
    # relation distribution
    from collections import Counter
    train_rels = Counter(s.answer for s in ds_train.samples)
    test_rels  = Counter(s.answer for s in ds_test.samples)
    print("\n  Train relation distribution:")
    for rel, cnt in sorted(train_rels.items(), key=lambda x: -x[1]):
        print(f"    {rel:<16} {cnt:>5}  ({100*cnt/len(ds_train):.1f}%)")
    print(f"\n  Comp-test relation distribution:")
    for rel, cnt in sorted(test_rels.items(), key=lambda x: -x[1]):
        print(f"    {rel:<16} {cnt:>5}  ({100*cnt/len(ds_test):.1f}%)")


def cmd_results(args):
    """Generate results tables from output directories."""
    from results import load_reports, render_results
    dirs = args.dirs or [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("outputs_")
    ]
    if not dirs:
        print("No output directories found. Run training first.")
        return
    reports = load_reports(dirs)
    if not reports:
        print("No reports found in output directories.")
        return
    out = render_results(reports, fmt=args.format)
    print(out)
    if args.save:
        ext = ".tex" if args.format == "latex" else ".md"
        with open(args.save, "w") as f:
            f.write(out)
        print(f"\nSaved to {args.save}")


def cmd_ablation(args):
    """Run all ablation experiments sequentially."""
    from training.train_nst import train

    configs = {
        "Pure Neural": "configs/digit_add_neural.yaml",
        "NST-Soft": "configs/digit_add_soft.yaml",
        "NST-Hard": "configs/digit_add_hard.yaml",
        "NST-CAGrad": "configs/digit_add_cagrad.yaml",
        "NST-Lagrangian": "configs/digit_add_lagrangian.yaml",
    }

    reports = {}
    for name, cfg_path in configs.items():
        if not os.path.exists(cfg_path):
            print(f"[ablation] Config not found: {cfg_path}, skipping {name}")
            continue
        print(f"\n{'='*60}")
        print(f"  ABLATION: {name}")
        print(f"{'='*60}")
        report = train(cfg_path)
        reports[name] = report
        print()

    # Summary table
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    for name, rep in reports.items():
        iid = rep.get("final_iid", {})
        comp = rep.get("final_comp", {})
        lam = rep.get("final_lambda", "N/A")
        extra = f" λ*={lam}" if lam != "N/A" else ""
        print(
            f"{name:<20} | IID sum_acc={iid.get('sum_acc', 'N/A'):>7} "
            f"CSR={iid.get('csr', 'N/A'):>7} | "
            f"COMP sum_acc={comp.get('sum_acc', 'N/A'):>7} "
            f"CSR={comp.get('csr', 'N/A'):>7}{extra}"
        )

    # Save combined report
    outpath = "outputs_ablation_summary.json"
    with open(outpath, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nCombined report saved to {outpath}")


def main():
    parser = argparse.ArgumentParser(
        prog="nst",
        description="Neuro-Symbolic Transformers — unified CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    p_train = subparsers.add_parser("train", help="Train digit-addition model")
    p_train.add_argument("--config", required=True, help="YAML config path")
    p_train.add_argument("--outdir", default=None, help="Override output directory")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate digit-addition model")
    p_eval.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--n_test", type=int, default=2000)
    p_eval.add_argument("--threshold", type=int, default=9)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--hard", action="store_true", help="Use Z3 hard constraints")
    p_eval.add_argument("--batch_size", type=int, default=64)
    p_eval.add_argument("--report", default=None, help="Save JSON report to path")

    # data-stats
    p_data = subparsers.add_parser("data-stats", help="Print dataset statistics")
    p_data.add_argument("--seed", type=int, default=42)
    p_data.add_argument("--n_train", type=int, default=5000)
    p_data.add_argument("--n_test", type=int, default=1000)
    p_data.add_argument("--threshold", type=int, default=9)

    # train-fever
    p_fever = subparsers.add_parser("train-fever", help="Train FEVER (original pipeline)")
    p_fever.add_argument("--config", required=True, help="YAML config path")
    p_fever.add_argument("--outdir", default=None)

    # train-kinship
    p_kin = subparsers.add_parser("train-kinship", help="Train kinship relational-reasoning model")
    p_kin.add_argument("--config", required=True, help="YAML config path")
    p_kin.add_argument("--outdir", default=None, help="Override output directory")

    # kinship-stats
    p_kstats = subparsers.add_parser("kinship-stats", help="Print kinship dataset statistics")
    p_kstats.add_argument("--seed", type=int, default=42)
    p_kstats.add_argument("--n_train", type=int, default=5000)
    p_kstats.add_argument("--n_test", type=int, default=2000)

    # train-multi-digit
    p_md = subparsers.add_parser("train-multi-digit", help="Train multi-digit addition model")
    p_md.add_argument("--config", required=True, help="YAML config path")
    p_md.add_argument("--outdir", default=None, help="Override output directory")

    # train-cegis
    p_cegis = subparsers.add_parser("train-cegis", help="Train multi-digit with Neural CEGIS")
    p_cegis.add_argument("--config", required=True, help="YAML config path")
    p_cegis.add_argument("--outdir", default=None, help="Override output directory")

    # multi-digit-stats
    p_mdstats = subparsers.add_parser("multi-digit-stats", help="Multi-digit dataset stats")
    p_mdstats.add_argument("--seed", type=int, default=42)
    p_mdstats.add_argument("--n_train", type=int, default=5000)
    p_mdstats.add_argument("--n_test", type=int, default=2000)

    # results
    p_res = subparsers.add_parser("results", help="Generate results tables from output dirs")
    p_res.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    p_res.add_argument("--dirs", nargs="*", help="Output directories (default: all outputs_*)")
    p_res.add_argument("--save", default=None, help="Save table to file")

    # ablation
    p_abl = subparsers.add_parser("ablation", help="Run all ablation experiments")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "train": cmd_train,
        "eval": cmd_eval,
        "data-stats": cmd_data_stats,
        "train-fever": cmd_train_fever,
        "train-kinship": cmd_train_kinship,
        "train-multi-digit": cmd_train_multi_digit,
        "train-cegis": cmd_train_cegis,
        "multi-digit-stats": cmd_multi_digit_stats,
        "kinship-stats": cmd_kinship_stats,
        "results": cmd_results,
        "ablation": cmd_ablation,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
