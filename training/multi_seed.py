"""Multi-seed training runner for producing error bars.

Runs the same training pipeline across multiple seeds, collects
per-seed reports, and aggregates with mean ± std.

Usage:
    python -m training.multi_seed --task train --config configs/digit_add_lagrangian.yaml --seeds 42,43,44
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from results import aggregate_results


def run_multi_seed(
    train_fn: Callable[[str, str | None], dict],
    config_path: str,
    seeds: list[int],
    base_outdir: str = "outputs_multiseed",
    task_name: str = "experiment",
) -> dict:
    """Run a training function across multiple seeds and aggregate.

    Args:
        train_fn: function(config_path, outdir_override) → report dict.
        config_path: path to YAML config.
        seeds: list of random seeds.
        base_outdir: base output directory.
        task_name: name for logging.

    Returns:
        Dict with per-seed reports and aggregated results.
    """
    import yaml

    per_seed_reports = []
    seed_dirs = []

    print(f"\n{'='*60}")
    print(f"  MULTI-SEED RUN: {task_name}")
    print(f"  Seeds: {seeds}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}\n")

    for i, seed in enumerate(seeds):
        seed_outdir = os.path.join(base_outdir, f"seed_{seed}")
        os.makedirs(seed_outdir, exist_ok=True)

        # Patch config with this seed
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        cfg["seed"] = seed
        patched_config = os.path.join(seed_outdir, "config_seed.yaml")
        with open(patched_config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        print(f"\n--- Seed {seed} ({i+1}/{len(seeds)}) ---")
        t0 = time.time()

        try:
            report = train_fn(patched_config, outdir_override=seed_outdir)
            report["seed"] = seed
            per_seed_reports.append(report)
            seed_dirs.append(seed_outdir)
            elapsed = time.time() - t0
            print(f"  Seed {seed} done in {elapsed:.1f}s")
        except Exception as e:
            print(f"  Seed {seed} FAILED: {e}")
            per_seed_reports.append({"seed": seed, "error": str(e)})

    # Aggregate results
    valid_reports = [r for r in per_seed_reports if "error" not in r]
    aggregated = aggregate_results(valid_reports) if valid_reports else {}

    # Build summary
    summary = {
        "task": task_name,
        "config": config_path,
        "seeds": seeds,
        "n_seeds_total": len(seeds),
        "n_seeds_success": len(valid_reports),
        "per_seed": per_seed_reports,
        "aggregated": aggregated,
    }

    # Save
    os.makedirs(base_outdir, exist_ok=True)
    summary_path = os.path.join(base_outdir, "multi_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMulti-seed summary saved to {summary_path}")

    # Print aggregated
    if aggregated:
        print(f"\n{'='*60}")
        print("  AGGREGATED RESULTS (mean ± std)")
        print(f"{'='*60}")
        for metric, stats in sorted(aggregated.items()):
            print(f"  {metric:<30} {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['n']})")

    return summary


def _resolve_train_fn(task: str) -> Callable:
    """Get training function for a task name."""
    if task in ("train", "digit_add"):
        from training.train_nst import train
        return train
    elif task in ("train-kinship", "kinship"):
        from training.train_kinship import train_kinship
        return train_kinship
    elif task in ("train-multi-digit", "multi_digit"):
        from training.train_multi_digit import train_multi_digit
        return train_multi_digit
    elif task in ("train-cegis", "cegis"):
        from training.cegis import train_multi_digit_cegis
        return train_multi_digit_cegis
    elif task in ("train-kinship-cegis", "kinship_cegis"):
        from training.cegis import train_kinship_cegis
        return train_kinship_cegis
    elif task in ("train-fever-nst", "fever_nst", "fever"):
        from training.train_fever_nst import train_fever_nst
        return train_fever_nst
    elif task in ("train-fever-veri", "fever_veri"):
        from training.train_fever_veri import train_fever_veri
        return train_fever_veri
    elif task in ("train-fever-veri-v2", "fever_veri_v2"):
        from training.train_fever_veri_v2 import train_fever_veri_v2
        return train_fever_veri_v2
    else:
        raise ValueError(f"Unknown task: {task}")


def parse_seeds(seed_str: str) -> list[int]:
    """Parse comma-separated seeds string into list of ints."""
    return [int(s.strip()) for s in seed_str.split(",")]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Multi-seed training runner")
    ap.add_argument("--task", required=True,
                    choices=["train", "train-kinship", "train-multi-digit",
                             "train-cegis", "train-kinship-cegis",
                             "train-fever-nst"],
                    help="Training task to run")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    ap.add_argument("--outdir", default=None, help="Base output directory")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    train_fn = _resolve_train_fn(args.task)
    base_outdir = args.outdir or f"outputs_{args.task}_multiseed"

    run_multi_seed(
        train_fn=train_fn,
        config_path=args.config,
        seeds=seeds,
        base_outdir=base_outdir,
        task_name=args.task,
    )
