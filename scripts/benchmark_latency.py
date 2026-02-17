#!/usr/bin/env python3
"""Inference latency benchmark for NST modes.

Measures wall-clock inference time for each constraint mode:
  - neural: pure MLP forward pass, no constraint computation.
  - soft: differentiable carry constraint (discrete convolution).
  - hard: soft + Z3 SMT solver verification/repair per sample.

Outputs a JSON report and optional CSV for plotting.

Usage:
    python scripts/benchmark_latency.py --n_samples 500 --device cpu
    python scripts/benchmark_latency.py --n_samples 1000 --device cuda --csv results/latency.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from data.multi_digit_addition import MultiDigitAdditionDataset, multi_digit_collate
from models.nst_multi_digit import MultiDigitModel
from torch.utils.data import DataLoader


def _auto_device(preferred: str | None = None) -> str:
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


def benchmark_mode(
    mode: str,
    n_samples: int,
    batch_size: int,
    device: str,
    seed: int = 42,
    warmup: int = 3,
    repeats: int = 5,
    use_z3: bool = False,
) -> dict:
    """Benchmark inference latency for a single mode.

    Args:
        mode: "neural", "soft", or "lagrangian".
        n_samples: number of samples to run inference on.
        batch_size: batch size for DataLoader.
        device: computation device.
        seed: random seed.
        warmup: number of warmup batches (excluded from timing).
        repeats: number of timing repetitions.
        use_z3: if True, run Z3 hard constraint repair (only for hard mode).

    Returns:
        Dict with per-sample and total latency stats.
    """
    ds = MultiDigitAdditionDataset(
        split="comp_test", n_samples=n_samples, seed=seed
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=multi_digit_collate
    )

    model_mode = mode if mode != "hard" else "lagrangian"
    model = MultiDigitModel(mode=model_mode)
    model.to(device)
    model.eval()

    # Warmup passes (exclude from timing)
    for i, batch in enumerate(loader):
        if i >= warmup:
            break
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            _ = model(batch_dev["img_a"], batch_dev["img_b"])

    # Timed runs
    latencies = []
    for rep in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        n_inferred = 0

        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.no_grad():
                result = model(batch_dev["img_a"], batch_dev["img_b"])

            # Hard mode: run Z3 repair per sample
            if use_z3:
                from symbolic.constraint_solver import hard_constraint_verify
                pred_a = result["probs_a_tens"].argmax(-1)
                pred_b = result["probs_a_ones"].argmax(-1)
                pred_s = result["probs_s_ones"].argmax(-1)
                for j in range(len(pred_a)):
                    hard_constraint_verify(
                        pred_a[j].item(), pred_b[j].item(), pred_s[j].item()
                    )

            n_inferred += len(batch_dev["img_a"])

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

    per_sample_ms = [(t / n_inferred) * 1000 for t in latencies]

    return {
        "mode": mode,
        "device": device,
        "n_samples": n_inferred,
        "batch_size": batch_size,
        "repeats": repeats,
        "total_s_mean": round(float(np.mean(latencies)), 4),
        "total_s_std": round(float(np.std(latencies)), 4),
        "per_sample_ms_mean": round(float(np.mean(per_sample_ms)), 3),
        "per_sample_ms_std": round(float(np.std(per_sample_ms)), 3),
        "throughput_samples_per_s": round(n_inferred / float(np.mean(latencies)), 1),
    }


def main():
    ap = argparse.ArgumentParser(description="NST inference latency benchmark")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default="results/latency_report.json")
    ap.add_argument("--csv", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    device = _auto_device(args.device)

    modes = [
        ("neural", False),
        ("soft", False),
        ("lagrangian", False),
        ("hard (Z3)", True),
    ]

    results = []
    print(f"Benchmarking inference latency on {device} ({args.n_samples} samples, "
          f"batch={args.batch_size}, repeats={args.repeats})")
    print("-" * 70)

    for mode_name, use_z3 in modes:
        actual_mode = mode_name.split(" ")[0]
        if actual_mode == "hard":
            actual_mode = "lagrangian"

        print(f"  {mode_name:>15} ... ", end="", flush=True)
        res = benchmark_mode(
            mode=actual_mode,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            repeats=args.repeats,
            use_z3=use_z3,
        )
        res["mode"] = mode_name
        results.append(res)
        print(f"{res['per_sample_ms_mean']:>8.3f} ms/sample  "
              f"({res['throughput_samples_per_s']:>7.1f} samples/s)")

    print("-" * 70)

    # Save JSON
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON report: {args.json}")

    # Optional CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w") as f:
            f.write("mode,device,n_samples,batch_size,per_sample_ms_mean,"
                    "per_sample_ms_std,throughput_sps\n")
            for r in results:
                f.write(f"{r['mode']},{r['device']},{r['n_samples']},"
                        f"{r['batch_size']},{r['per_sample_ms_mean']},"
                        f"{r['per_sample_ms_std']},{r['throughput_samples_per_s']}\n")
        print(f"CSV report:  {args.csv}")

    # Print summary table
    print("\n=== Latency Summary ===")
    print(f"{'Mode':<15} {'ms/sample':>12} {'±':>3} {'std':>8} {'throughput':>14}")
    for r in results:
        print(f"{r['mode']:<15} {r['per_sample_ms_mean']:>12.3f}   ± {r['per_sample_ms_std']:>7.3f}"
              f"  {r['throughput_samples_per_s']:>10.1f} s/s")


if __name__ == "__main__":
    main()
