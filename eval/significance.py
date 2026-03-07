"""Statistical significance tests for FEVER experiments.

Provides:
  - Bootstrap confidence intervals for accuracy / ECE / Brier
  - Paired bootstrap test (compare two systems)
  - McNemar's test (are two models' errors significantly different?)
  - Multi-seed aggregation with proper CIs

Usage:
    from eval.significance import bootstrap_ci, mcnemars_test, paired_bootstrap
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger("significance")


# ── Bootstrap CI ────────────────────────────────────────────────
def bootstrap_ci(
    metric_fn,
    *arrays,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: callable(*arrays) → float (e.g., accuracy_fn)
        *arrays: numpy arrays of equal length (predictions, labels, etc.)
        n_bootstrap: number of bootstrap samples
        alpha: significance level (0.05 → 95% CI)
        seed: random seed

    Returns:
        dict with: point, lower, upper, std, n_bootstrap
    """
    rng = np.random.RandomState(seed)
    n = len(arrays[0])

    point_estimate = metric_fn(*arrays)

    boot_values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        sampled = tuple(a[idx] for a in arrays)
        boot_values[b] = metric_fn(*sampled)

    lower = float(np.percentile(boot_values, 100 * alpha / 2))
    upper = float(np.percentile(boot_values, 100 * (1.0 - alpha / 2)))

    return {
        "point": round(float(point_estimate), 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "std": round(float(boot_values.std()), 4),
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


# ── Paired Bootstrap Test ──────────────────────────────────────
def paired_bootstrap(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap test: is system A better than system B?

    Tests H0: accuracy(A) <= accuracy(B)
    Reports one-tailed p-value for A > B.

    Args:
        preds_a: predictions from system A (N,)
        preds_b: predictions from system B (N,)
        labels: ground truth labels (N,)
        n_bootstrap: number of resamples

    Returns:
        dict with: delta, p_value, significant_at_05, ci_delta
    """
    rng = np.random.RandomState(seed)
    n = len(labels)

    acc_a = (preds_a == labels).mean()
    acc_b = (preds_b == labels).mean()
    delta = float(acc_a - acc_b)

    count_a_wins = 0
    boot_deltas = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_acc_a = (preds_a[idx] == labels[idx]).mean()
        boot_acc_b = (preds_b[idx] == labels[idx]).mean()
        boot_deltas[b] = boot_acc_a - boot_acc_b

        # Under H0, center at 0
        if boot_deltas[b] > 0:
            count_a_wins += 1

    p_value = 1.0 - count_a_wins / n_bootstrap

    return {
        "accuracy_a": round(float(acc_a), 4),
        "accuracy_b": round(float(acc_b), 4),
        "delta": round(delta, 4),
        "p_value": round(float(p_value), 5),
        "significant_at_05": float(p_value) < 0.05,
        "significant_at_01": float(p_value) < 0.01,
        "ci_delta_lower": round(float(np.percentile(boot_deltas, 2.5)), 4),
        "ci_delta_upper": round(float(np.percentile(boot_deltas, 97.5)), 4),
    }


# ── McNemar's Test ──────────────────────────────────────────────
def mcnemars_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
) -> dict[str, Any]:
    """McNemar's test: are two models' errors significantly different?

    Uses exact binomial test when cell counts < 25, else chi-squared.

    Args:
        correct_a: boolean array, True where model A is correct (N,)
        correct_b: boolean array, True where model B is correct (N,)

    Returns:
        dict with: b (A right, B wrong), c (A wrong, B right), chi2, p_value
    """
    # Contingency table
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    if b + c == 0:
        return {
            "b": int(b), "c": int(c),
            "chi2": 0.0, "p_value": 1.0,
            "significant_at_05": False,
            "note": "Models agree on all samples",
        }

    if b + c < 25:
        # Use exact binomial test
        from scipy.stats import binom_test
        try:
            p_value = binom_test(b, b + c, 0.5)
        except Exception:
            # Fallback if scipy not available
            p_value = _approximate_mcnemar(b, c)
    else:
        # Chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # One-sided p-value from chi2 distribution with df=1
        try:
            from scipy.stats import chi2 as chi2_dist
            p_value = 1.0 - chi2_dist.cdf(chi2, df=1)
        except ImportError:
            p_value = _chi2_cdf_approx(chi2)

    return {
        "b": int(b),
        "c": int(c),
        "chi2": round(float((abs(b - c) - 1) ** 2 / max(1, b + c)), 4),
        "p_value": round(float(p_value), 5),
        "significant_at_05": float(p_value) < 0.05,
    }


def _approximate_mcnemar(b: int, c: int) -> float:
    """Approximate McNemar without scipy."""
    n = b + c
    chi2 = (abs(b - c) - 1) ** 2 / n
    return _chi2_cdf_approx(chi2)


def _chi2_cdf_approx(x: float) -> float:
    """Rough chi2(df=1) p-value approximation."""
    # Using normal approximation: sqrt(chi2) ~ N(0,1)
    z = math.sqrt(max(0, x))
    # Two-tailed p-value from z
    return 2.0 * (1.0 - _normal_cdf(z))


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── Multi-Seed Aggregation ─────────────────────────────────────
def aggregate_seeds(
    seed_results: list[dict[str, Any]],
    metric_keys: list[str] = ("accuracy", "ece", "brier"),
) -> dict[str, Any]:
    """Aggregate results across multiple seeds.

    Args:
        seed_results: list of dicts, each with dev/dev_test metric dicts
        metric_keys: which metrics to aggregate

    Returns:
        dict with mean/std/min/max for each metric across seeds
    """
    agg = {}

    for split in ("dev", "dev_test"):
        split_results = [r.get(split, {}) for r in seed_results if r.get(split)]
        if not split_results:
            continue

        agg[split] = {}
        for key in metric_keys:
            values = [r[key] for r in split_results if key in r]
            if values:
                agg[split][key] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "min": round(float(np.min(values)), 4),
                    "max": round(float(np.max(values)), 4),
                    "n_seeds": len(values),
                    "values": [round(v, 4) for v in values],
                }

        # Per-label accuracy aggregation
        per_label_all = [r.get("per_label", {}) for r in split_results if "per_label" in r]
        if per_label_all:
            agg[split]["per_label"] = {}
            all_labels = set()
            for pl in per_label_all:
                all_labels.update(pl.keys())

            for label in all_labels:
                accs = [pl[label]["accuracy"] for pl in per_label_all if label in pl]
                if accs:
                    agg[split]["per_label"][label] = {
                        "accuracy_mean": round(float(np.mean(accs)), 4),
                        "accuracy_std": round(float(np.std(accs)), 4),
                    }

    return agg


# ── Convenience Functions ──────────────────────────────────────
def accuracy_with_ci(
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute accuracy with 95% bootstrap CI."""
    def acc_fn(p, l):
        return float((p == l).mean())

    return bootstrap_ci(acc_fn, preds, labels, n_bootstrap=n_bootstrap, alpha=alpha)


def ece_with_ci(
    probs: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    n_bootstrap: int = 5_000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute ECE with bootstrap CI."""
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    def ece_fn(p, l):
        pt = torch.from_numpy(p)
        lt = torch.from_numpy(l).long()
        ece_val, _ = expected_calibration_error(pt, lt)
        return float(ece_val)

    from eval.calibration_metrics import expected_calibration_error
    return bootstrap_ci(ece_fn, probs, labels, n_bootstrap=n_bootstrap, alpha=alpha)


def print_significance_report(
    name_a: str,
    name_b: str,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Print a formatted significance comparison between two systems."""
    boot = paired_bootstrap(preds_a, preds_b, labels)
    mcnemar = mcnemars_test(preds_a == labels, preds_b == labels)

    print(f"\n{'='*55}")
    print(f"  {name_a} vs {name_b}")
    print(f"{'='*55}")
    print(f"  {name_a}: {boot['accuracy_a']:.4f}")
    print(f"  {name_b}: {boot['accuracy_b']:.4f}")
    print(f"  Delta:   {boot['delta']:+.4f} "
          f"[{boot['ci_delta_lower']:+.4f}, {boot['ci_delta_upper']:+.4f}]")
    print(f"  Bootstrap p-value: {boot['p_value']:.5f}"
          f" {'**' if boot['significant_at_01'] else ('*' if boot['significant_at_05'] else 'n.s.')}")
    print(f"  McNemar: b={mcnemar['b']}, c={mcnemar['c']}, "
          f"chi2={mcnemar['chi2']:.2f}, p={mcnemar['p_value']:.5f}"
          f" {'*' if mcnemar['significant_at_05'] else 'n.s.'}")
