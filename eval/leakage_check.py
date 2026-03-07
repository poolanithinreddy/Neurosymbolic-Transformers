"""Data leakage detection for FEVER experiments.

Provides three critical checks:
1. Split disjointness — no claim appears in both train and dev/test
2. Claim text overlap — fuzzy detection of near-duplicate claims
3. Constraint independence — constraints don't encode label information

Run as: python -m eval.leakage_check --data-dir data/ --seed 42
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np
import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

logger = logging.getLogger("leakage_check")


# ── 1. Split Disjointness ──────────────────────────────────────
def check_split_disjointness(
    splits: dict[str, list[dict]],
    key: str = "claim",
) -> dict[str, Any]:
    """Verify strict disjointness between train/dev/dev_test/test splits.

    Checks exact string match on the `key` field (default: 'claim').

    Args:
        splits: dict mapping split name → list of examples with `key` field
        key: field to check for overlap

    Returns:
        dict with overlap statistics and any overlapping examples
    """
    results = {}

    split_names = list(splits.keys())
    for i, name_a in enumerate(split_names):
        set_a = {_normalize(item[key]) for item in splits[name_a]}
        for name_b in split_names[i + 1:]:
            set_b = {_normalize(item[key]) for item in splits[name_b]}
            overlap = set_a & set_b
            pair_key = f"{name_a}_vs_{name_b}"
            results[pair_key] = {
                "n_overlap": len(overlap),
                "n_a": len(set_a),
                "n_b": len(set_b),
                "overlap_pct_a": round(len(overlap) / max(1, len(set_a)) * 100, 2),
                "overlap_pct_b": round(len(overlap) / max(1, len(set_b)) * 100, 2),
                "examples": list(overlap)[:10],  # Show up to 10 examples
            }
            if overlap:
                logger.warning(
                    f"LEAK: {len(overlap)} overlapping claims "
                    f"between {name_a} ({len(set_a)}) and {name_b} ({len(set_b)})"
                )

    return results


def _normalize(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.strip().lower()


# ── 2. Claim Text Overlap (Fuzzy) ──────────────────────────────
def check_claim_text_overlap(
    train_examples: list[dict],
    eval_examples: list[dict],
    key: str = "claim",
    threshold: float = 0.9,
    max_comparisons: int = 100_000,
) -> dict[str, Any]:
    """Detect near-duplicate claims between train and eval splits.

    Uses character n-gram Jaccard similarity for efficiency.

    Args:
        train_examples: training set examples
        eval_examples: evaluation set examples  
        key: field to compare
        threshold: Jaccard similarity threshold for match
        max_comparisons: limit random comparisons for speed

    Returns:
        dict with fuzzy overlap statistics
    """
    def ngrams(text: str, n: int = 3) -> set:
        text = text.lower().strip()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / max(1, union)

    train_ngrams = [(item[key], ngrams(item[key])) for item in train_examples]
    eval_ngrams = [(item[key], ngrams(item[key])) for item in eval_examples]

    # Deterministic sampling for reproducibility
    rng = np.random.RandomState(42)

    near_duplicates = []
    n_compared = 0

    for eval_text, eval_ng in eval_ngrams:
        if n_compared >= max_comparisons:
            break

        for train_text, train_ng in train_ngrams:
            if n_compared >= max_comparisons:
                break

            sim = jaccard(eval_ng, train_ng)
            if sim >= threshold:
                near_duplicates.append({
                    "train_claim": train_text,
                    "eval_claim": eval_text,
                    "similarity": round(sim, 4),
                })
            n_compared += 1

    return {
        "n_near_duplicates": len(near_duplicates),
        "threshold": threshold,
        "n_compared": n_compared,
        "examples": near_duplicates[:20],
    }


# ── 3. Constraint Independence ─────────────────────────────────
def check_constraint_independence(
    claims: list[str],
    evidences: list[str],
    labels: list[int],
    n_samples: int = 5000,
) -> dict[str, Any]:
    """Check that constraint signals don't trivially encode label info.

    If a constraint has >0.8 mutual information with labels,
    it may be a proxy for the label rather than a useful inductive bias.

    We measure:
      - Per-constraint fire rate by label
      - Per-constraint accuracy if used as a classifier
      - Correlation between constraint confidence and label

    Args:
        claims: list of claim texts
        evidences: list of evidence texts  
        labels: list of label ids (0/1/2)
        n_samples: max examples to evaluate

    Returns:
        dict with per-constraint independence statistics
    """
    from symbolic.constraints_v2 import ConstraintEngineV2

    engine = ConstraintEngineV2()

    # Sample if too many
    if n_samples < len(claims):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(claims), n_samples, replace=False)
        claims = [claims[i] for i in indices]
        evidences = [evidences[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Evaluate constraints
    signals = engine.evaluate_batch(claims, evidences)
    fires = signals["fires"].numpy()       # (N, K)
    confidence = signals["confidence"].numpy()  # (N, K)
    direction = signals["direction"].numpy()     # (N, K, 3)
    labels_arr = np.array(labels)            # (N,)

    results = {}
    for k, name in enumerate(engine.constraint_names):
        fires_k = fires[:, k]
        conf_k = confidence[:, k]

        # Fire rate by label
        fire_by_label = {}
        for label_id in range(3):
            mask = labels_arr == label_id
            if mask.sum() > 0:
                fire_by_label[label_id] = float(fires_k[mask].mean())

        # If constraint direction is used as classifier, what accuracy?
        fired_mask = fires_k > 0
        n_fired = fired_mask.sum()
        classifier_acc = 0.0
        if n_fired > 0:
            dir_k = direction[fired_mask, k, :]  # (n_fired, 3)
            pred_labels = dir_k.argmax(axis=-1)
            true_labels = labels_arr[fired_mask]
            classifier_acc = float((pred_labels == true_labels).mean())

        # Correlation between confidence and correct prediction
        correct = (labels_arr == direction[:, k, :].argmax(axis=-1)).astype(float)
        if n_fired > 0:
            corr = float(np.corrcoef(conf_k[fired_mask], correct[fired_mask])[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Check if this constraint is suspiciously predictive
        is_suspicious = classifier_acc > 0.8 and n_fired > len(labels) * 0.1
        if is_suspicious:
            logger.warning(
                f"SUSPICIOUS: constraint '{name}' acts as classifier "
                f"(acc={classifier_acc:.3f} on {n_fired} examples)"
            )

        results[name] = {
            "fire_rate": float(fires_k.mean()),
            "fire_by_label": fire_by_label,
            "n_fired": int(n_fired),
            "classifier_accuracy": round(classifier_acc, 4),
            "confidence_correct_corr": round(corr, 4),
            "suspicious": is_suspicious,
        }

    return results


# ── 4. Full Leakage Report ──────────────────────────────────────
def full_leakage_report(
    splits: dict[str, list[dict]],
    claim_key: str = "claim",
    evidence_key: str = "gold_evidence_text",
    label_key: str = "label_id",
) -> dict[str, Any]:
    """Run all leakage checks and produce a comprehensive report.

    Args:
        splits: dict of split_name → list of examples

    Returns:
        Full report dict with all check results and an overall verdict.
    """
    report = {}

    # 1. Split disjointness
    report["split_disjointness"] = check_split_disjointness(splits, key=claim_key)

    # 2. Fuzzy claim overlap (train vs dev)
    if "train" in splits and "dev" in splits:
        report["fuzzy_overlap_train_dev"] = check_claim_text_overlap(
            splits["train"], splits["dev"], key=claim_key,
        )

    if "train" in splits and "dev_test" in splits:
        report["fuzzy_overlap_train_devtest"] = check_claim_text_overlap(
            splits["train"], splits["dev_test"], key=claim_key,
        )

    # 3. Constraint independence
    if "train" in splits:
        claims = [it[claim_key] for it in splits["train"][:5000]]
        evidences = [it.get(evidence_key, "") for it in splits["train"][:5000]]
        labels = [it.get(label_key, 0) for it in splits["train"][:5000]]
        report["constraint_independence"] = check_constraint_independence(
            claims, evidences, labels,
        )

    # Overall verdict
    issues = []
    for pair, stats in report.get("split_disjointness", {}).items():
        if stats["n_overlap"] > 0:
            issues.append(f"Split overlap: {pair} ({stats['n_overlap']} claims)")

    for key in ["fuzzy_overlap_train_dev", "fuzzy_overlap_train_devtest"]:
        if key in report and report[key]["n_near_duplicates"] > 0:
            issues.append(f"Near-duplicates: {key} ({report[key]['n_near_duplicates']})")

    for name, stats in report.get("constraint_independence", {}).items():
        if stats.get("suspicious"):
            issues.append(f"Suspicious constraint: {name}")

    report["verdict"] = {
        "clean": len(issues) == 0,
        "n_issues": len(issues),
        "issues": issues,
    }

    return report


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="FEVER data leakage check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-dev", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    from data.fever_dataset import load_fever_splits

    splits = load_fever_splits(
        seed=args.seed, max_train=args.max_train, max_dev=args.max_dev,
    )

    report = full_leakage_report(splits)

    # Print summary
    print("\n" + "=" * 60)
    print("  FEVER Data Leakage Report")
    print("=" * 60)

    for pair, stats in report.get("split_disjointness", {}).items():
        status = "CLEAN" if stats["n_overlap"] == 0 else "LEAK"
        print(f"  [{status}] {pair}: {stats['n_overlap']} overlapping claims")

    for key in ["fuzzy_overlap_train_dev", "fuzzy_overlap_train_devtest"]:
        if key in report:
            n = report[key]["n_near_duplicates"]
            status = "CLEAN" if n == 0 else "WARNING"
            print(f"  [{status}] {key}: {n} near-duplicates")

    if "constraint_independence" in report:
        print("\n  Constraint independence:")
        for name, stats in report["constraint_independence"].items():
            flag = " ⚠" if stats.get("suspicious") else ""
            print(f"    {name}: fire={stats['fire_rate']:.3f}, "
                  f"clf_acc={stats['classifier_accuracy']:.3f}{flag}")

    verdict = report["verdict"]
    print(f"\n  VERDICT: {'CLEAN' if verdict['clean'] else 'ISSUES FOUND'}")
    for issue in verdict["issues"]:
        print(f"    - {issue}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to {args.out}")
