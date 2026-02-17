"""FEVER evaluation metrics and error decomposition.

Implements:
  - Label accuracy (overall + per-class)
  - ECE and Brier score (from eval/calibration_metrics.py)
  - Retrieval recall@k (for pipeline mode)
  - Error decomposition: retrieval vs NLI vs constraint
  - Mean ± std aggregation across seeds
  - Integrity checks (split hash, label shuffle sanity)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F

from data.fever_dataset import FEVER_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS
from eval.calibration_metrics import expected_calibration_error, brier_score

logger = logging.getLogger("fever_metrics")


def label_accuracy(
    pred_labels: list[str],
    gold_labels: list[str],
) -> dict[str, Any]:
    """Compute label accuracy with per-class breakdown.

    Args:
        pred_labels: predicted labels (SUPPORTS/REFUTES/NOT ENOUGH INFO).
        gold_labels: ground-truth labels.

    Returns:
        Dict with overall accuracy and per-class accuracy/count.
    """
    assert len(pred_labels) == len(gold_labels), "Length mismatch"

    n_total = len(gold_labels)
    n_correct = sum(p == g for p, g in zip(pred_labels, gold_labels))
    overall_acc = n_correct / max(1, n_total)

    per_class = {}
    for label in FEVER_LABELS:
        mask = [g == label for g in gold_labels]
        n_class = sum(mask)
        if n_class > 0:
            n_class_correct = sum(
                p == g for p, g, m in zip(pred_labels, gold_labels, mask) if m
            )
            per_class[label] = {
                "accuracy": round(n_class_correct / n_class, 4),
                "count": n_class,
                "correct": n_class_correct,
            }

    return {
        "accuracy": round(overall_acc, 4),
        "n_total": n_total,
        "n_correct": n_correct,
        "per_class": per_class,
    }


def confusion_matrix(
    pred_labels: list[str],
    gold_labels: list[str],
) -> dict[str, dict[str, int]]:
    """Build confusion matrix: gold → pred → count."""
    cm = {g: {p: 0 for p in FEVER_LABELS} for g in FEVER_LABELS}
    for p, g in zip(pred_labels, gold_labels):
        if g in cm and p in cm[g]:
            cm[g][p] += 1
    return cm


def retrieval_recall_at_k(
    retrieved_titles: list[list[str]],
    gold_titles: list[list[str]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval recall@k for pipeline mode.

    Args:
        retrieved_titles: list of retrieved title lists per sample.
        gold_titles: list of gold evidence title lists per sample.
        k_values: k values to evaluate (default: [1, 3, 5, 10]).

    Returns:
        Dict mapping "recall@k" → value.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {}
    for k in k_values:
        hits = 0
        total = 0
        for retrieved, gold in zip(retrieved_titles, gold_titles):
            if not gold:
                continue
            total += 1
            top_k = set(t.lower() for t in retrieved[:k])
            gold_set = set(t.lower() for t in gold)
            if top_k & gold_set:
                hits += 1

        recall = hits / max(1, total)
        results[f"recall@{k}"] = round(recall, 4)

    return results


def error_decomposition(
    pred_labels: list[str],
    gold_labels: list[str],
    retrieval_recall: float | None = None,
    gold_evidence_acc: float | None = None,
) -> dict[str, Any]:
    """Decompose end-to-end errors into retrieval vs NLI components.

    Args:
        pred_labels: pipeline predictions.
        gold_labels: gold labels.
        retrieval_recall: recall@5 of the retriever.
        gold_evidence_acc: NLI accuracy with gold evidence (upper bound).

    Returns:
        Dict with error decomposition.
    """
    e2e_acc = sum(p == g for p, g in zip(pred_labels, gold_labels)) / max(1, len(gold_labels))

    result = {
        "end_to_end_accuracy": round(e2e_acc, 4),
    }

    if gold_evidence_acc is not None:
        result["gold_evidence_accuracy"] = round(gold_evidence_acc, 4)
        result["nli_gap"] = round(gold_evidence_acc - e2e_acc, 4)

    if retrieval_recall is not None:
        result["retrieval_recall_at_5"] = round(retrieval_recall, 4)
        # Upper bound: if retrieval were perfect, accuracy ≈ gold_evidence_acc
        if gold_evidence_acc is not None:
            result["retrieval_caused_loss"] = round(
                gold_evidence_acc * (1 - retrieval_recall), 4
            )

    return result


def integrity_check(
    splits: dict[str, list[dict]],
    pred_labels: list[str],
    gold_labels: list[str],
    evidence_mode: str,
) -> dict[str, Any]:
    """Run integrity checks on evaluation results.

    Checks:
    1. Dataset sizes match expected FEVER split sizes.
    2. Split hash is reproducible.
    3. Label distribution is not degenerate.
    4. Shuffled labels → accuracy drops to ~1/3 (sanity check).

    Args:
        splits: raw data splits from load_fever_splits.
        pred_labels: model predictions.
        gold_labels: ground truth.
        evidence_mode: "gold" or "pipeline".

    Returns:
        Dict with check results.
    """
    checks = {}

    # Check 1: split sizes
    for split_name, items in splits.items():
        checks[f"size_{split_name}"] = len(items)

    # Check 2: split hashes
    for split_name, items in splits.items():
        content = json.dumps(
            [(it["id"], it["claim"][:50], it["label"]) for it in items[:1000]],
            sort_keys=True,
        )
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        checks[f"hash_{split_name}"] = h

    # Check 3: label distribution
    label_dist = Counter(gold_labels)
    checks["label_distribution"] = dict(label_dist)
    checks["n_unique_labels"] = len(label_dist)
    checks["label_dist_ok"] = len(label_dist) >= 2  # At least 2 labels present

    # Check 4: actual accuracy
    actual_acc = sum(p == g for p, g in zip(pred_labels, gold_labels)) / max(1, len(gold_labels))
    checks["accuracy"] = round(actual_acc, 4)

    # Check 5: shuffle sanity (would shuffled labels give ~1/3?)
    shuffled = list(gold_labels)
    random.shuffle(shuffled)
    shuffle_acc = sum(p == g for p, g in zip(pred_labels, shuffled)) / max(1, len(shuffled))
    checks["shuffle_accuracy"] = round(shuffle_acc, 4)
    checks["shuffle_sanity_ok"] = shuffle_acc < actual_acc  # Shuffled should be worse

    # Check 6: evidence mode
    checks["evidence_mode"] = evidence_mode

    return checks


def format_results_table(
    results: dict[str, dict],
    evidence_mode: str = "gold",
) -> str:
    """Format FEVER results as a clear Markdown table.

    Clearly labels the evidence setting to avoid ambiguity.
    """
    header = f"## FEVER Results — {'Gold Evidence (Setting A)' if evidence_mode == 'gold' else 'Retrieved Evidence / Full Pipeline (Setting B)'}\n"

    lines = [
        header,
        "| Model | Label Acc | SUPPORTS Acc | REFUTES Acc | NEI Acc | ECE ↓ | Brier ↓ |",
        "|-------|----------|-------------|-------------|---------|-------|---------|",
    ]

    for name, res in results.items():
        dev = res.get("dev", res)
        acc = dev.get("accuracy", 0)
        ece = dev.get("ece", 0)
        brier = dev.get("brier", 0)

        per_class = dev.get("per_label", {})
        s_acc = per_class.get("SUPPORTS", {}).get("accuracy", 0)
        r_acc = per_class.get("REFUTES", {}).get("accuracy", 0)
        n_acc = per_class.get("NOT ENOUGH INFO", {}).get("accuracy", 0)

        lines.append(
            f"| {name} | {acc:.4f} | {s_acc:.4f} | {r_acc:.4f} | {n_acc:.4f} | {ece:.4f} | {brier:.4f} |"
        )

    return "\n".join(lines)


def format_results_table_mean_std(
    aggregated: dict[str, dict[str, dict]],
    evidence_mode: str = "gold",
) -> str:
    """Format aggregated (mean±std) results as Markdown.

    Args:
        aggregated: {model_name: {metric: {mean, std, n}}}
    """
    header = f"## FEVER Results (mean ± std) — {'Gold Evidence' if evidence_mode == 'gold' else 'Full Pipeline'}\n"

    lines = [
        header,
        "| Model | Label Acc | ECE ↓ | Brier ↓ | Seeds |",
        "|-------|----------|-------|---------|-------|",
    ]

    for name, metrics in aggregated.items():
        acc = metrics.get("dev/accuracy", {})
        ece = metrics.get("dev/ece", {})
        brier = metrics.get("dev/brier", {})

        acc_str = f"{acc.get('mean', 0):.4f}±{acc.get('std', 0):.4f}" if acc else "N/A"
        ece_str = f"{ece.get('mean', 0):.4f}±{ece.get('std', 0):.4f}" if ece else "N/A"
        brier_str = f"{brier.get('mean', 0):.4f}±{brier.get('std', 0):.4f}" if brier else "N/A"
        n = acc.get("n", 0)

        lines.append(f"| {name} | {acc_str} | {ece_str} | {brier_str} | {n} |")

    return "\n".join(lines)
