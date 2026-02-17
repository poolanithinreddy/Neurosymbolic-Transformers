"""Rule-satisfaction checker for neuro-symbolic predictions.

Given model predictions and ground-truth labels, computes per-rule
satisfaction rates using the symbolic rule engine.
"""

from __future__ import annotations

from typing import Any

import torch


def rule_satisfaction_report(
    pred_probs: torch.Tensor,
    labels: torch.Tensor,
    chain_lengths: list[int] | None = None,
    task: str = "digit_add",
) -> dict[str, Any]:
    """Compute per-rule satisfaction statistics.

    Args:
        pred_probs: [N, C] predicted probabilities (post-softmax).
        labels: [N] ground-truth class indices.
        chain_lengths: optional chain lengths for kinship task.
        task: "digit_add" or "kinship".

    Returns:
        Dict with overall CSR and per-rule breakdowns.
    """
    preds = pred_probs.argmax(dim=-1)
    N = len(labels)

    if task == "kinship" and chain_lengths is not None:
        return _kinship_rule_check(preds, labels, chain_lengths)
    elif task == "digit_add":
        return _digit_add_rule_check(preds, labels)

    return {"csr": 0.0, "rules": [], "task": task, "n_samples": N}


def _kinship_rule_check(
    preds: torch.Tensor,
    labels: torch.Tensor,
    chain_lengths: list[int],
) -> dict:
    """Check kinship rule satisfaction by chain length."""
    from data.kinship import check_kinship_constraint

    probs_one_hot = torch.nn.functional.one_hot(preds, num_classes=8).float()
    csr, viol = check_kinship_constraint(probs_one_hot, chain_lengths)

    # Per-depth breakdown
    depth_stats: dict[int, dict] = {}
    for i, cl in enumerate(chain_lengths):
        if cl not in depth_stats:
            depth_stats[cl] = {"correct": 0, "total": 0}
        depth_stats[cl]["total"] += 1
        if preds[i] == labels[i]:
            depth_stats[cl]["correct"] += 1

    rules = []
    for depth in sorted(depth_stats):
        s = depth_stats[depth]
        rules.append({
            "rule": f"depth_{depth}_accuracy",
            "satisfied": s["correct"],
            "total": s["total"],
            "rate": s["correct"] / max(1, s["total"]),
        })

    return {
        "csr": csr,
        "violation_rate": viol,
        "rules": rules,
        "n_samples": len(labels),
        "task": "kinship",
    }


def _digit_add_rule_check(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Check digit addition accuracy and constraint satisfaction."""
    correct = (preds == labels).sum().item()
    total = len(labels)

    return {
        "csr": correct / max(1, total),
        "rules": [
            {
                "rule": "arithmetic_correctness",
                "satisfied": correct,
                "total": total,
                "rate": correct / max(1, total),
            }
        ],
        "n_samples": total,
        "task": "digit_add",
    }
