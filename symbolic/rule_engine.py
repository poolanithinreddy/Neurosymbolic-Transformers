"""Rule engine: load YAML rules and evaluate them over grounded predicate values.

Reuses the t-norm logic from logic/logic.py and wraps it in a rule-loading
and grounding interface for the R-CBM predicate heads.
"""

import os
from typing import Any

import torch
import yaml

from logic.logic import horn_truth, horn_violation, neg


def load_rules(yaml_path: str | None = None) -> list[dict]:
    """Load rules from YAML file.

    If no path is given, defaults to logic/rules.yaml relative to project root.

    Returns:
        List of rule dicts with keys: id, hard, weight, quant, body, head.
    """
    if yaml_path is None:
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "logic", "rules.yaml"
        )
    with open(yaml_path) as f:
        rules = yaml.safe_load(f)
    return rules


def load_predicates(yaml_path: str | None = None) -> list[dict]:
    """Load predicate definitions from YAML file.

    Returns:
        List of predicate dicts with keys: name, arity, grounding, essential.
    """
    if yaml_path is None:
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "logic", "predicates.yaml"
        )
    with open(yaml_path) as f:
        preds = yaml.safe_load(f)
    return preds


def evaluate_rule(
    rule: dict,
    predicate_values: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a single rule against grounded predicate values.

    Args:
        rule: rule dict from YAML with body/head structure.
        predicate_values: dict mapping predicate names to [B] tensors of
            soft truth values in [0, 1].

    Returns:
        (truth, violation): both [B] tensors.
            truth: soft truth value of the rule (via Reichenbach implication).
            violation: body * (1 - head), zero when rule is satisfied.
    """
    # Collect body values
    body_vals = []
    for atom in rule["body"]:
        pred_name = atom["pred"]
        if pred_name == "Not":
            # Negation: Not(P(x)) -> 1 - P(x)
            inner = atom["args"][0]
            if isinstance(inner, dict):
                inner_pred = inner["pred"]
                if inner_pred in predicate_values:
                    body_vals.append(neg(predicate_values[inner_pred]))
            elif inner in predicate_values:
                body_vals.append(neg(predicate_values[inner]))
        elif pred_name in predicate_values:
            body_vals.append(predicate_values[pred_name])

    # Head value
    head_spec = rule["head"]
    head_pred = head_spec["pred"]
    if head_pred == "Not":
        inner = head_spec["args"][0]
        if isinstance(inner, dict):
            inner_pred = inner["pred"]
            head_val = neg(predicate_values.get(inner_pred, torch.tensor(0.5)))
        else:
            head_val = neg(predicate_values.get(inner, torch.tensor(0.5)))
    else:
        head_val = predicate_values.get(head_pred, torch.tensor(0.5))

    if len(body_vals) == 0:
        # No grounded body atoms available — rule is vacuously true
        B = head_val.shape[0] if head_val.dim() > 0 else 1
        device = head_val.device
        return torch.ones(B, device=device), torch.zeros(B, device=device)

    body_stack = torch.stack(body_vals, dim=-1)
    truth = horn_truth(body_stack, head_val)
    violation = horn_violation(body_stack, head_val)

    return truth, violation


def evaluate_all_rules(
    rules: list[dict],
    predicate_values: dict[str, torch.Tensor],
    hard_only: bool = False,
    soft_only: bool = False,
) -> dict[str, Any]:
    """Evaluate all rules and return aggregated results.

    Args:
        rules: list of rule dicts.
        predicate_values: dict[pred_name] -> [B] tensor.
        hard_only: only evaluate hard rules.
        soft_only: only evaluate soft rules.

    Returns:
        Dict with:
            total_violation: scalar mean violation across all rules.
            total_truth: scalar mean truth across all rules.
            per_rule: list of {id, truth, violation, weight} dicts.
    """
    per_rule = []
    weighted_violations = []

    for rule in rules:
        if hard_only and not rule.get("hard", False):
            continue
        if soft_only and rule.get("hard", False):
            continue

        truth, violation = evaluate_rule(rule, predicate_values)
        w = float(rule.get("weight", 1.0))
        per_rule.append({
            "id": rule["id"],
            "truth": truth.mean().item(),
            "violation": violation.mean().item(),
            "weight": w,
        })
        weighted_violations.append(w * violation.mean())

    if len(weighted_violations) == 0:
        total_violation = torch.tensor(0.0)
        total_truth = torch.tensor(1.0)
    else:
        total_violation = torch.stack(weighted_violations).mean()
        total_truth = torch.tensor(
            sum(r["truth"] for r in per_rule) / len(per_rule)
        )

    return {
        "total_violation": total_violation,
        "total_truth": total_truth.item(),
        "per_rule": per_rule,
    }
