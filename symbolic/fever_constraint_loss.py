"""Differentiable neuro-symbolic constraints for FEVER fact verification.

These constraints encode domain knowledge as differentiable losses on
the model's label probability distribution P(SUPPORTS), P(REFUTES), P(NEI).

Constraint design principles:
  1. Constraints come from EXTRACTED SIGNALS (numbers, dates, negation, entities),
     not from gold labels. The extractor is noisy — that's expected.
  2. Constraints are SOFT: they produce a loss penalty, not hard decisions.
  3. Constraints push probabilities in the right direction but never force labels.

Implemented constraints (differentiable Horn-clause form):

  C1: DATE_CONTRADICTION ∧ ¬low_overlap → P(SUPPORTS) should be low
      "If dates conflict and entities overlap, it's probably REFUTES."

  C2: NUMBER_CONTRADICTION ∧ ¬low_overlap → P(SUPPORTS) should be low
      "If numbers conflict and entities overlap, it's probably REFUTES."

  C3: NEGATION_MISMATCH → P(SUPPORTS) should decrease
      "If one side negates and the other doesn't, SUPPORTS is unlikely."

  C4: LOW_ENTITY_OVERLAP → P(NEI) should increase
      "If claim and evidence share few entities, evidence may be irrelevant."

  C5: VERY_LOW_OVERLAP ∧ empty_evidence → P(NEI) should be high
      "If evidence is empty or irrelevant, predict NOT ENOUGH INFO."

Each constraint returns a scalar violation loss ∈ [0, 1].
"""

from __future__ import annotations

import torch
from logic.logic import neg, t_and, imply, horn_violation
from symbolic.fever_constraints import StructuredFacts


def _facts_to_signals(
    facts_batch: list[StructuredFacts],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert structured facts to differentiable signal tensors.

    All signals ∈ [0, 1], ready for fuzzy logic operations.
    """
    B = len(facts_batch)

    date_contradiction = torch.tensor(
        [1.0 if f.date_contradiction else 0.0 for f in facts_batch],
        device=device,
    )
    number_contradiction = torch.tensor(
        [1.0 if f.number_contradiction else 0.0 for f in facts_batch],
        device=device,
    )
    negation_mismatch = torch.tensor(
        [1.0 if f.negation_mismatch else 0.0 for f in facts_batch],
        device=device,
    )
    entity_overlap = torch.tensor(
        [f.entity_overlap_score for f in facts_batch],
        device=device,
    )
    has_evidence = torch.tensor(
        [1.0 if (f.entities_evidence or f.numbers_evidence or f.dates_evidence)
         else 0.0 for f in facts_batch],
        device=device,
    )

    return {
        "date_contradiction": date_contradiction,
        "number_contradiction": number_contradiction,
        "negation_mismatch": negation_mismatch,
        "entity_overlap": entity_overlap,
        "has_evidence": has_evidence,
    }


def fever_constraint_loss(
    p_supports: torch.Tensor,    # [B]
    p_refutes: torch.Tensor,     # [B]
    p_nei: torch.Tensor,         # [B]
    facts_batch: list[StructuredFacts],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute differentiable constraint loss for FEVER.

    Args:
        p_supports: P(SUPPORTS) for each sample.
        p_refutes: P(REFUTES) for each sample.
        p_nei: P(NEI) for each sample.
        facts_batch: extracted structured facts for each sample.
        weights: optional per-constraint weights.

    Returns:
        (total_loss, info_dict) where total_loss is a scalar.
    """
    if not facts_batch:
        return torch.tensor(0.0, device=p_supports.device), {}

    w = weights or {}
    w_date = w.get("date_contradiction", 1.0)
    w_number = w.get("number_contradiction", 1.0)
    w_negation = w.get("negation_mismatch", 0.5)
    w_low_overlap = w.get("low_entity_overlap", 0.5)
    w_empty = w.get("empty_evidence", 1.0)

    device = p_supports.device
    signals = _facts_to_signals(facts_batch, device)

    # C1: date_contradiction ∧ has_evidence → ¬SUPPORTS
    # Horn violation: body * (1 - head), where head = ¬SUPPORTS = (1 - P(S))
    body_c1 = t_and(signals["date_contradiction"], signals["has_evidence"])
    head_c1 = neg(p_supports)  # want P(SUPPORTS) low → head = 1 - P(S)
    v_c1 = (body_c1 * (1 - head_c1.clamp(0, 1))).mean()

    # C2: number_contradiction ∧ has_evidence → ¬SUPPORTS
    body_c2 = t_and(signals["number_contradiction"], signals["has_evidence"])
    head_c2 = neg(p_supports)
    v_c2 = (body_c2 * (1 - head_c2.clamp(0, 1))).mean()

    # C3: negation_mismatch → ¬SUPPORTS
    body_c3 = signals["negation_mismatch"]
    head_c3 = neg(p_supports)
    v_c3 = (body_c3 * (1 - head_c3.clamp(0, 1))).mean()

    # C4: low_entity_overlap → NEI (should increase)
    # low overlap = 1 - entity_overlap (fuzzy "low" is high when overlap is low)
    low_overlap = neg(signals["entity_overlap"])
    body_c4 = low_overlap
    head_c4 = p_nei  # want P(NEI) high
    v_c4 = (body_c4 * (1 - head_c4.clamp(0, 1))).mean()

    # C5: no_evidence → NEI (strong push)
    no_evidence = neg(signals["has_evidence"])
    body_c5 = no_evidence
    head_c5 = p_nei
    v_c5 = (body_c5 * (1 - head_c5.clamp(0, 1))).mean()

    total = (w_date * v_c1 + w_number * v_c2 + w_negation * v_c3 +
             w_low_overlap * v_c4 + w_empty * v_c5)

    info = {
        "v_date_contradiction": v_c1.item(),
        "v_number_contradiction": v_c2.item(),
        "v_negation_mismatch": v_c3.item(),
        "v_low_entity_overlap": v_c4.item(),
        "v_empty_evidence": v_c5.item(),
        "constraint_loss_total": total.item(),
        "n_date_contradictions": signals["date_contradiction"].sum().item(),
        "n_number_contradictions": signals["number_contradiction"].sum().item(),
        "n_negation_mismatches": signals["negation_mismatch"].sum().item(),
        "mean_entity_overlap": signals["entity_overlap"].mean().item(),
    }

    return total, info


def verify_fever_constraints(
    pred_labels: list[str],
    facts_batch: list[StructuredFacts],
) -> tuple[list[bool], float]:
    """Hard verification of FEVER constraints (for CEGIS counterexample mining).

    Checks whether model predictions are consistent with extracted signals.
    Returns violations (True = violated) and constraint satisfaction rate.

    Violation criteria:
    - Predicts SUPPORTS but date/number contradiction detected → violation
    - Predicts SUPPORTS but negation mismatch → violation
    - Predicts SUPPORTS/REFUTES but entity overlap < 0.1 → violation
    """
    violations = []

    for pred, facts in zip(pred_labels, facts_batch):
        violated = False

        if pred == "SUPPORTS":
            # Date/number contradiction → should not support
            if facts.date_contradiction or facts.number_contradiction:
                violated = True
            # Negation mismatch → should not support
            if facts.negation_mismatch:
                violated = True

        if pred in ("SUPPORTS", "REFUTES"):
            # Very low entity overlap → evidence probably irrelevant
            if facts.entity_overlap_score < 0.1 and not facts.entities_evidence:
                violated = True

        violations.append(violated)

    csr = 1.0 - (sum(violations) / max(1, len(violations)))
    return violations, csr
