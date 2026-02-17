"""Evidence-Conditioned Constraint Gating (ECCG) for FEVER.

THE CORE NOVEL CONCEPT:

Standard neuro-symbolic constraints apply fixed weights to all samples.
But symbolic extractors are NOISY — sometimes the number extractor fires
on irrelevant numbers, or the entity matcher misses a synonym.

ECCG learns per-sample, per-constraint reliability gates:

    α_j(s) = σ(W_j · s + b_j)

where s is the vector of extracted symbolic signals and α_j ∈ [0,1] is the
gate for constraint j.  The gated constraint loss becomes:

    L_constraint^gated = Σ_j α_j(s) · v_j

Key properties:
1. Gates are trained END-TO-END via backprop through the constraint loss.
2. Gates learn "when does this constraint's signal actually help?"
3. At inference, gates use ONLY extracted signals (no label leakage).
4. Supervision comes from whether constraint direction matches the training
   label — this is training-set-only, no leakage.

Why this is NEW (vs standard approaches):
- Fixed λ: one scalar for all constraints and all samples.
- Per-constraint λ: one scalar per constraint, same for all samples.
- ECCG: per-constraint, per-sample reliability based on signal quality.
  This is analogous to Mixture of Experts gating for symbolic rules.

Theory: ECCG is a meta-learning layer for constraint reliability.
If constraint j has high precision on certain signal patterns but low
precision on others, the gate learns to fire strongly in the former
and weakly in the latter.  This is provably at least as good as
fixed weights (the gate can learn to output a constant).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from symbolic.fever_constraints import StructuredFacts


# Number of input features to the gate
_N_SIGNALS = 7  # date_contra, num_contra, neg_mismatch, overlap, has_evid, n_nums, n_dates

# Number of constraints
_N_CONSTRAINTS = 5  # C1-C5


def facts_to_gate_features(
    facts_batch: list[StructuredFacts],
    device: torch.device,
) -> torch.Tensor:
    """Convert structured facts to a feature vector for the gate network.

    Returns: [B, 7] tensor of signal features (all in [0,1] or normalised).
    """
    B = len(facts_batch)
    features = torch.zeros(B, _N_SIGNALS, device=device)

    for i, f in enumerate(facts_batch):
        features[i, 0] = 1.0 if f.date_contradiction else 0.0
        features[i, 1] = 1.0 if f.number_contradiction else 0.0
        features[i, 2] = 1.0 if f.negation_mismatch else 0.0
        features[i, 3] = f.entity_overlap_score
        features[i, 4] = 1.0 if (f.entities_evidence or f.numbers_evidence or f.dates_evidence) else 0.0
        # Signal richness: how many numbers/dates were found (normalised)
        features[i, 5] = min(1.0, len(f.numbers_claim + f.numbers_evidence) / 5.0)
        features[i, 6] = min(1.0, len(f.dates_claim + f.dates_evidence) / 3.0)

    return features


class ConstraintGate(nn.Module):
    """Lightweight gating network for per-constraint, per-sample reliability.

    Architecture: signal features → hidden → 5 gate values α_j ∈ [0,1].

    The gate is intentionally small (< 1K params) to avoid overfitting
    and to maintain interpretability.
    """

    def __init__(
        self,
        n_signals: int = _N_SIGNALS,
        n_constraints: int = _N_CONSTRAINTS,
        hidden_dim: int = 16,
        dropout: float = 0.1,
        init_bias: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_signals, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_constraints),
        )
        # Initialise bias so gates start near init_bias (not too aggressive)
        with torch.no_grad():
            self.net[-1].bias.fill_(init_bias)

        self.n_constraints = n_constraints

    def forward(self, signal_features: torch.Tensor) -> torch.Tensor:
        """Compute gate values.

        Args:
            signal_features: [B, n_signals] extracted signal features.

        Returns:
            gates: [B, n_constraints] gate values in [0, 1].
        """
        return torch.sigmoid(self.net(signal_features))


def gated_fever_constraint_loss(
    p_supports: torch.Tensor,
    p_refutes: torch.Tensor,
    p_nei: torch.Tensor,
    facts_batch: list[StructuredFacts],
    gate: ConstraintGate | None = None,
    base_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute ECCG-gated constraint loss for FEVER.

    If gate is None, falls back to standard fixed-weight constraint loss.

    Args:
        p_supports, p_refutes, p_nei: [B] label probabilities.
        facts_batch: extracted structured facts.
        gate: ConstraintGate module (None = fixed weights).
        base_weights: base per-constraint weights (used when gate=None).

    Returns:
        (total_loss, info_dict)
    """
    from symbolic.fever_constraint_loss import _facts_to_signals
    from logic.logic import neg, t_and

    if not facts_batch:
        return torch.tensor(0.0, device=p_supports.device), {}

    device = p_supports.device
    signals = _facts_to_signals(facts_batch, device)

    # Compute raw violations for each constraint (same as original)
    # C1: date_contradiction ∧ has_evidence → ¬SUPPORTS
    body_c1 = t_and(signals["date_contradiction"], signals["has_evidence"])
    head_c1 = neg(p_supports)
    v_c1 = body_c1 * (1 - head_c1.clamp(0, 1))

    # C2: number_contradiction ∧ has_evidence → ¬SUPPORTS
    body_c2 = t_and(signals["number_contradiction"], signals["has_evidence"])
    head_c2 = neg(p_supports)
    v_c2 = body_c2 * (1 - head_c2.clamp(0, 1))

    # C3: negation_mismatch → ¬SUPPORTS
    body_c3 = signals["negation_mismatch"]
    head_c3 = neg(p_supports)
    v_c3 = body_c3 * (1 - head_c3.clamp(0, 1))

    # C4: low_entity_overlap → NEI
    low_overlap = neg(signals["entity_overlap"])
    body_c4 = low_overlap
    head_c4 = p_nei
    v_c4 = body_c4 * (1 - head_c4.clamp(0, 1))

    # C5: no_evidence → NEI
    no_evidence = neg(signals["has_evidence"])
    body_c5 = no_evidence
    head_c5 = p_nei
    v_c5 = body_c5 * (1 - head_c5.clamp(0, 1))

    # Stack violations: [B, 5]
    violations = torch.stack([v_c1, v_c2, v_c3, v_c4, v_c5], dim=-1)

    if gate is not None:
        # ECCG: learned per-sample gates
        signal_features = facts_to_gate_features(facts_batch, device)
        gates = gate(signal_features)  # [B, 5]
        weighted_violations = violations * gates  # [B, 5]
        total = weighted_violations.mean()

        info = {
            "constraint_loss_total": total.item(),
            "v_date_contradiction": v_c1.mean().item(),
            "v_number_contradiction": v_c2.mean().item(),
            "v_negation_mismatch": v_c3.mean().item(),
            "v_low_entity_overlap": v_c4.mean().item(),
            "v_empty_evidence": v_c5.mean().item(),
            "gate_mean_c1": gates[:, 0].mean().item(),
            "gate_mean_c2": gates[:, 1].mean().item(),
            "gate_mean_c3": gates[:, 2].mean().item(),
            "gate_mean_c4": gates[:, 3].mean().item(),
            "gate_mean_c5": gates[:, 4].mean().item(),
            "n_date_contradictions": signals["date_contradiction"].sum().item(),
            "n_number_contradictions": signals["number_contradiction"].sum().item(),
            "n_negation_mismatches": signals["negation_mismatch"].sum().item(),
            "mean_entity_overlap": signals["entity_overlap"].mean().item(),
        }
    else:
        # Fixed weights (fallback)
        w = base_weights or {}
        w_vec = torch.tensor([
            w.get("date_contradiction", 1.0),
            w.get("number_contradiction", 1.0),
            w.get("negation_mismatch", 0.5),
            w.get("low_entity_overlap", 0.5),
            w.get("empty_evidence", 1.0),
        ], device=device)

        weighted_violations = violations * w_vec.unsqueeze(0)
        total = weighted_violations.mean()

        info = {
            "constraint_loss_total": total.item(),
            "v_date_contradiction": v_c1.mean().item(),
            "v_number_contradiction": v_c2.mean().item(),
            "v_negation_mismatch": v_c3.mean().item(),
            "v_low_entity_overlap": v_c4.mean().item(),
            "v_empty_evidence": v_c5.mean().item(),
            "n_date_contradictions": signals["date_contradiction"].sum().item(),
            "n_number_contradictions": signals["number_contradiction"].sum().item(),
            "n_negation_mismatches": signals["negation_mismatch"].sum().item(),
            "mean_entity_overlap": signals["entity_overlap"].mean().item(),
        }

    return total, info
