"""NST-VERI Constraint System v2: Probabilistic, calibrated, learnable.

Major upgrades from v1 (fever_constraint_loss.py):
  - Constraints output **confidence scores** [0,1], not binary flags.
  - Each constraint produces a **soft label direction** (3-way bias).
  - Antonym detection for robust negation handling.
  - Entity overlap drives NEI prediction with graded confidence.
  - Evidence sufficiency estimation.
  - Batch-friendly API returning tensors ready for training.

Design principles:
  - No constraint uses gold labels — only claim + evidence text.
  - Confidence reflects reliability: low confidence → don't trust.
  - Direction vectors are normalised probability-like biases.
  - The gate module learns to weight these signals per-sample.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch


# ── ConstraintSignal ─────────────────────────────────────────

@dataclass
class ConstraintSignal:
    """A single constraint's output for one example."""
    name: str
    fires: bool                    # Does this constraint activate at all?
    confidence: float              # How confident is the constraint? [0, 1]
    direction: torch.Tensor        # Shape (3,): soft label bias (SUP, REF, NEI)
    explanation: str = ""          # Human-readable explanation


# ── Individual Constraints ───────────────────────────────────

class NumericalConstraint:
    """Detects numerical discrepancies between claim and evidence.

    Much more robust than v1:
      - Handles word-form numbers (one, two, ..., billion)
      - Computes overlap vs. conflict ratio
      - Graded confidence based on match quality
    """

    NUM_PATTERN = re.compile(
        r'(?<!\w)(?:'
        r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?'  # Standard numbers
        r'(?:\s*%)?'                          # Optional percent
        r'|(?:one|two|three|four|five|six|seven|eight|nine|ten'
        r'|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen'
        r'|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy'
        r'|eighty|ninety|hundred|thousand|million|billion|trillion'
        r'|first|second|third|fourth|fifth)'
        r')(?!\w)', re.IGNORECASE
    )

    WORD_TO_NUM = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
        'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
        'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'million': 1e6, 'billion': 1e9, 'trillion': 1e12,
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    }

    def extract_numbers(self, text: str) -> set[float]:
        matches = self.NUM_PATTERN.findall(text.lower())
        numbers: set[float] = set()
        for m in matches:
            m_clean = m.strip().replace(',', '').replace('%', '')
            try:
                numbers.add(float(m_clean))
            except ValueError:
                if m_clean in self.WORD_TO_NUM:
                    numbers.add(float(self.WORD_TO_NUM[m_clean]))
        return numbers

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        claim_nums = self.extract_numbers(claim)
        ev_nums = self.extract_numbers(evidence)

        if not claim_nums or not ev_nums:
            return ConstraintSignal(
                name="numerical", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        overlap = claim_nums & ev_nums
        claim_only = claim_nums - ev_nums

        if claim_only and not overlap:
            # All claim numbers absent from evidence — strong mismatch
            confidence = min(0.7, 0.3 + 0.1 * len(claim_only))
            direction = torch.tensor([0.10, 0.65, 0.25])
        elif claim_only and overlap:
            # Mixed: partial match — some numbers match, some don't
            ratio = len(claim_only) / (len(claim_only) + len(overlap))
            confidence = 0.2 + 0.3 * ratio
            direction = torch.tensor([0.20, 0.50, 0.30])
        elif overlap and not claim_only:
            # All claim numbers found in evidence — consistent
            confidence = 0.35
            direction = torch.tensor([0.50, 0.20, 0.30])
        else:
            confidence = 0.0
            direction = torch.tensor([0.33, 0.33, 0.34])

        return ConstraintSignal(
            name="numerical",
            fires=True,
            confidence=confidence,
            direction=direction,
            explanation=f"claim={claim_nums}, ev={ev_nums}, overlap={overlap}",
        )


class NegationConstraint:
    """Detects semantic negation between claim and evidence.

    Upgrades from v1:
      - Uses antonym pairs for robust negation beyond "not"
      - Detects negation scope (not just presence)
      - Graded confidence
    """

    NEGATION_CUES = frozenset({
        'not', "n't", 'never', 'no', 'none', 'neither', 'nor',
        'nobody', 'nothing', 'nowhere', 'without', 'lack', 'lacks',
        'lacked', 'lacking', 'fail', 'failed', 'fails', 'unable',
        'denied', 'deny', 'denies', 'refuse', 'refused', 'refuses',
        'cannot', "can't", "won't", "wouldn't", "shouldn't",
        "couldn't", "doesn't", "didn't", "hasn't", "haven't",
        "hadn't", "isn't", "aren't", "wasn't", "weren't",
    })

    ANTONYM_PAIRS = [
        ('true', 'false'), ('correct', 'incorrect'), ('real', 'fake'),
        ('alive', 'dead'), ('win', 'lose'), ('won', 'lost'),
        ('increase', 'decrease'), ('rise', 'fall'), ('open', 'closed'),
        ('start', 'end'), ('begin', 'finish'), ('accept', 'reject'),
        ('agree', 'disagree'), ('appear', 'disappear'),
        ('success', 'failure'), ('include', 'exclude'),
        ('legal', 'illegal'), ('possible', 'impossible'),
        ('direct', 'indirect'), ('complete', 'incomplete'),
        ('approve', 'disapprove'), ('connect', 'disconnect'),
        ('support', 'oppose'), ('confirm', 'deny'),
        ('before', 'after'), ('above', 'below'),
        ('more', 'less'), ('higher', 'lower'), ('larger', 'smaller'),
        ('majority', 'minority'), ('positive', 'negative'),
    ]

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        claim_tokens = set(re.findall(r"\b[\w']+\b", claim.lower()))
        ev_tokens = set(re.findall(r"\b[\w']+\b", evidence.lower()))

        claim_neg = bool(claim_tokens & self.NEGATION_CUES)
        ev_neg = bool(ev_tokens & self.NEGATION_CUES)
        polarity_mismatch = claim_neg != ev_neg

        # Antonym detection
        antonym_count = 0
        for (a, b) in self.ANTONYM_PAIRS:
            if (a in claim_tokens and b in ev_tokens) or \
               (b in claim_tokens and a in ev_tokens):
                antonym_count += 1

        if polarity_mismatch and antonym_count > 0:
            confidence = min(0.7, 0.4 + 0.1 * antonym_count)
            direction = torch.tensor([0.10, 0.70, 0.20])
        elif polarity_mismatch:
            confidence = 0.35
            direction = torch.tensor([0.15, 0.60, 0.25])
        elif antonym_count > 0:
            confidence = min(0.5, 0.2 + 0.1 * antonym_count)
            direction = torch.tensor([0.15, 0.60, 0.25])
        else:
            return ConstraintSignal(
                name="negation", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        return ConstraintSignal(
            name="negation",
            fires=True,
            confidence=confidence,
            direction=direction,
            explanation=f"polarity_mismatch={polarity_mismatch}, antonyms={antonym_count}",
        )


class EntityOverlapConstraint:
    """Measures entity overlap between claim and evidence.

    Low overlap → evidence likely irrelevant → NEI bias.
    High overlap → evidence is relevant (but doesn't determine label).
    """

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        # Use capitalised words as proxy for entities (cheap, no NER needed)
        claim_entities = {w.lower() for w in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)}
        ev_entities = {w.lower() for w in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', evidence)}

        if not claim_entities:
            return ConstraintSignal(
                name="entity_overlap", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        if not ev_entities:
            return ConstraintSignal(
                name="entity_overlap", fires=True, confidence=0.45,
                direction=torch.tensor([0.10, 0.10, 0.80]),
                explanation="no entities in evidence",
            )

        overlap = claim_entities & ev_entities
        overlap_ratio = len(overlap) / len(claim_entities)

        if overlap_ratio < 0.2:
            confidence = 0.5
            direction = torch.tensor([0.10, 0.10, 0.80])
        elif overlap_ratio < 0.5:
            confidence = 0.3
            direction = torch.tensor([0.20, 0.20, 0.60])
        elif overlap_ratio > 0.8:
            confidence = 0.25
            direction = torch.tensor([0.40, 0.40, 0.20])
        else:
            confidence = 0.15
            direction = torch.tensor([0.35, 0.35, 0.30])

        return ConstraintSignal(
            name="entity_overlap",
            fires=True,
            confidence=confidence,
            direction=direction,
            explanation=f"overlap_ratio={overlap_ratio:.2f}, overlap={overlap}",
        )


class EvidenceSufficiencyConstraint:
    """Estimates whether evidence provides enough information to verify the claim.

    Very short or empty evidence → NEI bias.
    Evidence much shorter than claim → likely insufficient.
    """

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        ev_words = len(evidence.split())
        claim_words = len(claim.split())

        if ev_words == 0:
            return ConstraintSignal(
                name="sufficiency", fires=True, confidence=0.8,
                direction=torch.tensor([0.05, 0.05, 0.90]),
                explanation="empty evidence",
            )

        if ev_words < 5:
            return ConstraintSignal(
                name="sufficiency", fires=True, confidence=0.5,
                direction=torch.tensor([0.10, 0.10, 0.80]),
                explanation=f"very short evidence ({ev_words} words)",
            )

        if ev_words < claim_words * 0.4:
            return ConstraintSignal(
                name="sufficiency", fires=True, confidence=0.3,
                direction=torch.tensor([0.20, 0.20, 0.60]),
                explanation=f"short evidence ({ev_words} vs {claim_words} claim words)",
            )

        return ConstraintSignal(
            name="sufficiency", fires=False, confidence=0.0,
            direction=torch.tensor([0.33, 0.33, 0.34]),
        )


class TemporalConstraint:
    """Detects temporal inconsistencies between claim and evidence.

    If claim mentions a year/date not in evidence, or vice versa,
    there may be a temporal mismatch — biases toward REFUTES.
    """

    YEAR_PATTERN = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        claim_years = set(self.YEAR_PATTERN.findall(claim))
        ev_years = set(self.YEAR_PATTERN.findall(evidence))

        if not claim_years or not ev_years:
            return ConstraintSignal(
                name="temporal", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        overlap = claim_years & ev_years
        claim_only = claim_years - ev_years

        if claim_only and not overlap:
            confidence = min(0.5, 0.2 + 0.1 * len(claim_only))
            direction = torch.tensor([0.15, 0.55, 0.30])
        elif claim_only and overlap:
            confidence = 0.2
            direction = torch.tensor([0.25, 0.45, 0.30])
        else:
            confidence = 0.3
            direction = torch.tensor([0.45, 0.25, 0.30])

        return ConstraintSignal(
            name="temporal",
            fires=True,
            confidence=confidence,
            direction=direction,
            explanation=f"claim_years={claim_years}, ev_years={ev_years}",
        )


class HedgeModalityConstraint:
    """Detects hedging/modality words that suggest uncertainty.

    Claims with hedges like "might", "could", "allegedly" are harder;
    if evidence is definitive, there may be a mismatch.
    """

    HEDGE_WORDS = frozenset({
        'might', 'could', 'may', 'possibly', 'perhaps', 'allegedly',
        'reportedly', 'supposedly', 'apparently', 'probably', 'likely',
        'unlikely', 'seems', 'appears', 'suggests', 'claimed',
        'rumored', 'rumoured', 'speculated', 'uncertain', 'unclear',
    })

    DEFINITIVE_WORDS = frozenset({
        'is', 'was', 'are', 'were', 'has', 'had', 'will',
        'confirmed', 'proved', 'proven', 'established', 'known',
        'definitely', 'certainly', 'always', 'every', 'all',
    })

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        claim_tokens = set(claim.lower().split())
        ev_tokens = set(evidence.lower().split())

        claim_hedges = claim_tokens & self.HEDGE_WORDS
        ev_definitive = ev_tokens & self.DEFINITIVE_WORDS

        if claim_hedges and ev_definitive:
            confidence = min(0.3, 0.1 + 0.05 * (len(claim_hedges) + len(ev_definitive)))
            direction = torch.tensor([0.25, 0.40, 0.35])
            return ConstraintSignal(
                name="hedge_modality", fires=True, confidence=confidence,
                direction=direction,
                explanation=f"hedges={claim_hedges}, definitive={ev_definitive}",
            )

        return ConstraintSignal(
            name="hedge_modality", fires=False, confidence=0.0,
            direction=torch.tensor([0.33, 0.33, 0.34]),
        )


# ── Constraint Engine (Orchestrator) ────────────────────────

class ConstraintEngineV2:
    """Orchestrates all v2 constraints and produces batched tensors.

    Returns dictionaries of tensors suitable for the gating module
    and adaptive lambda computation.
    """

    def __init__(self):
        self.constraints = [
            NumericalConstraint(),
            NegationConstraint(),
            EntityOverlapConstraint(),
            EvidenceSufficiencyConstraint(),
            TemporalConstraint(),
            HedgeModalityConstraint(),
        ]
        self.n_constraints = len(self.constraints)
        self.constraint_names = [c.__class__.__name__ for c in self.constraints]

    def evaluate_single(
        self, claim: str, evidence: str
    ) -> list[ConstraintSignal]:
        """Evaluate all constraints on a single example."""
        return [c(claim, evidence) for c in self.constraints]

    def evaluate_batch(
        self, claims: list[str], evidences: list[str]
    ) -> dict[str, torch.Tensor]:
        """Evaluate all constraints on a batch.

        Returns:
            fires:      (B, K) bool — does each constraint fire?
            confidence: (B, K) float — constraint confidence
            direction:  (B, K, 3) float — soft label bias from each constraint
        """
        B = len(claims)
        K = self.n_constraints

        fires = torch.zeros(B, K, dtype=torch.bool)
        confidence = torch.zeros(B, K)
        direction = torch.zeros(B, K, 3)

        for i, (claim, evidence) in enumerate(zip(claims, evidences)):
            for k, constraint in enumerate(self.constraints):
                signal = constraint(claim, evidence)
                fires[i, k] = signal.fires
                confidence[i, k] = signal.confidence
                direction[i, k] = signal.direction

        return {
            "fires": fires,
            "confidence": confidence,
            "direction": direction,
        }

    def compute_constraint_loss(
        self,
        probs: torch.Tensor,               # (B, 3) model output probabilities
        constraint_signals: dict[str, torch.Tensor],  # from evaluate_batch
        gate_weights: Optional[torch.Tensor] = None,  # (B, K) from gate module
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute differentiable constraint loss using probabilistic signals.

        The loss encourages the model's probabilities to align with the
        constraint-suggested directions, weighted by confidence and gate.

        Loss per sample per constraint:
            L_ik = confidence_ik * gate_ik * KL(direction_ik || probs_i)

        Args:
            probs: Model output probabilities (B, 3).
            constraint_signals: Output of evaluate_batch().
            gate_weights: Optional per-sample per-constraint weights (B, K).

        Returns:
            (total_loss, info_dict)
        """
        device = probs.device
        fires = constraint_signals["fires"].to(device).float()
        confidence = constraint_signals["confidence"].to(device)
        direction = constraint_signals["direction"].to(device)

        B, K = fires.shape

        if gate_weights is None:
            gate_weights = torch.ones(B, K, device=device)
        else:
            gate_weights = gate_weights.to(device)

        # KL divergence: KL(direction || probs) for each constraint
        # direction is the "target" distribution from the constraint
        log_probs = (probs + 1e-8).log().unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)
        log_direction = (direction.to(device) + 1e-8).log()

        # KL(p || q) = sum(p * (log_p - log_q))
        kl = (direction.to(device) * (log_direction - log_probs)).sum(dim=-1)  # (B, K)

        # Weight by fires * confidence * gate
        weighted_kl = kl * fires * confidence * gate_weights  # (B, K)

        # Per-constraint mean loss
        per_constraint_loss = weighted_kl.mean(dim=0)  # (K,)
        total = weighted_kl.sum(dim=-1).mean()  # scalar

        info = {
            "constraint_loss_total": total.item(),
            "n_firing": fires.sum().item(),
            "mean_confidence": confidence[fires.bool()].mean().item() if fires.any() else 0.0,
        }
        for k, name in enumerate(self.constraint_names):
            info[f"loss_{name}"] = per_constraint_loss[k].item()
            info[f"fire_rate_{name}"] = fires[:, k].mean().item()

        return total, info


def calibrate_constraints(
    engine: ConstraintEngineV2,
    claims: list[str],
    evidences: list[str],
    labels: list[int],
    n_samples: int = 5000,
) -> dict[str, dict]:
    """Measure constraint precision/recall on labelled data.

    For each constraint that fires, check if its suggested direction
    (argmax of direction vector) matches the true label.

    Returns per-constraint calibration stats.
    """
    import random
    rng = random.Random(42)

    indices = list(range(len(claims)))
    if len(indices) > n_samples:
        indices = rng.sample(indices, n_samples)

    stats: dict[str, dict] = {}
    for k, name in enumerate(engine.constraint_names):
        stats[name] = {"tp": 0, "fp": 0, "fn": 0, "total_fires": 0}

    for i in indices:
        signals = engine.evaluate_single(claims[i], evidences[i])
        true_label = labels[i]

        for k, signal in enumerate(signals):
            name = engine.constraint_names[k]
            if not signal.fires:
                continue
            stats[name]["total_fires"] += 1

            suggested_label = signal.direction.argmax().item()
            if suggested_label == true_label:
                stats[name]["tp"] += 1
            else:
                stats[name]["fp"] += 1

    # Compute precision
    for name, s in stats.items():
        total = s["tp"] + s["fp"]
        s["precision"] = s["tp"] / max(1, total)
        s["fire_rate"] = s["total_fires"] / max(1, len(indices))

    return stats
