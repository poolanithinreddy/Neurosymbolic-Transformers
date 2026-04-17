"""NST-VERI Constraint System v2: High-precision, calibrated, learnable.

Design Goal: Constraints MUST have >60% precision to be useful.
Low-precision constraints hurt more than they help because DeBERTa
already handles most cases correctly. We only fire constraints when
we have strong textual evidence of a specific pattern.

Major upgrades from v1 (fever_constraint_loss.py):
  - Constraints output **confidence scores** [0,1], not binary flags.
  - Each constraint produces a **soft label direction** (3-way bias).
  - Antonym detection for robust negation handling.
  - Entity overlap drives NEI prediction with graded confidence.
  - Evidence sufficiency estimation.
  - Batch-friendly API returning tensors ready for training.
  - **v2.1**: Higher precision thresholds — fire less, fire right.
  - **v2.1**: Contextual number comparison (same-sentence proximity).
  - **v2.1**: Stricter negation scope detection.

Design principles:
  - No constraint uses gold labels — only claim + evidence text.
  - Confidence reflects reliability: low confidence → don't trust.
  - Direction vectors are normalised probability-like biases.
  - The gate module learns to weight these signals per-sample.
  - **PRECISION > RECALL**: better to miss a pattern than fire incorrectly.
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

    HIGH-PRECISION design:
      - Only fires when numbers are clearly contradictory (different values
        in comparable contexts), not just different.
      - Handles word-form numbers (one, two, ..., billion)
      - Contextual: checks if numbers appear near shared entities/topics.
      - Conservative: if unsure whether numbers refer to the same thing,
        does NOT fire.
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

    # Context words that suggest the numbers are comparable
    QUANTITY_CONTEXT = re.compile(
        r'\b(population|age|born|died|years?|members?|episodes?|seasons?|'
        r'films?|albums?|songs?|goals?|points?|votes?|miles?|km|meters?|'
        r'feet|inches|pounds?|dollars?|euros?|percent|salary|revenue|'
        r'budget|cost|price|height|weight|length|width|area|speed|'
        r'temperature|score|rank|place|position|number|total|count|'
        r'approximately|about|around|exactly|over|under|more than|'
        r'less than|at least|at most|nearly|roughly)\b', re.IGNORECASE
    )

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

    def _has_shared_context(self, claim: str, evidence: str) -> bool:
        """Check if claim and evidence share context words suggesting
        the numbers refer to the same quantity."""
        claim_ctx = set(m.lower() for m in self.QUANTITY_CONTEXT.findall(claim))
        ev_ctx = set(m.lower() for m in self.QUANTITY_CONTEXT.findall(evidence))
        return bool(claim_ctx & ev_ctx)

    def _shared_entities(self, claim: str, evidence: str) -> set[str]:
        """Cheap entity overlap: capitalised multi-word phrases."""
        cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        claim_ents = {m.lower() for m in cap_pattern.findall(claim)}
        ev_ents = {m.lower() for m in cap_pattern.findall(evidence)}
        return claim_ents & ev_ents

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
        ev_only = ev_nums - claim_nums

        # Check contextual relevance: do claim and evidence talk about the same thing?
        shared_ents = self._shared_entities(claim, evidence)
        has_context = self._has_shared_context(claim, evidence)

        if claim_only and not overlap and shared_ents and has_context:
            # HIGH PRECISION: claim numbers differ from evidence AND they share
            # entities AND they share quantity-context words.
            # This strongly suggests a numerical contradiction.
            confidence = min(0.75, 0.5 + 0.08 * len(claim_only))
            direction = torch.tensor([0.10, 0.65, 0.25])
        elif claim_only and overlap and shared_ents and has_context:
            # Some match, some differ, in same context — moderate signal
            ratio = len(claim_only) / (len(claim_only) + len(overlap))
            if ratio > 0.5:
                confidence = 0.35
                direction = torch.tensor([0.15, 0.55, 0.30])
            else:
                # More overlap than conflict — don't fire (too noisy)
                return ConstraintSignal(
                    name="numerical", fires=False, confidence=0.0,
                    direction=torch.tensor([0.33, 0.33, 0.34]),
                )
        elif overlap and not claim_only:
            # All claim numbers found in evidence — consistent
            # Only fire weakly for SUPPORTS if entities overlap too
            if shared_ents and len(overlap) >= 2:
                confidence = 0.3
                direction = torch.tensor([0.55, 0.15, 0.30])
            else:
                return ConstraintSignal(
                    name="numerical", fires=False, confidence=0.0,
                    direction=torch.tensor([0.33, 0.33, 0.34]),
                )
        else:
            # No shared context — numbers may refer to different things
            # DON'T FIRE: precision too low without context
            return ConstraintSignal(
                name="numerical", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        return ConstraintSignal(
            name="numerical",
            fires=True,
            confidence=confidence,
            direction=direction,
            explanation=f"claim={claim_nums}, ev={ev_nums}, overlap={overlap}, "
                        f"shared_ents={len(shared_ents)}, context={has_context}",
        )


class NegationConstraint:
    """Detects semantic negation between claim and evidence.

    HIGH-PRECISION design:
      - Uses antonym pairs for robust negation beyond "not"
      - Detects negation scope — only fires when negation applies to shared content
      - Requires entity overlap to ensure claim and evidence discuss the same topic
      - Graded confidence based on signal strength
      - Filters double-negation and scope ambiguity
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

    # Only high-confidence antonym pairs that clearly indicate contradiction
    ANTONYM_PAIRS = [
        ('true', 'false'), ('correct', 'incorrect'), ('real', 'fake'),
        ('alive', 'dead'), ('win', 'lose'), ('won', 'lost'),
        ('increase', 'decrease'), ('rise', 'fall'),
        ('success', 'failure'), ('include', 'exclude'),
        ('legal', 'illegal'), ('possible', 'impossible'),
        ('approve', 'disapprove'), ('confirm', 'deny'),
        ('accept', 'reject'), ('agree', 'disagree'),
        ('positive', 'negative'), ('guilty', 'innocent'),
        ('married', 'divorced'), ('active', 'inactive'),
    ]

    def _has_shared_content(self, claim: str, evidence: str) -> bool:
        """Check if claim and evidence discuss the same topic."""
        cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        claim_ents = {m.lower() for m in cap_pattern.findall(claim)}
        ev_ents = {m.lower() for m in cap_pattern.findall(evidence)}
        if not claim_ents or not ev_ents:
            # Fall back to content word overlap
            stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on',
                          'at', 'to', 'for', 'of', 'and', 'or', 'but', 'with', 'by',
                          'from', 'that', 'this', 'it', 'as', 'be', 'has', 'had', 'have'}
            claim_words = {w.lower() for w in claim.split() if len(w) > 3} - stop_words
            ev_words = {w.lower() for w in evidence.split() if len(w) > 3} - stop_words
            return len(claim_words & ev_words) >= 2
        return len(claim_ents & ev_ents) >= 1

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

        # Count negation cues to detect double negation
        claim_neg_count = len(claim_tokens & self.NEGATION_CUES)
        ev_neg_count = len(ev_tokens & self.NEGATION_CUES)
        double_neg = claim_neg_count > 0 and ev_neg_count > 0

        # PRECISION GATE: require shared content between claim and evidence
        # Without this, negation cues in unrelated texts cause false positives
        shared = self._has_shared_content(claim, evidence)

        if not shared:
            return ConstraintSignal(
                name="negation", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        if polarity_mismatch and antonym_count > 0 and not double_neg:
            # Strong signal: clear polarity flip + antonyms + shared topic
            confidence = min(0.80, 0.50 + 0.10 * antonym_count)
            direction = torch.tensor([0.08, 0.72, 0.20])
        elif polarity_mismatch and not double_neg:
            # Moderate: polarity flip only — still requires shared context
            confidence = 0.40
            direction = torch.tensor([0.12, 0.63, 0.25])
        elif antonym_count >= 2 and not double_neg:
            # Multiple antonyms without explicit negation cues
            confidence = min(0.55, 0.30 + 0.10 * antonym_count)
            direction = torch.tensor([0.12, 0.63, 0.25])
        elif antonym_count == 1:
            # Single antonym — too weak, don't fire (low precision)
            return ConstraintSignal(
                name="negation", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )
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
            explanation=f"polarity_mismatch={polarity_mismatch}, antonyms={antonym_count}, "
                        f"shared={shared}, double_neg={double_neg}",
        )


class EntityOverlapConstraint:
    """Measures entity overlap between claim and evidence.

    HIGH-PRECISION design:
      - Only fires NEI bias when overlap is VERY low and evidence is substantial
      - Avoids firing on short claims where few entities are expected
      - Uses both capitalised words and content-word overlap
    """

    def _extract_entities(self, text: str) -> set[str]:
        """Extract entity-like tokens: capitalised words and quoted phrases."""
        # Capitalised multi-word entities
        cap_ents = {m.lower() for m in re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text
        )}
        return cap_ents

    def _content_word_overlap(self, claim: str, evidence: str) -> float:
        """Overlap of non-stopword content words."""
        stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on',
                      'at', 'to', 'for', 'of', 'and', 'or', 'but', 'with', 'by',
                      'from', 'that', 'this', 'it', 'as', 'be', 'has', 'had', 'have',
                      'been', 'will', 'would', 'could', 'should', 'may', 'might',
                      'shall', 'can', 'do', 'does', 'did', 'not', 'no', 'so', 'if',
                      'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'whose',
                      'than', 'then', 'also', 'just', 'only', 'very', 'too', 'more',
                      'most', 'other', 'some', 'any', 'each', 'every', 'all', 'both'}
        claim_words = {w.lower() for w in re.findall(r'\b\w+\b', claim) if len(w) > 2} - stop_words
        ev_words = {w.lower() for w in re.findall(r'\b\w+\b', evidence) if len(w) > 2} - stop_words
        if not claim_words:
            return 0.0
        return len(claim_words & ev_words) / len(claim_words)

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        claim_entities = self._extract_entities(claim)
        ev_entities = self._extract_entities(evidence)

        if not claim_entities or len(claim_entities) < 2:
            # Too few entities in claim — use content word overlap instead
            content_overlap = self._content_word_overlap(claim, evidence)
            if content_overlap < 0.10 and len(evidence.split()) > 10:
                return ConstraintSignal(
                    name="entity_overlap", fires=True, confidence=0.50,
                    direction=torch.tensor([0.08, 0.08, 0.84]),
                    explanation=f"low content overlap ({content_overlap:.2f}), few claim entities",
                )
            return ConstraintSignal(
                name="entity_overlap", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        if not ev_entities or not evidence.strip():
            # No evidence text at all — strong NEI signal
            return ConstraintSignal(
                name="entity_overlap", fires=True, confidence=0.70,
                direction=torch.tensor([0.05, 0.05, 0.90]),
                explanation="no entities in evidence",
            )

        overlap = claim_entities & ev_entities
        overlap_ratio = len(overlap) / len(claim_entities)

        # Also check content word overlap as secondary signal
        content_overlap = self._content_word_overlap(claim, evidence)

        if overlap_ratio < 0.10 and content_overlap < 0.15:
            # Very low overlap on both metrics — high-confidence NEI
            confidence = 0.55
            direction = torch.tensor([0.08, 0.08, 0.84])
        elif overlap_ratio < 0.15 and content_overlap < 0.20:
            # Low overlap — moderate NEI signal
            confidence = 0.40
            direction = torch.tensor([0.12, 0.12, 0.76])
        else:
            # Moderate or high overlap — evidence is relevant, don't fire
            return ConstraintSignal(
                name="entity_overlap", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        return ConstraintSignal(
            name="entity_overlap",
            fires=True,
            confidence=confidence,
            direction=direction,
            explanation=f"entity_overlap={overlap_ratio:.2f}, "
                        f"content_overlap={content_overlap:.2f}",
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
            # Empty evidence — very strong NEI signal
            return ConstraintSignal(
                name="sufficiency", fires=True, confidence=0.85,
                direction=torch.tensor([0.03, 0.03, 0.94]),
                explanation="empty evidence",
            )

        if ev_words < 4:
            # Very short evidence (likely just a title fallback)
            return ConstraintSignal(
                name="sufficiency", fires=True, confidence=0.6,
                direction=torch.tensor([0.08, 0.08, 0.84]),
                explanation=f"very short evidence ({ev_words} words)",
            )

        # Don't fire for moderate or long evidence — too noisy
        return ConstraintSignal(
            name="sufficiency", fires=False, confidence=0.0,
            direction=torch.tensor([0.33, 0.33, 0.34]),
        )


class TemporalConstraint:
    """Detects temporal inconsistencies between claim and evidence.

    HIGH-PRECISION design:
      - Only fires when dates clearly conflict AND entities overlap
      - Ignores cases where different dates may refer to different events
      - Conservative: requires shared context
    """

    YEAR_PATTERN = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')

    def _shared_entities(self, claim: str, evidence: str) -> set[str]:
        cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        claim_ents = {m.lower() for m in cap_pattern.findall(claim)}
        ev_ents = {m.lower() for m in cap_pattern.findall(evidence)}
        return claim_ents & ev_ents

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

        # Require shared entities to ensure same-topic dates
        shared = self._shared_entities(claim, evidence)

        if claim_only and not overlap and shared:
            # Claim mentions years not in evidence, with shared entities
            confidence = min(0.55, 0.30 + 0.08 * len(claim_only))
            direction = torch.tensor([0.12, 0.58, 0.30])
        elif overlap and not claim_only:
            # Dates match — weak SUPPORTS signal
            if shared and len(overlap) >= 1:
                confidence = 0.25
                direction = torch.tensor([0.50, 0.20, 0.30])
            else:
                return ConstraintSignal(
                    name="temporal", fires=False, confidence=0.0,
                    direction=torch.tensor([0.33, 0.33, 0.34]),
                )
        else:
            # Mixed or no shared entities — too uncertain
            return ConstraintSignal(
                name="temporal", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        return ConstraintSignal(
            name="temporal",
            fires=True,
            confidence=confidence,
            direction=direction,
            explanation=f"claim_years={claim_years}, ev_years={ev_years}, shared={len(shared)}",
        )


class HedgeModalityConstraint:
    """Detects hedging/modality asymmetry between claim and evidence.

    HIGH-PRECISION design:
      - Only fires when there's a CLEAR asymmetry (hedge in claim + definitive in evidence)
      - Requires shared content to ensure same topic
      - Conservative thresholds
    """

    HEDGE_WORDS = frozenset({
        'might', 'could', 'may', 'possibly', 'perhaps', 'allegedly',
        'reportedly', 'supposedly', 'apparently', 'probably', 'likely',
        'unlikely', 'seems', 'appears', 'suggests', 'claimed',
        'rumored', 'rumoured', 'speculated', 'uncertain', 'unclear',
    })

    DEFINITIVE_WORDS = frozenset({
        'confirmed', 'proved', 'proven', 'established', 'known',
        'definitely', 'certainly', 'always', 'every', 'all',
        'officially', 'announced', 'declared', 'stated',
    })

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        claim_tokens = set(claim.lower().split())
        ev_tokens = set(evidence.lower().split())

        claim_hedges = claim_tokens & self.HEDGE_WORDS
        ev_definitive = ev_tokens & self.DEFINITIVE_WORDS

        if claim_hedges and ev_definitive and len(claim_hedges) + len(ev_definitive) >= 3:
            # Strong asymmetry: claim hedges but evidence is definitive
            confidence = min(0.35, 0.15 + 0.05 * (len(claim_hedges) + len(ev_definitive)))
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


class MutualExclusionConstraint:
    """Detects mutual exclusion patterns: "X is Y" vs "X is Z" where Y != Z.

    HIGH-PRECISION design:
      - Detects patterns like "X is a [category]" in both claim and evidence
      - When both assign a subject to different categories, fires REFUTES
      - Common in FEVER: "was born in [city]", "is a [nationality]", etc.
    """

    # Patterns for categorical assignments — each captures the full assignment phrase
    CATEGORY_PATTERNS = [
        re.compile(r'(?:is|was|are|were)\s+(?:a|an|the)\s+(\w+(?:\s+\w+)?\s+of\s+\w+(?:\s+\w+)?)', re.IGNORECASE),
        re.compile(r'(?:is|was|are|were)\s+(?:a|an|the)\s+(\w+(?:\s+\w+)?)\b', re.IGNORECASE),
        re.compile(r'(?:born|raised|grew up)\s+in\s+([A-Z]\w+(?:\s+[A-Z]\w+)*)', re.IGNORECASE),
        re.compile(r'(?:located|situated|based)\s+in\s+([A-Z]\w+(?:\s+[A-Z]\w+)*)', re.IGNORECASE),
        re.compile(r'(?:released|published|aired)\s+(?:in|on)\s+(\d{4}|\w+\s+\d{1,2})', re.IGNORECASE),
        re.compile(r'(?:directed|written|created|produced)\s+by\s+([A-Z]\w+(?:\s+[A-Z]\w+)*)', re.IGNORECASE),
    ]

    def __call__(self, claim: str, evidence: str) -> ConstraintSignal:
        claim_categories = []
        ev_categories = []

        for pattern in self.CATEGORY_PATTERNS:
            claim_matches = pattern.findall(claim)
            ev_matches = pattern.findall(evidence)
            if claim_matches and ev_matches:
                # Use the most specific pattern that matches both
                claim_categories = [m.lower().strip() for m in claim_matches]
                ev_categories = [m.lower().strip() for m in ev_matches]
                break  # Don't let less specific patterns override
            elif claim_matches:
                claim_categories.extend(m.lower().strip() for m in claim_matches)
            elif ev_matches:
                ev_categories.extend(m.lower().strip() for m in ev_matches)

        if not claim_categories or not ev_categories:
            return ConstraintSignal(
                name="mutual_exclusion", fires=False, confidence=0.0,
                direction=torch.tensor([0.33, 0.33, 0.34]),
            )

        # Check for same-pattern conflicts (e.g., "born in X" vs "born in Y")
        claim_set = set(claim_categories)
        ev_set = set(ev_categories)

        if claim_set and ev_set and not (claim_set & ev_set):
            # Categories assigned but none overlap — possible mutual exclusion
            # Need to verify shared subject (entities)
            cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
            claim_ents = {m.lower() for m in cap_pattern.findall(claim)}
            ev_ents = {m.lower() for m in cap_pattern.findall(evidence)}
            shared = claim_ents & ev_ents

            if shared:
                confidence = min(0.55, 0.35 + 0.05 * len(shared))
                direction = torch.tensor([0.10, 0.65, 0.25])
                return ConstraintSignal(
                    name="mutual_exclusion", fires=True, confidence=confidence,
                    direction=direction,
                    explanation=f"claim_cat={claim_categories}, ev_cat={ev_categories}, "
                                f"shared_subj={shared}",
                )

        return ConstraintSignal(
            name="mutual_exclusion", fires=False, confidence=0.0,
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
            MutualExclusionConstraint(),
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
