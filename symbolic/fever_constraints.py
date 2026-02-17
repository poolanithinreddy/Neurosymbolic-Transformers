"""Structured fact extraction for FEVER neuro-symbolic constraints.

Extracts signals from claim + evidence text that drive constraint logic:
  1. Numbers (integers, floats, years)
  2. Dates (year patterns, month-day patterns)
  3. Negation cues ("not", "never", "no", "denied", etc.)
  4. Entity overlap (named entity / noun-phrase overlap score)

These signals are NOISY by design — they come from lightweight regex/heuristics,
not from a perfect oracle.  The constraints operate on probabilities, so
approximate extraction is sufficient.

The extractor does NOT use gold labels — only claim + evidence text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Patterns ─────────────────────────────────────────────────

# Numbers: integers, decimals, percentages (with optional commas)
_NUM_PATTERN = re.compile(
    r"(?<!\w)"
    r"-?(?:\d{1,3}(?:,\d{3})*|\d+)"
    r"(?:\.\d+)?"
    r"(?:\s*%|(?:\s+(?:million|billion|trillion|thousand|hundred)))?"
    r"(?!\w)",
    re.IGNORECASE,
)

# Dates: year patterns (1000-2099), month-day patterns
_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_DATE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b",
    re.IGNORECASE,
)

# Negation cues
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "none", "nobody",
    "nothing", "nowhere", "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "doesn't", "didn't", "hasn't",
    "haven't", "hadn't", "isn't", "aren't", "wasn't", "weren't",
    "denied", "refused", "rejected", "false", "incorrect",
    "untrue", "inaccurate", "wrong", "misleading",
})

# Simple "noun phrase" pattern: consecutive capitalized words
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


@dataclass
class StructuredFacts:
    """Extracted structured signals from claim + evidence pair.

    All fields are populated by lightweight heuristics (no model inference).
    """

    # Numbers found in claim and evidence
    numbers_claim: list[str] = field(default_factory=list)
    numbers_evidence: list[str] = field(default_factory=list)

    # Dates/years found
    dates_claim: list[str] = field(default_factory=list)
    dates_evidence: list[str] = field(default_factory=list)

    # Negation
    negation_claim: bool = False
    negation_evidence: bool = False
    negation_mismatch: bool = False  # one has negation, other doesn't

    # Entity overlap
    entities_claim: list[str] = field(default_factory=list)
    entities_evidence: list[str] = field(default_factory=list)
    entity_overlap_score: float = 0.0  # Jaccard similarity of entity sets

    # Derived conflict signals
    number_contradiction: bool = False  # claim and evidence have DIFFERENT numbers
    date_contradiction: bool = False    # claim and evidence have DIFFERENT dates

    def to_dict(self) -> dict:
        return {
            "numbers_claim": self.numbers_claim,
            "numbers_evidence": self.numbers_evidence,
            "dates_claim": self.dates_claim,
            "dates_evidence": self.dates_evidence,
            "negation_claim": self.negation_claim,
            "negation_evidence": self.negation_evidence,
            "negation_mismatch": self.negation_mismatch,
            "entities_claim": list(self.entities_claim),
            "entities_evidence": list(self.entities_evidence),
            "entity_overlap_score": round(self.entity_overlap_score, 4),
            "number_contradiction": self.number_contradiction,
            "date_contradiction": self.date_contradiction,
        }


def _extract_numbers(text: str) -> list[str]:
    """Extract number strings from text."""
    return _NUM_PATTERN.findall(text)


def _normalise_number(s: str) -> float | None:
    """Try to parse a number string to float."""
    s = s.strip().replace(",", "").replace("%", "")
    for suffix, multiplier in [("million", 1e6), ("billion", 1e9),
                                ("trillion", 1e12), ("thousand", 1e3),
                                ("hundred", 1e2)]:
        if suffix in s.lower():
            s = re.sub(rf"\s*{suffix}", "", s, flags=re.IGNORECASE).strip()
            try:
                return float(s) * multiplier
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_dates(text: str) -> list[str]:
    """Extract date/year strings from text."""
    years = _YEAR_PATTERN.findall(text)
    dates = _DATE_PATTERN.findall(text)
    return years + dates


def _has_negation(text: str) -> bool:
    """Check if text contains negation cues."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(words & _NEGATION_WORDS)


def _extract_entities(text: str) -> list[str]:
    """Extract capitalized multi-word spans as entity candidates."""
    return _ENTITY_PATTERN.findall(text)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union) if union else 0.0


def _numbers_conflict(nums_a: list[str], nums_b: list[str]) -> bool:
    """Check if claim and evidence have conflicting numbers.

    Returns True if both have numbers AND at least one number in the claim
    doesn't appear in the evidence numbers (suggesting a factual mismatch).
    """
    if not nums_a or not nums_b:
        return False

    # Normalise to floats for comparison
    vals_a = set()
    for n in nums_a:
        v = _normalise_number(n)
        if v is not None:
            vals_a.add(v)

    vals_b = set()
    for n in nums_b:
        v = _normalise_number(n)
        if v is not None:
            vals_b.add(v)

    if not vals_a or not vals_b:
        return False

    # Conflict: claim has numbers not present in evidence
    return bool(vals_a - vals_b)


def _dates_conflict(dates_a: list[str], dates_b: list[str]) -> bool:
    """Check if claim and evidence have conflicting dates."""
    if not dates_a or not dates_b:
        return False
    set_a = set(d.strip().lower() for d in dates_a)
    set_b = set(d.strip().lower() for d in dates_b)
    return bool(set_a - set_b)


def extract_structured_facts(claim: str, evidence: str) -> StructuredFacts:
    """Extract structured facts from a claim-evidence pair.

    This is the main entry point. All extraction is rule-based (no model).

    Args:
        claim: the claim text.
        evidence: the evidence text.

    Returns:
        StructuredFacts dataclass with all extracted signals.
    """
    facts = StructuredFacts()

    # Numbers
    facts.numbers_claim = _extract_numbers(claim)
    facts.numbers_evidence = _extract_numbers(evidence)
    facts.number_contradiction = _numbers_conflict(
        facts.numbers_claim, facts.numbers_evidence
    )

    # Dates
    facts.dates_claim = _extract_dates(claim)
    facts.dates_evidence = _extract_dates(evidence)
    facts.date_contradiction = _dates_conflict(
        facts.dates_claim, facts.dates_evidence
    )

    # Negation
    facts.negation_claim = _has_negation(claim)
    facts.negation_evidence = _has_negation(evidence)
    facts.negation_mismatch = facts.negation_claim != facts.negation_evidence

    # Entity overlap
    facts.entities_claim = _extract_entities(claim)
    facts.entities_evidence = _extract_entities(evidence)
    set_claim = set(e.lower() for e in facts.entities_claim)
    set_evidence = set(e.lower() for e in facts.entities_evidence)
    facts.entity_overlap_score = _jaccard(set_claim, set_evidence)

    return facts


def extract_batch_facts(
    claims: list[str],
    evidences: list[str],
) -> list[StructuredFacts]:
    """Extract structured facts for a batch of claim-evidence pairs."""
    return [
        extract_structured_facts(c, e)
        for c, e in zip(claims, evidences)
    ]
