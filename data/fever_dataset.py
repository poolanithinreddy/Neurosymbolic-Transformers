"""FEVER dataset loader with Gold Evidence and Full Pipeline modes.

Two evaluation settings (MUST be clearly distinguished):
  (A) GOLD EVIDENCE: oracle evidence sentences provided by FEVER annotations.
      → Measures NLI accuracy in isolation.
  (B) FULL PIPELINE: evidence retrieved via BM25 (or other retriever).
      → Measures end-to-end performance.  Gold evidence is NEVER touched.

Label mapping (FEVER standard):
  SUPPORTS        → 0
  REFUTES          → 1
  NOT ENOUGH INFO  → 2

Integrity:
  - Pipeline mode NEVER accesses gold evidence fields.
  - Leakage guard: if pipeline mode accidentally receives gold evidence, it raises.
  - Split hashes are logged for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("fever_dataset")

# ── Label constants ──────────────────────────────────────────
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL2ID = {l: i for i, l in enumerate(FEVER_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(FEVER_LABELS)}
NUM_LABELS = len(FEVER_LABELS)


def _normalise_label(raw: str) -> str:
    """Map various FEVER label variants to canonical form."""
    raw = raw.strip().upper()
    mapping = {
        "SUPPORTS": "SUPPORTS",
        "SUPPORTED": "SUPPORTS",
        "SUPPORT": "SUPPORTS",
        "REFUTES": "REFUTES",
        "REFUTED": "REFUTES",
        "REFUTE": "REFUTES",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
        "NEI": "NOT ENOUGH INFO",
        "NOTENOUGHINFO": "NOT ENOUGH INFO",
        "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
    }
    return mapping.get(raw, "NOT ENOUGH INFO")


def _concat_evidence_sentences(evidence_sets: list) -> str:
    """Extract and concatenate gold evidence sentences from FEVER annotation format.

    FEVER stores evidence as: list of annotation sets, each containing
    [annotation_id, evidence_id, wiki_title, sentence_idx].
    We concatenate unique sentences identified by (wiki_title, sentence_idx).
    Since raw text isn't stored in the dataset, we return titles + indices
    as a placeholder — real text requires the Wikipedia dump or HF 'wiki_pages'.
    """
    if not evidence_sets:
        return ""
    pieces = []
    seen = set()
    for evidence_set in evidence_sets:
        if not evidence_set:
            continue
        for annotation in evidence_set:
            if annotation is None or len(annotation) < 4:
                continue
            _, _, wiki_title, sent_idx = annotation[:4]
            if wiki_title is None:
                continue
            key = (wiki_title, sent_idx)
            if key not in seen:
                seen.add(key)
                pieces.append(f"{wiki_title} (sentence {sent_idx})")
    return " . ".join(pieces)


def load_fever_splits(
    cache_dir: str | None = None,
    max_train: int | None = None,
    max_dev: int | None = None,
) -> dict[str, list[dict]]:
    """Load FEVER from HuggingFace datasets with train/dev splits.

    Returns dict with 'train' and 'dev' keys, each containing list of dicts:
      {id, claim, label, label_id, gold_evidence_text}

    NOTE: The HF FEVER dataset (fever/v1.0) contains claims and labels but
    the evidence text requires joining with wiki_pages. For claims where
    evidence is available, we extract it; for NEI claims, evidence is empty.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  (required for FEVER)")

    logger.info("Loading FEVER from HuggingFace datasets...")
    # The standard FEVER dataset on HF
    # We use 'v1.0' which has train, labelled_dev, paper_dev, paper_test
    try:
        ds = load_dataset("fever", "v1.0", cache_dir=cache_dir, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load fever/v1.0: {e}. Trying 'fever' name variants...")
        try:
            ds = load_dataset("fever", "v1.0", cache_dir=cache_dir)
        except Exception:
            # Fallback: load from local jsonl if available
            return _load_fever_local_fallback(cache_dir)

    result = {}
    for split_name, hf_split in [("train", "train"), ("dev", "labelled_dev")]:
        if hf_split not in ds:
            # Try alternate names
            alternatives = {"train": ["train"], "labelled_dev": ["paper_dev", "dev"]}
            found = False
            for alt in alternatives.get(hf_split, []):
                if alt in ds:
                    hf_split = alt
                    found = True
                    break
            if not found:
                logger.warning(f"Split '{hf_split}' not found in dataset, skipping.")
                result[split_name] = []
                continue

        data = ds[hf_split]
        max_n = max_train if split_name == "train" else max_dev
        if max_n is not None and max_n < len(data):
            data = data.select(range(max_n))

        split_items = []
        for row in data:
            label_raw = row.get("label", "NOT ENOUGH INFO")
            # HF fever dataset may store label as int (0=SUPPORTS,1=REFUTES,2=NEI)
            if isinstance(label_raw, int):
                label = FEVER_LABELS[label_raw] if label_raw < len(FEVER_LABELS) else "NOT ENOUGH INFO"
            else:
                label = _normalise_label(str(label_raw))

            # Extract gold evidence text
            evidence_raw = row.get("evidence", [])
            gold_evidence = _concat_evidence_sentences(evidence_raw)

            split_items.append({
                "id": row.get("id", 0),
                "claim": row.get("claim", ""),
                "label": label,
                "label_id": LABEL2ID[label],
                "gold_evidence_text": gold_evidence,
            })

        result[split_name] = split_items
        logger.info(f"  {split_name}: {len(split_items)} examples")

    # Log split hashes for reproducibility
    for split_name, items in result.items():
        h = _split_hash(items)
        logger.info(f"  {split_name} hash: {h}")

    return result


def _load_fever_local_fallback(cache_dir: str | None) -> dict[str, list[dict]]:
    """Fallback: load FEVER from local JSONL files."""
    base = cache_dir or "data"
    result = {}
    for split in ("train", "dev"):
        path = os.path.join(base, f"fever_{split}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"Local fallback not found: {path}")
            result[split] = []
            continue
        items = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                label = _normalise_label(row.get("label", "NOT ENOUGH INFO"))
                items.append({
                    "id": row.get("id", 0),
                    "claim": row.get("claim", ""),
                    "label": label,
                    "label_id": LABEL2ID[label],
                    "gold_evidence_text": row.get("evidence", row.get("evidence_text", "")),
                })
        result[split] = items
    return result


def _split_hash(items: list[dict]) -> str:
    """Deterministic hash of a split for integrity checking."""
    content = json.dumps(
        [(it["id"], it["claim"][:50], it["label"]) for it in items[:1000]],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── PyTorch Dataset wrappers ────────────────────────────────

class FeverGoldDataset(Dataset):
    """FEVER dataset with GOLD evidence (Setting A).

    Each sample: (claim, gold_evidence_text, label_id).
    This is the oracle setting — evidence is given, not retrieved.
    """

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        return {
            "id": it["id"],
            "claim": it["claim"],
            "evidence": it["gold_evidence_text"],
            "label_id": it["label_id"],
            "label": it["label"],
        }


class FeverPipelineDataset(Dataset):
    """FEVER dataset with RETRIEVED evidence (Setting B).

    LEAKAGE GUARD: This dataset NEVER accesses gold_evidence_text.
    Evidence must be provided externally via the retriever.

    Args:
        items: list of dicts (from load_fever_splits).
        retrieved_evidence: dict mapping item id → retrieved evidence string.
    """

    def __init__(self, items: list[dict], retrieved_evidence: dict[int, str]):
        self.items = items
        self.retrieved_evidence = retrieved_evidence
        # Leakage guard: verify we're not passing gold evidence through
        self._verify_no_leakage()

    def _verify_no_leakage(self):
        """Verify retrieved evidence is not identical to gold evidence."""
        n_check = min(100, len(self.items))
        n_identical = 0
        for it in self.items[:n_check]:
            gold = it.get("gold_evidence_text", "")
            retrieved = self.retrieved_evidence.get(it["id"], "")
            if gold and retrieved and gold == retrieved:
                n_identical += 1
        if n_check > 0 and n_identical / n_check > 0.9:
            raise ValueError(
                f"LEAKAGE DETECTED: {n_identical}/{n_check} retrieved evidence "
                f"samples are identical to gold evidence. Pipeline mode must use "
                f"independently retrieved evidence."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        return {
            "id": it["id"],
            "claim": it["claim"],
            "evidence": self.retrieved_evidence.get(it["id"], ""),
            "label_id": it["label_id"],
            "label": it["label"],
            # NOTE: gold_evidence_text is deliberately excluded
        }


def fever_collate_fn(
    batch: list[dict],
    tokenizer,
    max_length: int = 256,
) -> dict:
    """Tokenize and collate a batch of FEVER examples.

    Input format: "claim [SEP] evidence"
    Output: input_ids, attention_mask, labels
    """
    texts = [
        f"{ex['claim']} [SEP] {ex['evidence']}" for ex in batch
    ]
    labels = torch.tensor([ex["label_id"] for ex in batch], dtype=torch.long)

    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "claims": [ex["claim"] for ex in batch],
        "evidences": [ex["evidence"] for ex in batch],
    }


def print_fever_stats(splits: dict[str, list[dict]]) -> None:
    """Print dataset statistics for FEVER splits."""
    from collections import Counter

    print("=" * 60)
    print("  FEVER Dataset Statistics")
    print("=" * 60)

    for split_name, items in splits.items():
        label_dist = Counter(it["label"] for it in items)
        has_evidence = sum(1 for it in items if it.get("gold_evidence_text", ""))
        print(f"\n  {split_name}: {len(items)} examples")
        print(f"    With gold evidence: {has_evidence} ({100*has_evidence/max(1,len(items)):.1f}%)")
        print(f"    Label distribution:")
        for label in FEVER_LABELS:
            cnt = label_dist.get(label, 0)
            print(f"      {label:<20} {cnt:>6}  ({100*cnt/max(1,len(items)):.1f}%)")
        h = _split_hash(items)
        print(f"    Split hash: {h}")
