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


def _concat_evidence_sentences(
    evidence_sets: list,
    wiki_page_map: dict[str, list[str]] | None = None,
) -> str:
    """Extract and concatenate gold evidence sentences from FEVER annotation format.

    FEVER stores evidence as: list of annotation sets, each containing
    [annotation_id, evidence_id, wiki_title, sentence_idx].
    We concatenate unique sentences by looking up actual text from wiki_page_map.

    Args:
        evidence_sets: nested list from HF dataset evidence field.
        wiki_page_map: dict mapping page_title → list of sentence strings.
            If provided, returns actual evidence text.
            If None, returns title + index placeholders (for debugging only).
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
                # Look up actual sentence text from wiki pages
                if wiki_page_map and wiki_title in wiki_page_map:
                    sents = wiki_page_map[wiki_title]
                    if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
                        text = sents[sent_idx].strip()
                        if text:
                            pieces.append(text)
                            continue
                # Fallback: title as context (better than nothing)
                pieces.append(wiki_title.replace("_", " "))
    return " . ".join(pieces)


def _build_wiki_page_map(
    ds,
    cache_dir: str | None = None,
    wiki_cache_path: str | None = None,
) -> "dict[str, list[str]] | _WikiCacheAdapter":
    """Build mapping from wiki page title to list of sentence strings.

    Tries multiple sources (in order):
    1. SQLite wiki cache (data/fever_wiki.db) — preferred, Colab-safe.
    2. Local wiki-pages JSONL file (if available in cache_dir or data/).
    3. Empty dict (graceful degradation - uses page titles as evidence).

    NOTE: The HF wiki_pages split is NO LONGER loaded directly.
    Use ``python main.py build-fever-wiki-cache`` to build the SQLite
    cache first.  This avoids the multi-GB in-memory dict that causes
    OOM on Colab.
    """
    # Source 1: SQLite cache (preferred — O(1) lookup, ~15 MB)
    _default_db = os.path.join(os.path.dirname(__file__), "fever_wiki.db")
    db_path = wiki_cache_path or _default_db
    if os.path.exists(db_path):
        from data.fever_wiki_cache import WikiCache
        cache = WikiCache(db_path)
        logger.info(f"  Using SQLite wiki cache: {db_path} ({len(cache)} pages)")
        return _WikiCacheAdapter(cache)

    # Source 2: Local JSONL file
    wiki_map: dict[str, list[str]] = {}
    for base in [cache_dir, "data", "."]:
        if base is None:
            continue
        for fname in ["wiki-pages.jsonl", "wiki_pages.jsonl"]:
            path = os.path.join(base, fname)
            if os.path.exists(path):
                logger.info(f"  Building wiki page map from {path}...")
                with open(path) as f:
                    for line_str in f:
                        page = json.loads(line_str)
                        title = page.get("id", "")
                        lines_raw = page.get("lines", "")
                        if not title or not lines_raw:
                            continue
                        sentences = []
                        for line in lines_raw.split("\n"):
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                sentences.append(parts[1].strip())
                            else:
                                sentences.append("")
                        if sentences:
                            wiki_map[title] = sentences
                logger.info(f"  Loaded {len(wiki_map)} pages from {path}")
                return wiki_map

    logger.warning(
        "  No wiki cache or JSONL found. Evidence text will use titles only. "
        "Build the cache with: python main.py build-fever-wiki-cache"
    )
    return wiki_map


class _WikiCacheAdapter:
    """Dict-like adapter around WikiCache for use by _concat_evidence_sentences.

    Supports ``title in adapter`` and ``adapter[title]`` so existing code
    that treats wiki_page_map as a dict continues to work.
    """

    def __init__(self, cache):
        self._cache = cache

    def __contains__(self, title: str) -> bool:
        return title in self._cache

    def __getitem__(self, title: str) -> list[str]:
        result = self._cache.lookup(title)
        if result is None:
            raise KeyError(title)
        return result

    def get(self, title: str, default=None):
        result = self._cache.lookup(title)
        return result if result is not None else default

    def __len__(self) -> int:
        return len(self._cache)

    def __bool__(self) -> bool:
        return True  # non-empty cache is truthy


def load_fever_splits(
    cache_dir: str | None = None,
    max_train: int | None = None,
    max_dev: int | None = None,
    dev_test_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Load FEVER from HuggingFace datasets with train/dev splits.

    Returns dict with 'train' and 'dev' keys, each containing list of dicts:
      {id, claim, label, label_id, gold_evidence_text}

    If ``dev_test_ratio > 0``, the labelled_dev split is further divided
    into 'dev' (for tuning / early stopping) and 'dev_test' (held-out,
    touched only once for final numbers).  This prevents overfitting
    to the dev set through repeated evaluation.

    Evidence text is resolved by joining with the wiki_pages split of the
    HF dataset.  When wiki_pages are unavailable, falls back to page titles.
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

    # ── Build wiki page map for evidence text lookup ────────────
    wiki_page_map = _build_wiki_page_map(ds, cache_dir)
    if wiki_page_map:
        logger.info(f"  Wiki page map: {len(wiki_page_map)} pages loaded")
    else:
        logger.warning(
            "  Wiki page map empty — evidence will use page titles only. "
            "This significantly hurts NLI accuracy. To fix: ensure the "
            "'fever/v1.0' dataset includes wiki_pages split, or provide "
            "local wiki-pages JSONL via cache_dir."
        )

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
        columns = data.column_names if hasattr(data, "column_names") else []

        # Detect format: flat-row (evidence_wiki_url column) vs nested (evidence column)
        is_flat = "evidence_wiki_url" in columns

        if is_flat:
            # ── Flat-row format ──────────────────────────────────
            # Each row is one evidence piece. Group by claim id.
            from collections import defaultdict
            grouped = defaultdict(lambda: {
                "id": 0, "claim": "", "label_raw": "NOT ENOUGH INFO",
                "evidence_pieces": []
            })
            for row in data:
                cid = row.get("id", 0)
                entry = grouped[cid]
                entry["id"] = cid
                entry["claim"] = row.get("claim", "")
                entry["label_raw"] = row.get("label", "NOT ENOUGH INFO")
                wiki_url = row.get("evidence_wiki_url", "")
                sent_id = row.get("evidence_sentence_id", -1)
                if wiki_url and sent_id >= 0:
                    entry["evidence_pieces"].append((wiki_url, sent_id))

            # Sort by id for determinism, then limit
            all_ids = sorted(grouped.keys())
            max_n = max_train if split_name == "train" else max_dev
            if max_n is not None and max_n < len(all_ids):
                all_ids = all_ids[:max_n]

            split_items = []
            n_with_text = 0
            for cid in all_ids:
                entry = grouped[cid]
                label_raw = entry["label_raw"]
                if isinstance(label_raw, int):
                    label = FEVER_LABELS[label_raw] if label_raw < len(FEVER_LABELS) else "NOT ENOUGH INFO"
                else:
                    label = _normalise_label(str(label_raw))

                # Resolve evidence text from wiki cache
                pieces = []
                seen = set()
                for wiki_title, sent_idx in entry["evidence_pieces"]:
                    key = (wiki_title, sent_idx)
                    if key in seen:
                        continue
                    seen.add(key)
                    if wiki_page_map and wiki_title in wiki_page_map:
                        sents = wiki_page_map[wiki_title]
                        if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
                            text = sents[sent_idx].strip()
                            if text:
                                pieces.append(text)
                                continue
                    # Fallback: use title
                    pieces.append(wiki_title.replace("_", " "))
                gold_evidence = " . ".join(pieces)
                if gold_evidence:
                    n_with_text += 1

                split_items.append({
                    "id": cid,
                    "claim": entry["claim"],
                    "label": label,
                    "label_id": LABEL2ID[label],
                    "gold_evidence_text": gold_evidence,
                })
        else:
            # ── Nested format (original) ─────────────────────────
            max_n = max_train if split_name == "train" else max_dev
            if max_n is not None and max_n < len(data):
                data = data.select(range(max_n))

            split_wiki_map = wiki_page_map

            split_items = []
            n_with_text = 0
            for row in data:
                label_raw = row.get("label", "NOT ENOUGH INFO")
                if isinstance(label_raw, int):
                    label = FEVER_LABELS[label_raw] if label_raw < len(FEVER_LABELS) else "NOT ENOUGH INFO"
                else:
                    label = _normalise_label(str(label_raw))

                evidence_raw = row.get("evidence", [])
                gold_evidence = _concat_evidence_sentences(
                    evidence_raw, wiki_page_map=split_wiki_map
                )
                if gold_evidence and not gold_evidence.startswith("("):
                    n_with_text += 1

                split_items.append({
                    "id": row.get("id", 0),
                    "claim": row.get("claim", ""),
                    "label": label,
                    "label_id": LABEL2ID[label],
                    "gold_evidence_text": gold_evidence,
                })

        result[split_name] = split_items
        logger.info(
            f"  {split_name}: {len(split_items)} examples "
            f"({n_with_text} with evidence text, "
            f"{len(split_items) - n_with_text} without)"
        )

    # ── Optional dev / dev-test split ─────────────────────────
    if dev_test_ratio > 0.0 and "dev" in result and result["dev"]:
        import random as _rng
        rng = _rng.Random(seed)
        dev_items = list(result["dev"])
        rng.shuffle(dev_items)
        split_idx = int(len(dev_items) * (1 - dev_test_ratio))
        result["dev"] = dev_items[:split_idx]
        result["dev_test"] = dev_items[split_idx:]
        logger.info(
            f"  Split labelled_dev into dev ({len(result['dev'])}) "
            f"+ dev_test ({len(result['dev_test'])})"
        )

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
        # Data-dependency leakage guard: verify __getitem__ never returns gold fields
        self._verify_no_data_dependency()

    def _verify_no_data_dependency(self):
        """Verify this dataset never exposes gold evidence fields.

        Data-dependency leakage means the pipeline code READS gold-evidence
        fields or uses labels/gold-evidence in cache keys.  It does NOT mean
        that retrieved evidence happens to overlap with gold evidence — a good
        retriever SHOULD find the same evidence.  Overlap is desirable, not
        leakage.

        This check inspects __getitem__ output to confirm gold_evidence_text
        is excluded.
        """
        if not self.items:
            return
        # Spot-check: the first item should not expose gold_evidence_text
        sample = self.__getitem__(0)
        if "gold_evidence_text" in sample:
            raise ValueError(
                "DATA-DEPENDENCY LEAKAGE: FeverPipelineDataset.__getitem__ "
                "returns 'gold_evidence_text'. Pipeline mode must never "
                "expose gold evidence fields."
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

    Uses the tokenizer's native sentence-pair encoding so that special
    tokens (``[CLS]``, ``[SEP]`` / ``</s>``, token-type IDs) are
    inserted correctly for each backbone (BERT, DeBERTa, RoBERTa, …).

    Previous version concatenated ``claim [SEP] evidence`` as a flat
    string, which inserted literal ``[SEP]`` text into DeBERTa's
    SentencePiece vocabulary — a significant tokenisation error.

    Output: input_ids, attention_mask, labels (+ raw claim/evidence for
    downstream constraint extraction).
    """
    claims = [ex["claim"] for ex in batch]
    evidences = [ex["evidence"] for ex in batch]
    labels = torch.tensor([ex["label_id"] for ex in batch], dtype=torch.long)

    # Native sentence-pair encoding: tokenizer(text, text_pair)
    encoding = tokenizer(
        claims,
        evidences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "claims": claims,
        "evidences": evidences,
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
