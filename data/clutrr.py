"""CLUTRR dataset loader for natural-language kinship reasoning.

CLUTRR (Compositional Language Understanding and Reasoning with Textual
Relational Data) is a benchmark by Sinha et al. (2019, EMNLP) that tests
systematic compositional generalisation on kinship relations expressed
in natural language stories.

We use it as a realistic benchmark alongside our synthetic kinship task.
The key advantage: natural language input (not template-generated), which
tests whether symbolic constraints help even with noisy perception.

Access:
    The dataset is available via HuggingFace:
        pip install datasets
        from datasets import load_dataset
        ds = load_dataset("CLUTRR/v1", "gen_train23_test2to10")

    If HuggingFace is unavailable, we fall back to a small bundled sample.

Constraints:
    Same chain-length consistency rules as our synthetic kinship benchmark:
    - depth 1: parent/child
    - depth 2: grandparent/grandchild/sibling
    - depth 3+: ancestor/descendant/sibling

Note: We do NOT claim SOTA on CLUTRR. We use it to show that symbolic
constraints improve generalisation for a fixed architecture, comparing
neural vs. NST-CEGIS on the same Transformer backbone.
"""

from __future__ import annotations

import os
import random
from typing import Literal

import torch
from torch.utils.data import Dataset

from data.kinship import (
    RELATION_TO_IDX,
    RELATIONS,
    VOCAB_SIZE,
    tokenise,
    kinship_constraint_loss,
    check_kinship_constraint,
)

# Mapping from CLUTRR relation labels to our 8-relation vocabulary
_CLUTRR_TO_NST = {
    "father": "parent",
    "mother": "parent",
    "son": "child",
    "daughter": "child",
    "grandfather": "grandparent",
    "grandmother": "grandparent",
    "grandson": "grandchild",
    "granddaughter": "grandchild",
    "brother": "sibling",
    "sister": "sibling",
    "uncle": "ancestor",
    "aunt": "ancestor",
    "nephew": "descendant",
    "niece": "descendant",
    "husband": "self",
    "wife": "self",
    "father-in-law": "ancestor",
    "mother-in-law": "ancestor",
    "son-in-law": "descendant",
    "daughter-in-law": "descendant",
}


class CLUTRRDataset(Dataset):
    """CLUTRR dataset wrapper for NST kinship experiments.

    Loads CLUTRR from HuggingFace and maps it to our relation vocabulary.
    Falls back to a small synthetic sample if HuggingFace is unavailable.

    Args:
        split: "train" or "test".
        max_chain_len: maximum chain length to include (for split control).
        min_chain_len: minimum chain length to include.
        max_seq_len: maximum token sequence length.
        max_samples: cap on number of samples (for compute budget).
        seed: random seed for reproducibility.
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        min_chain_len: int = 2,
        max_chain_len: int = 10,
        max_seq_len: int = 512,
        max_samples: int = 5000,
        seed: int = 42,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.samples: list[dict] = []

        rng = random.Random(seed)

        try:
            self._load_from_hf(split, min_chain_len, max_chain_len, max_samples, rng)
        except Exception as e:
            print(f"[CLUTRRDataset] HuggingFace load failed ({e}), using synthetic fallback")
            self._load_synthetic_fallback(split, min_chain_len, max_chain_len, max_samples, rng)

    def _load_from_hf(
        self, split: str, min_cl: int, max_cl: int, max_n: int, rng: random.Random
    ):
        """Load from HuggingFace datasets."""
        from datasets import load_dataset

        # CLUTRR/v1 has several config splits; use the standard one
        ds = load_dataset("CLUTRR/v1", "gen_train23_test2to10", trust_remote_code=True)
        hf_split = "train" if split == "train" else "test"

        if hf_split not in ds:
            # Some configs only have certain splits
            hf_split = list(ds.keys())[0]

        raw = ds[hf_split]

        for item in raw:
            story = item.get("story", item.get("clean_story", ""))
            target = item.get("target", item.get("target_text", ""))
            chain_len = item.get("story_length", item.get("num_edges", 2))

            if not (min_cl <= chain_len <= max_cl):
                continue

            # Map CLUTRR relation to our vocabulary
            target_lower = target.strip().lower().replace("-", "-")
            nst_rel = _CLUTRR_TO_NST.get(target_lower, None)
            if nst_rel is None:
                continue

            query = item.get("query", "")
            text = f"{story} {query}" if query else story

            self.samples.append({
                "text": text,
                "answer": nst_rel,
                "answer_idx": RELATION_TO_IDX[nst_rel],
                "chain_length": chain_len,
                "original_target": target,
            })

            if len(self.samples) >= max_n:
                break

        rng.shuffle(self.samples)

    def _load_synthetic_fallback(
        self, split: str, min_cl: int, max_cl: int, max_n: int, rng: random.Random
    ):
        """Generate synthetic CLUTRR-style samples as fallback."""
        from data.kinship import generate_sample

        if split == "train":
            depths = list(range(min_cl, min(max_cl, 4)))
        else:
            depths = list(range(max(3, min_cl), max_cl + 1))

        for _ in range(max_n):
            depth = rng.choice(depths)
            sample = generate_sample(depth, rng, direction_mix=True, n_distractors=1)
            self.samples.append({
                "text": sample.text,
                "answer": sample.answer,
                "answer_idx": sample.answer_idx,
                "chain_length": sample.chain_length,
                "original_target": sample.answer,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "input_ids": tokenise(s["text"], self.max_seq_len),
            "label": torch.tensor(s["answer_idx"], dtype=torch.long),
            "chain_length": s["chain_length"],
            "text": s["text"],
            "answer": s["answer"],
        }


def clutrr_collate_fn(batch: list[dict]) -> dict:
    """Collate function for CLUTRRDataset."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "chain_lengths": [b["chain_length"] for b in batch],
        "texts": [b["text"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }


def print_clutrr_stats(split: str = "train", max_samples: int = 5000, seed: int = 42):
    """Print CLUTRR dataset statistics."""
    ds = CLUTRRDataset(split=split, max_samples=max_samples, seed=seed)
    print(f"=== CLUTRR Dataset ({split}) ===")
    print(f"  Samples: {len(ds)}")

    from collections import Counter
    rel_counts = Counter(s["answer"] for s in ds.samples)
    cl_counts = Counter(s["chain_length"] for s in ds.samples)

    print(f"  Relation distribution:")
    for rel in RELATIONS:
        print(f"    {rel:<16} {rel_counts.get(rel, 0):>5}")

    print(f"  Chain length distribution:")
    for cl in sorted(cl_counts):
        print(f"    depth {cl}: {cl_counts[cl]:>5}")
