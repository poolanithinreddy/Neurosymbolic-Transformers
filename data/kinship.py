"""Synthetic kinship relational reasoning dataset (CLUTRR-inspired).

Generates family relationship chains where the model must infer an
unstated relationship from a sequence of stated relationships.

Example:
    Input:  "Alice is Bob's mother. Bob is Carol's parent."
    Query:  "What is Alice's relation to Carol?"
    Answer: "grandparent"

Compositional split: train on chains of length ≤ max_train_depth,
test on chains of length > max_train_depth. This measures whether
the model can generalise transitivity rules to longer chains.

Rules (Horn clauses):
    parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z)
    parent(X,Y) → ancestor(X,Y)
    grandparent(X,Y) → ancestor(X,Y)
    parent(X,Y) ∧ parent(X,Z) ∧ X≠Z → sibling(Y,Z)

All data is synthetic — no downloads required. Fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Relation vocabulary
# ---------------------------------------------------------------------------

RELATIONS = [
    "parent",
    "child",
    "grandparent",
    "grandchild",
    "sibling",
    "ancestor",
    "descendant",
    "self",
]

RELATION_TO_IDX = {r: i for i, r in enumerate(RELATIONS)}
IDX_TO_RELATION = {i: r for r, i in RELATION_TO_IDX.items()}
NUM_RELATIONS = len(RELATIONS)

# Inverse relations
INVERSE = {
    "parent": "child",
    "child": "parent",
    "grandparent": "grandchild",
    "grandchild": "grandparent",
    "sibling": "sibling",
    "ancestor": "descendant",
    "descendant": "ancestor",
    "self": "self",
}

# Names pool (enough for chains up to length 10)
_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Pete",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Vic", "Wendy", "Xander",
]


# ---------------------------------------------------------------------------
# Inference engine (ground truth)
# ---------------------------------------------------------------------------

def infer_relation(chain: list[str]) -> str:
    """Infer the end-to-end relation from a chain of base relations.

    Each element in chain is "parent" or "child". The chain represents
    the path from person A to person Z through intermediate people.

    Rules:
        - len 0: "self"
        - len 1: the relation itself
        - len 2 (parent, parent): "grandparent"
        - len 2 (child, child): "grandchild"
        - len 2 (parent, child): "sibling" (simplified)
        - len 3+ (all parent): "ancestor"
        - len 3+ (all child): "descendant"
        - mixed: "ancestor" if net direction is up, "descendant" if down
    """
    if len(chain) == 0:
        return "self"
    if len(chain) == 1:
        return chain[0]

    # Count direction: parent = +1 (up), child = -1 (down)
    up = sum(1 for r in chain if r == "parent")
    down = sum(1 for r in chain if r == "child")

    if up == len(chain):
        # All parent links
        if up == 2:
            return "grandparent"
        return "ancestor"
    elif down == len(chain):
        # All child links
        if down == 2:
            return "grandchild"
        return "descendant"
    elif up == 1 and down == 1:
        return "sibling"
    elif up > down:
        return "ancestor"
    elif down > up:
        return "descendant"
    else:
        return "sibling"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class KinshipSample:
    """A single kinship reasoning sample.

    Attributes:
        premises: list of (person_a, relation, person_b) triples.
        query: (person_a, person_b) — what relation?
        answer: relation string.
        chain_length: number of reasoning steps.
        text: natural language encoding of premises + query.
        answer_idx: integer index into RELATIONS.
    """
    premises: list[tuple[str, str, str]]
    query: tuple[str, str]
    answer: str
    chain_length: int
    text: str
    answer_idx: int


def generate_chain(
    depth: int,
    rng: random.Random,
    direction_mix: bool = True,
) -> tuple[list[str], list[tuple[str, str, str]], tuple[str, str]]:
    """Generate a random kinship reasoning chain.

    Args:
        depth: number of hops (reasoning steps).
        rng: random number generator.
        direction_mix: if True, allow mixed parent/child chains.

    Returns:
        (chain_relations, premises, query)
    """
    # Pick names (no repeats)
    names = rng.sample(_NAMES, k=depth + 1)

    # Decide direction for each hop
    chain_rels = []
    for _ in range(depth):
        if direction_mix:
            rel = rng.choice(["parent", "child"])
        else:
            rel = "parent"  # All parent → tests pure transitivity
        chain_rels.append(rel)

    # Build premises
    premises = []
    for i in range(depth):
        a = names[i]
        b = names[i + 1]
        rel = chain_rels[i]
        # "A is B's parent" means parent(A, B)
        premises.append((a, rel, b))

    query = (names[0], names[-1])
    return chain_rels, premises, query


def premises_to_text(premises: list[tuple[str, str, str]]) -> str:
    """Convert premises to natural language text."""
    parts = []
    for a, rel, b in premises:
        if rel == "parent":
            parts.append(f"{a} is {b}'s parent.")
        elif rel == "child":
            parts.append(f"{a} is {b}'s child.")
        else:
            parts.append(f"{a} is {b}'s {rel}.")
    return " ".join(parts)


def generate_sample(
    depth: int,
    rng: random.Random,
    direction_mix: bool = False,
) -> KinshipSample:
    """Generate a single kinship reasoning sample."""
    chain_rels, premises, query = generate_chain(
        depth, rng, direction_mix=direction_mix
    )
    answer = infer_relation(chain_rels)

    text = premises_to_text(premises) + f" What is {query[0]}'s relation to {query[1]}?"

    return KinshipSample(
        premises=premises,
        query=query,
        answer=answer,
        chain_length=depth,
        text=text,
        answer_idx=RELATION_TO_IDX[answer],
    )


# ---------------------------------------------------------------------------
# Tokeniser (character-level for simplicity + Colab friendliness)
# ---------------------------------------------------------------------------

# Build vocabulary from all possible characters in names + relation words
_VOCAB_CHARS = sorted(set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " .,'?!0123456789"
))
_CHAR_TO_IDX = {c: i + 2 for i, c in enumerate(_VOCAB_CHARS)}  # 0=PAD, 1=UNK
_CHAR_TO_IDX["<PAD>"] = 0
_CHAR_TO_IDX["<UNK>"] = 1
VOCAB_SIZE = len(_CHAR_TO_IDX)


def tokenise(text: str, max_len: int = 256) -> torch.Tensor:
    """Convert text to integer token IDs (character-level).

    Args:
        text: input string.
        max_len: maximum sequence length (pad/truncate).

    Returns:
        [max_len] integer tensor.
    """
    ids = [_CHAR_TO_IDX.get(c, 1) for c in text[:max_len]]
    # Pad
    while len(ids) < max_len:
        ids.append(0)
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class KinshipDataset(Dataset):
    """Synthetic kinship relational reasoning dataset.

    Splits:
    - "train": chains of length 1 to max_train_depth.
    - "iid_test": same depth distribution as train.
    - "comp_test": chains of length max_train_depth+1 to max_test_depth.
    """

    def __init__(
        self,
        split: Literal["train", "iid_test", "comp_test"] = "train",
        n_samples: int = 5000,
        max_train_depth: int = 3,
        max_test_depth: int = 5,
        max_seq_len: int = 256,
        direction_mix: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        rng = random.Random(seed)

        if split == "train":
            depths = list(range(1, max_train_depth + 1))
        elif split == "iid_test":
            depths = list(range(1, max_train_depth + 1))
        elif split == "comp_test":
            depths = list(range(max_train_depth + 1, max_test_depth + 1))
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples: list[KinshipSample] = []
        for _ in range(n_samples):
            depth = rng.choice(depths)
            sample = generate_sample(depth, rng, direction_mix=direction_mix)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "input_ids": tokenise(s.text, self.max_seq_len),
            "label": torch.tensor(s.answer_idx, dtype=torch.long),
            "chain_length": s.chain_length,
            "text": s.text,
            "answer": s.answer,
        }


def kinship_collate_fn(batch: list[dict]) -> dict:
    """Collate function for KinshipDataset."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "chain_lengths": [b["chain_length"] for b in batch],
        "texts": [b["text"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Rule definitions for kinship domain
# ---------------------------------------------------------------------------

KINSHIP_RULES = [
    {
        "id": "transitivity",
        "desc": "parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z)",
        "weight": 1.0,
    },
    {
        "id": "ancestor_base",
        "desc": "parent(X,Y) → ancestor(X,Y)",
        "weight": 0.5,
    },
    {
        "id": "ancestor_transitive",
        "desc": "grandparent(X,Y) → ancestor(X,Y)",
        "weight": 0.5,
    },
    {
        "id": "inverse_parent",
        "desc": "parent(X,Y) → child(Y,X)",
        "weight": 1.0,
    },
    {
        "id": "inverse_grandparent",
        "desc": "grandparent(X,Y) → grandchild(Y,X)",
        "weight": 1.0,
    },
]


def check_kinship_constraint(
    pred_probs: torch.Tensor,
    chain_lengths: list[int],
) -> tuple[float, float]:
    """Check if predictions satisfy kinship rules.

    Simple consistency check: for chains of length 2 with all parent links,
    the answer must be grandparent (index 2).

    Args:
        pred_probs: [B, NUM_RELATIONS] prediction probabilities.
        chain_lengths: list of chain lengths for each sample.

    Returns:
        (csr, violation_rate): constraint satisfaction rate and violation rate.
    """
    preds = pred_probs.argmax(dim=-1)
    total = 0
    satisfied = 0

    for i, cl in enumerate(chain_lengths):
        if cl == 1:
            # Single hop: must be parent or child (idx 0 or 1)
            if preds[i].item() in (0, 1):
                satisfied += 1
            total += 1
        elif cl == 2:
            # Two hops: must be grandparent, grandchild, or sibling (idx 2, 3, 4)
            if preds[i].item() in (2, 3, 4):
                satisfied += 1
            total += 1
        elif cl >= 3:
            # Three+ hops: must be ancestor, descendant, or sibling (idx 5, 6, 4)
            if preds[i].item() in (4, 5, 6):
                satisfied += 1
            total += 1

    csr = satisfied / max(1, total)
    return csr, 1.0 - csr


def kinship_constraint_loss(
    pred_probs: torch.Tensor,
    chain_lengths: list[int],
) -> torch.Tensor:
    """Differentiable constraint loss for kinship rules.

    For each sample, penalises probability mass on relations that are
    inconsistent with the chain length.

    Args:
        pred_probs: [B, NUM_RELATIONS] softmax probabilities.
        chain_lengths: list of chain lengths.

    Returns:
        Scalar loss.
    """
    B = pred_probs.size(0)
    losses = []

    for i in range(B):
        cl = chain_lengths[i]
        if cl == 1:
            # Valid: parent (0), child (1)
            valid_mask = torch.zeros(NUM_RELATIONS, device=pred_probs.device)
            valid_mask[0] = 1.0  # parent
            valid_mask[1] = 1.0  # child
        elif cl == 2:
            # Valid: grandparent (2), grandchild (3), sibling (4)
            valid_mask = torch.zeros(NUM_RELATIONS, device=pred_probs.device)
            valid_mask[2] = 1.0
            valid_mask[3] = 1.0
            valid_mask[4] = 1.0
        else:
            # Valid: ancestor (5), descendant (6), sibling (4)
            valid_mask = torch.zeros(NUM_RELATIONS, device=pred_probs.device)
            valid_mask[4] = 1.0
            valid_mask[5] = 1.0
            valid_mask[6] = 1.0

        # Penalise probability mass on invalid relations
        invalid_mass = (pred_probs[i] * (1.0 - valid_mask)).sum()
        losses.append(invalid_mass)

    return torch.stack(losses).mean()


def generate_stats(
    seed: int = 42,
    n_train: int = 5000,
    n_test: int = 1000,
    max_train_depth: int = 3,
    max_test_depth: int = 5,
) -> None:
    """Print dataset statistics for kinship benchmark."""
    train_ds = KinshipDataset("train", n_train, max_train_depth, max_test_depth, seed=seed)
    iid_ds = KinshipDataset("iid_test", n_test, max_train_depth, max_test_depth, seed=seed + 1)
    comp_ds = KinshipDataset("comp_test", n_test, max_train_depth, max_test_depth, seed=seed + 2)

    print("=== Kinship Dataset Statistics ===")
    print(f"  Train:     {len(train_ds)} samples (chains 1-{max_train_depth})")
    print(f"  IID Test:  {len(iid_ds)} samples (chains 1-{max_train_depth})")
    print(f"  Comp Test: {len(comp_ds)} samples (chains {max_train_depth+1}-{max_test_depth})")

    # Label distribution
    for name, ds in [("Train", train_ds), ("IID", iid_ds), ("Comp", comp_ds)]:
        from collections import Counter
        counts = Counter(s.answer for s in ds.samples)
        depth_counts = Counter(s.chain_length for s in ds.samples)
        print(f"\n  {name} label distribution:")
        for rel in RELATIONS:
            print(f"    {rel}: {counts.get(rel, 0)}")
        print(f"  {name} depth distribution:")
        for d in sorted(depth_counts):
            print(f"    depth {d}: {depth_counts[d]}")
