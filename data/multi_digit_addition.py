"""Multi-digit addition dataset: 2-digit + 2-digit → up to 3-digit result.

Compositional split based on carry propagation:
  - Train: pairs where NO carry occurs (ones-digit sum ≤ 9 AND tens-digit sum ≤ 9).
  - IID test: same distribution as train.
  - Comp test: pairs requiring at least one carry.
  - Hard test: pairs requiring TWO carries (both ones and tens overflow).

Each number is rendered as a [1, 28, 56] image (two 28×28 digit images
concatenated horizontally). Optional distractor digit overlay.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.digit_addition import render_digit  # reuse single-digit renderer


# ---------------------------------------------------------------------------
# Carry utilities
# ---------------------------------------------------------------------------

def has_carry(a: int, b: int) -> tuple[bool, bool]:
    """Check which digit positions produce a carry.

    Args:
        a, b: two-digit numbers (10–99).

    Returns:
        (ones_carry, tens_carry): True if that position overflows.
    """
    ones_a, tens_a = a % 10, a // 10
    ones_b, tens_b = b % 10, b // 10

    ones_carry = (ones_a + ones_b) >= 10
    carry_in = 1 if ones_carry else 0
    tens_carry = (tens_a + tens_b + carry_in) >= 10

    return ones_carry, tens_carry


def count_carries(a: int, b: int) -> int:
    """Count how many carries occur in a + b."""
    oc, tc = has_carry(a, b)
    return int(oc) + int(tc)


# ---------------------------------------------------------------------------
# Number image rendering
# ---------------------------------------------------------------------------

def render_number(
    number: int,
    img_size: int = 28,
    noise_std: float = 0.15,
    rng: random.Random | None = None,
    add_distractor: bool = False,
) -> np.ndarray:
    """Render a 2-digit number as a [img_size, 2*img_size] image.

    The tens-digit is on the left, ones-digit on the right.

    Args:
        number: integer 10–99.
        img_size: height/width of each digit cell.
        noise_std: Gaussian noise standard deviation.
        rng: random number generator.
        add_distractor: if True, add a faint random digit in the corner.

    Returns:
        [img_size, 2*img_size] numpy array in [0, 1].
    """
    if rng is None:
        rng = random.Random()

    tens = number // 10
    ones = number % 10

    img_tens = render_digit(tens, size=img_size, noise_std=noise_std, rng=rng)
    img_ones = render_digit(ones, size=img_size, noise_std=noise_std, rng=rng)

    # Concatenate horizontally
    img = np.concatenate([img_tens, img_ones], axis=1)  # [28, 56]

    if add_distractor:
        # Render a faint distractor digit in a random corner
        d = rng.randint(0, 9)
        distractor = render_digit(d, size=img_size // 2, noise_std=0.3, rng=rng)
        # Place in top-right corner at 30% opacity
        corner_h = img_size // 2
        corner_w = img_size // 2
        y0 = 0
        x0 = 2 * img_size - corner_w
        img[y0:y0 + corner_h, x0:x0 + corner_w] = (
            0.7 * img[y0:y0 + corner_h, x0:x0 + corner_w] + 0.3 * distractor
        )
        img = np.clip(img, 0.0, 1.0)

    return img


def decompose_sum(s: int) -> tuple[int, int, int]:
    """Decompose a sum (0–198) into (hundreds, tens, ones) digits."""
    hundreds = s // 100
    tens = (s % 100) // 10
    ones = s % 10
    return hundreds, tens, ones


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiDigitAdditionDataset(Dataset):
    """Multi-digit addition dataset (2-digit + 2-digit).

    Splits based on carry propagation:
      - "train": no carries (both ones_sum ≤ 9 and tens_sum ≤ 9)
      - "iid_test": same distribution as train
      - "comp_test": at least one carry
      - "hard_test": exactly two carries (both positions overflow)
    """

    def __init__(
        self,
        split: Literal["train", "iid_test", "comp_test", "hard_test"] = "train",
        n_samples: int = 10000,
        img_size: int = 28,
        noise_std: float = 0.15,
        distractor_rate: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.noise_std = noise_std
        self.distractor_rate = distractor_rate
        self.rng = random.Random(seed)

        # Generate all valid 2-digit pairs (10–99 × 10–99)
        all_pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]

        if split in ("train", "iid_test"):
            # No-carry pairs: ones sum ≤ 9 AND tens sum ≤ 9
            pairs = [(a, b) for a, b in all_pairs if count_carries(a, b) == 0]
        elif split == "comp_test":
            # At least one carry
            pairs = [(a, b) for a, b in all_pairs if count_carries(a, b) >= 1]
        elif split == "hard_test":
            # Both carries
            pairs = [(a, b) for a, b in all_pairs if count_carries(a, b) == 2]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples = []
        for _ in range(n_samples):
            a, b = self.rng.choice(pairs)
            s = a + b
            ones_carry, tens_carry = has_carry(a, b)
            self.samples.append({
                "a": a,
                "b": b,
                "sum": s,
                "a_tens": a // 10,
                "a_ones": a % 10,
                "b_tens": b // 10,
                "b_ones": b % 10,
                "sum_hundreds": s // 100,
                "sum_tens": (s % 100) // 10,
                "sum_ones": s % 10,
                "ones_carry": int(ones_carry),
                "tens_carry": int(tens_carry),
                "n_carries": int(ones_carry) + int(tens_carry),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        add_dist = self.rng.random() < self.distractor_rate

        img_a = render_number(
            s["a"], self.img_size, self.noise_std, self.rng, add_distractor=add_dist
        )
        img_b = render_number(
            s["b"], self.img_size, self.noise_std, self.rng, add_distractor=add_dist
        )

        return {
            "img_a": torch.tensor(img_a, dtype=torch.float32).unsqueeze(0),  # [1, 28, 56]
            "img_b": torch.tensor(img_b, dtype=torch.float32).unsqueeze(0),
            "a_tens": torch.tensor(s["a_tens"], dtype=torch.long),
            "a_ones": torch.tensor(s["a_ones"], dtype=torch.long),
            "b_tens": torch.tensor(s["b_tens"], dtype=torch.long),
            "b_ones": torch.tensor(s["b_ones"], dtype=torch.long),
            "sum_hundreds": torch.tensor(s["sum_hundreds"], dtype=torch.long),
            "sum_tens": torch.tensor(s["sum_tens"], dtype=torch.long),
            "sum_ones": torch.tensor(s["sum_ones"], dtype=torch.long),
            "ones_carry": torch.tensor(s["ones_carry"], dtype=torch.long),
            "tens_carry": torch.tensor(s["tens_carry"], dtype=torch.long),
            "full_sum": torch.tensor(s["sum"], dtype=torch.long),
        }


def multi_digit_collate(batch: list[dict]) -> dict:
    """Collate function for MultiDigitAdditionDataset."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def generate_stats(seed: int = 42, n_train: int = 10000, n_test: int = 2000):
    """Print dataset statistics."""
    splits = {
        "Train (no carry)": ("train", n_train),
        "IID Test": ("iid_test", n_test),
        "Comp Test (≥1 carry)": ("comp_test", n_test),
        "Hard Test (2 carries)": ("hard_test", min(n_test, 1000)),
    }

    print("=== Multi-Digit Addition Dataset Statistics ===")
    for name, (split, n) in splits.items():
        ds = MultiDigitAdditionDataset(split=split, n_samples=n, seed=seed)
        carries = [s["n_carries"] for s in ds.samples]
        print(f"\n  {name}: {len(ds)} samples")
        print(f"    Carry distribution: 0={carries.count(0)}, 1={carries.count(1)}, 2={carries.count(2)}")
        sums = [s["sum"] for s in ds.samples]
        print(f"    Sum range: {min(sums)}–{max(sums)}")

    # Count available pairs per split
    all_pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    no_carry = sum(1 for a, b in all_pairs if count_carries(a, b) == 0)
    one_carry = sum(1 for a, b in all_pairs if count_carries(a, b) == 1)
    two_carry = sum(1 for a, b in all_pairs if count_carries(a, b) == 2)
    print(f"\n  Pair pool sizes: no_carry={no_carry}, 1_carry={one_carry}, 2_carry={two_carry}")
