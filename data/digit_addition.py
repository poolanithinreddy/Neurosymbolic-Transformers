"""Synthetic digit-addition dataset generator.

Generates pairs of digit images (from a procedural renderer or MNIST-style)
with labels (digit_a, digit_b, sum) and controlled IID / compositional splits.

This generator uses procedural font rendering (no external dataset downloads)
so the experiment is fully self-contained and reproducible.
"""

import os
import random
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Procedural digit image renderer (no PIL/external dependency)
# ---------------------------------------------------------------------------

# 5x3 bitmap font for digits 0-9
_DIGIT_BITMAPS = {
    0: [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    1: [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
    ],
    2: [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
    ],
    3: [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
    ],
    4: [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ],
    5: [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
    ],
    6: [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    7: [
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ],
    8: [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    9: [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
    ],
}


def render_digit(
    digit: int,
    size: int = 28,
    noise_std: float = 0.15,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Render a single digit as a size×size grayscale image.

    Uses a 5×3 bitmap upscaled to target size with Gaussian noise.

    Args:
        digit: integer 0-9.
        size: output image size (square).
        noise_std: standard deviation of additive Gaussian noise.
        rng: optional Random instance for reproducibility.

    Returns:
        numpy array of shape (size, size) with values in [0, 1].
    """
    if rng is None:
        rng = random.Random()

    bitmap = np.array(_DIGIT_BITMAPS[digit], dtype=np.float32)  # 5x3
    # Upscale via nearest-neighbor to target size
    # Place the 5x3 bitmap in the center with some padding
    img = np.zeros((size, size), dtype=np.float32)
    # Scale factors
    bh, bw = bitmap.shape
    margin = max(2, size // 7)
    cell_h = (size - 2 * margin) / bh
    cell_w = (size - 2 * margin) / bw

    for r in range(bh):
        for c in range(bw):
            if bitmap[r, c] > 0.5:
                y0 = int(margin + r * cell_h)
                y1 = int(margin + (r + 1) * cell_h)
                x0 = int(margin + c * cell_w)
                x1 = int(margin + (c + 1) * cell_w)
                img[y0:y1, x0:x1] = 1.0

    # Add random position jitter (±2 pixels)
    jitter_y = rng.randint(-2, 2)
    jitter_x = rng.randint(-2, 2)
    img = np.roll(img, jitter_y, axis=0)
    img = np.roll(img, jitter_x, axis=1)

    # Add Gaussian noise
    noise = np.array(
        [[rng.gauss(0, noise_std) for _ in range(size)] for _ in range(size)],
        dtype=np.float32,
    )
    img = img + noise
    img = np.clip(img, 0.0, 1.0)

    return img


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class DigitAdditionDataset(Dataset):
    """Synthetic digit addition dataset.

    Each sample is (img_a, img_b, digit_a, digit_b, sum_value) where:
    - img_a, img_b: [1, 28, 28] tensors (grayscale digit images).
    - digit_a, digit_b: integer labels 0-9.
    - sum_value: digit_a + digit_b (0-18).

    Splits:
    - "iid": all 100 digit pairs (0-9 × 0-9) in both train and test.
    - "comp": train on pairs with sum ≤ threshold (default 9);
              test on pairs with sum > threshold (compositional).
    """

    def __init__(
        self,
        split: Literal["train", "iid_test", "comp_test"] = "train",
        n_samples: int = 10000,
        comp_threshold: int = 9,
        img_size: int = 28,
        noise_std: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        # Determine which digit pairs are allowed for this split
        all_pairs = [(a, b) for a in range(10) for b in range(10)]
        if split == "train":
            pairs = [(a, b) for a, b in all_pairs if a + b <= comp_threshold]
        elif split == "comp_test":
            pairs = [(a, b) for a, b in all_pairs if a + b > comp_threshold]
        else:  # iid_test — use all pairs
            pairs = all_pairs

        self.samples = []
        for _ in range(n_samples):
            a, b = self.rng.choice(pairs)
            self.samples.append((a, b, a + b))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        a, b, s = self.samples[idx]
        img_a = render_digit(a, size=self.img_size, noise_std=self.noise_std, rng=self.rng)
        img_b = render_digit(b, size=self.img_size, noise_std=self.noise_std, rng=self.rng)

        return {
            "img_a": torch.tensor(img_a, dtype=torch.float32).unsqueeze(0),  # [1, H, W]
            "img_b": torch.tensor(img_b, dtype=torch.float32).unsqueeze(0),
            "digit_a": torch.tensor(a, dtype=torch.long),
            "digit_b": torch.tensor(b, dtype=torch.long),
            "sum": torch.tensor(s, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# CLI for generating and inspecting data
# ---------------------------------------------------------------------------


def generate_stats(seed: int = 42, n_train: int = 5000, n_test: int = 1000, threshold: int = 9):
    """Print dataset statistics for verification."""
    train = DigitAdditionDataset("train", n_samples=n_train, comp_threshold=threshold, seed=seed)
    iid = DigitAdditionDataset("iid_test", n_samples=n_test, comp_threshold=threshold, seed=seed + 1)
    comp = DigitAdditionDataset("comp_test", n_samples=n_test, comp_threshold=threshold, seed=seed + 2)

    print(f"Train: {len(train)} samples")
    print(f"  Sum range: {min(s for _, _, s in train.samples)}-{max(s for _, _, s in train.samples)}")
    train_pairs = set((a, b) for a, b, _ in train.samples)
    print(f"  Unique pairs: {len(train_pairs)}")

    print(f"IID Test: {len(iid)} samples")
    iid_pairs = set((a, b) for a, b, _ in iid.samples)
    print(f"  Unique pairs: {len(iid_pairs)}")

    print(f"Compositional Test: {len(comp)} samples")
    print(f"  Sum range: {min(s for _, _, s in comp.samples)}-{max(s for _, _, s in comp.samples)}")
    comp_pairs = set((a, b) for a, b, _ in comp.samples)
    print(f"  Unique pairs: {len(comp_pairs)}")
    overlap = train_pairs & comp_pairs
    print(f"  Overlap with train pairs: {len(overlap)} (should be 0)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Digit addition dataset stats")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--threshold", type=int, default=9)
    args = ap.parse_args()
    generate_stats(args.seed, args.n_train, args.n_test, args.threshold)
