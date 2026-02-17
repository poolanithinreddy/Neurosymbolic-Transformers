"""Neural perception module: CNN encoder for digit recognition.

A small, efficient CNN that maps 28×28 grayscale digit images to class
probability distributions.  Used as the perception backbone in the
digit-addition neuro-symbolic experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Small CNN for single-digit recognition (0-9).

    Architecture:
        Conv(1→16, 3×3) → ReLU → MaxPool(2)
        Conv(16→32, 3×3) → ReLU → MaxPool(2)
        Conv(32→64, 3×3) → ReLU → AdaptiveAvgPool(1)
        FC(64 → 10)

    Input:  [B, 1, 28, 28]
    Output: [B, 10] (raw logits — apply softmax externally)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 1, 28, 28] grayscale images.

        Returns:
            logits: [B, 10] class logits.
        """
        x = self.pool(F.relu(self.conv1(x)))     # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))     # [B, 32, 7, 7]
        x = self.adaptive_pool(F.relu(self.conv3(x)))  # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)                # [B, 64]
        return self.fc(x)                         # [B, 10]

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x), dim=-1)

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class predictions."""
        return self.forward(x).argmax(dim=-1)


class SumHead(nn.Module):
    """MLP head that predicts sum class (0-18) from concatenated digit features.

    Used in the pure-neural baseline (no symbolic constraint).

    Input:  [B, 128] concatenated features from two DigitCNN encoders.
    Output: [B, 19] logits over sum classes.
    """

    def __init__(self, in_dim: int = 128, num_classes: int = 19):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class DigitEncoder(nn.Module):
    """Feature extractor variant of DigitCNN that returns embeddings.

    Same architecture but stops before the classification head.

    Input:  [B, 1, 28, 28]
    Output: [B, 64] feature embeddings.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(F.relu(self.conv3(x)))
        return x.view(x.size(0), -1)  # [B, 64]
