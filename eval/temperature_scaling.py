"""Post-hoc temperature scaling for calibration.

Temperature scaling (Guo et al., 2017) learns a single scalar T > 0
such that softmax(logits / T) is better calibrated.

This is applied AFTER training, on a held-out calibration set (dev).
It does NOT retrain the model — only learns T.

Usage:
    from eval.temperature_scaling import learn_temperature, apply_temperature
    T = learn_temperature(model, dev_loader, device)
    calibrated_probs = apply_temperature(logits, T)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Learns optimal temperature for post-hoc calibration."""

    def __init__(self):
        super().__init__()
        # Initialise T = 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature.clamp(min=0.01)


def learn_temperature(
    model,
    dataloader,
    device: str,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Learn optimal temperature on a calibration set.

    Args:
        model: trained model with forward(input_ids, attention_mask) → {logits}.
        dataloader: calibration dataloader (typically dev set).
        device: computation device.
        max_iter: LBFGS iterations.
        lr: learning rate.

    Returns:
        Optimal temperature T (float).
    """
    model.eval()

    # Collect all logits and labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            out = model(input_ids, attention_mask)
            all_logits.append(out["logits"].cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Optimise temperature
    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def eval_fn():
        optimizer.zero_grad()
        scaled_logits = scaler(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)

    optimal_T = scaler.temperature.item()
    print(f"  Learned temperature: T = {optimal_T:.4f}")

    return optimal_T


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: [B, C] raw logits.
        temperature: learned temperature T > 0.

    Returns:
        Calibrated probabilities [B, C].
    """
    T = max(temperature, 0.01)  # safety clamp
    return F.softmax(logits / T, dim=-1)
