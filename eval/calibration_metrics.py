"""Calibration and robustness metrics for neuro-symbolic evaluation.

Implements:
- Expected Calibration Error (ECE) with configurable bins.
- Brier Score (mean squared error of probability estimates).
- Reliability diagram data (for plotting).
- Noise robustness evaluation (accuracy under varying noise levels).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    where B_b is the set of samples whose max predicted probability falls
    in bin b, acc is the accuracy within the bin, and conf is the mean
    confidence within the bin.

    Args:
        probs: [N, C] predicted probabilities (post-softmax).
        labels: [N] ground-truth class indices.
        n_bins: number of confidence bins (default 15).

    Returns:
        (ece, bin_data): ECE value and list of per-bin statistics.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    n_total = len(labels)

    for i in range(n_bins):
        lo = bin_boundaries[i].item()
        hi = bin_boundaries[i + 1].item()

        # Include right boundary for last bin
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_in_bin = mask.sum().item()
        if n_in_bin == 0:
            bin_data.append({
                "bin_lo": lo, "bin_hi": hi,
                "count": 0, "accuracy": 0.0, "confidence": 0.0,
                "gap": 0.0,
            })
            continue

        bin_acc = accuracies[mask].mean().item()
        bin_conf = confidences[mask].mean().item()
        gap = abs(bin_acc - bin_conf)

        ece += (n_in_bin / n_total) * gap

        bin_data.append({
            "bin_lo": round(lo, 4),
            "bin_hi": round(hi, 4),
            "count": int(n_in_bin),
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
            "gap": round(gap, 4),
        })

    return round(ece, 6), bin_data


def brier_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute Brier Score (multi-class).

    BS = (1/N) Σ_i Σ_c (p_{i,c} − y_{i,c})²

    where y_{i,c} is the one-hot encoding of the true label.
    Lower is better. Perfect = 0.

    Args:
        probs: [N, C] predicted probabilities.
        labels: [N] ground-truth class indices.

    Returns:
        Brier score (float).
    """
    N, C = probs.shape
    one_hot = F.one_hot(labels, num_classes=C).float()
    bs = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()
    return round(bs, 6)


def reliability_diagram_data(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """Generate data for a reliability diagram plot.

    Returns a dict with bin midpoints, accuracies, confidences, and counts,
    suitable for matplotlib bar/line charts.

    Args:
        probs: [N, C] predicted probabilities.
        labels: [N] ground-truth class indices.
        n_bins: number of bins.

    Returns:
        Dict with keys: midpoints, accuracies, confidences, counts, ece.
    """
    ece, bin_data = expected_calibration_error(probs, labels, n_bins)

    midpoints = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bin_data]
    accuracies = [b["accuracy"] for b in bin_data]
    confidences = [b["confidence"] for b in bin_data]
    counts = [b["count"] for b in bin_data]

    return {
        "midpoints": midpoints,
        "accuracies": accuracies,
        "confidences": confidences,
        "counts": counts,
        "ece": ece,
        "n_bins": n_bins,
    }


def noise_robustness_eval(
    model,
    dataloader,
    device: str,
    noise_levels: list[float] | None = None,
    predict_fn=None,
) -> list[dict]:
    """Evaluate model accuracy under varying levels of input noise.

    Adds Gaussian noise to inputs and measures degradation in accuracy.

    Args:
        model: PyTorch model with a forward method.
        dataloader: DataLoader with img_a, img_b, digit_a, digit_b, sum keys.
        device: computation device.
        noise_levels: list of σ values for Gaussian noise.
        predict_fn: optional custom prediction function.

    Returns:
        List of dicts with noise_level, digit_acc, sum_acc, csr.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    results = []
    model.eval()

    for sigma in noise_levels:
        correct_a = correct_b = correct_sum = total = 0
        csr_total = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                img_a = batch["img_a"].to(device)
                img_b = batch["img_b"].to(device)
                digit_a = batch["digit_a"].to(device)
                digit_b = batch["digit_b"].to(device)
                sum_target = batch["sum"].to(device)

                # Add noise
                if sigma > 0:
                    img_a = img_a + sigma * torch.randn_like(img_a)
                    img_b = img_b + sigma * torch.randn_like(img_b)
                    img_a = img_a.clamp(0, 1)
                    img_b = img_b.clamp(0, 1)

                if predict_fn:
                    preds = predict_fn(model, img_a, img_b)
                else:
                    preds = model.predict(img_a, img_b)

                correct_a += (preds["pred_a"] == digit_a).sum().item()
                correct_b += (preds["pred_b"] == digit_b).sum().item()
                correct_sum += (preds["pred_sum"] == sum_target).sum().item()
                total += digit_a.size(0)
                csr_total += preds["csr"]
                n_batches += 1

        results.append({
            "noise_level": sigma,
            "digit_acc": round((correct_a + correct_b) / max(1, 2 * total), 4),
            "sum_acc": round(correct_sum / max(1, total), 4),
            "csr": round(csr_total / max(1, n_batches), 4),
            "n_samples": total,
        })

    return results


def collect_calibration_data(
    model,
    dataloader,
    device: str,
    task: str = "digit_add",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect all prediction probabilities and labels from a dataloader.

    Returns:
        (all_probs, all_labels): concatenated tensors.
    """
    all_probs = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if task == "digit_add":
                img_a = batch["img_a"].to(device)
                img_b = batch["img_b"].to(device)
                result = model(img_a, img_b)
                all_probs.append(result["probs_sum"].cpu())
                all_labels.append(batch["sum"])
            elif task == "kinship":
                input_ids = batch["input_ids"].to(device)
                result = model(input_ids)
                all_probs.append(result["probs"].cpu())
                all_labels.append(batch["label"])

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)
