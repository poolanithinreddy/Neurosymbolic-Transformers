"""Multi-digit addition constraints and verification.

Soft differentiable constraints for 2-digit + 2-digit addition with carry:
    ones_a + ones_b = ones_result + 10 * carry_ones
    tens_a + tens_b + carry_ones = tens_result + 10 * carry_tens
    hundreds_result = carry_tens

Verification: checks whether model predictions satisfy arithmetic constraints,
returning counterexamples for CEGIS training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def carry_constraint_soft(
    p_a_ones: torch.Tensor,   # [B, 10]
    p_a_tens: torch.Tensor,   # [B, 10]
    p_b_ones: torch.Tensor,   # [B, 10]
    p_b_tens: torch.Tensor,   # [B, 10]
    p_s_ones: torch.Tensor,   # [B, 10]
    p_s_tens: torch.Tensor,   # [B, 10]
    p_s_hund: torch.Tensor,   # [B, 10] (only 0 or 1 used)
) -> tuple[torch.Tensor, dict]:
    """Differentiable carry-propagation constraint for multi-digit addition.

    Computes the expected ones-column sum and carry, then the tens-column
    sum and carry, and penalises deviation from the predicted distribution.

    Returns:
        loss: scalar constraint violation loss.
        info: dict with intermediate expected distributions.
    """
    B = p_a_ones.size(0)
    device = p_a_ones.device

    # --- Ones column: a_ones + b_ones = s_ones + 10 * carry_ones ---
    # Expected distribution over (a_ones + b_ones) via discrete convolution
    ones_sum_dist = torch.zeros(B, 19, device=device)  # 0..18
    for k in range(19):
        i_min, i_max = max(0, k - 9), min(9, k)
        for i in range(i_min, i_max + 1):
            j = k - i
            ones_sum_dist[:, k] += p_a_ones[:, i] * p_b_ones[:, j]

    # Expected P(carry_ones = 1) = P(ones_sum ≥ 10)
    p_carry_ones = ones_sum_dist[:, 10:].sum(dim=-1)  # [B]

    # Expected ones digit result: modulo 10
    p_ones_result_expected = torch.zeros(B, 10, device=device)
    for d in range(10):
        p_ones_result_expected[:, d] = ones_sum_dist[:, d]
        if d + 10 < 19:
            p_ones_result_expected[:, d] = p_ones_result_expected[:, d] + ones_sum_dist[:, d + 10]

    p_ones_result_expected = p_ones_result_expected / p_ones_result_expected.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-8)

    # KL loss for ones digit
    loss_ones = F.kl_div(
        torch.log(p_s_ones.clamp(min=1e-8)),
        p_ones_result_expected,
        reduction="batchmean",
    )

    # --- Tens column: a_tens + b_tens + carry_ones = s_tens + 10 * carry_tens ---
    # We need to marginalise over carry_ones
    tens_sum_dist = torch.zeros(B, 20, device=device)  # 0..19
    for k in range(20):
        for carry in (0, 1):
            p_c = p_carry_ones if carry == 1 else (1 - p_carry_ones)  # [B]
            remaining = k - carry
            if remaining < 0 or remaining > 18:
                continue
            i_min, i_max = max(0, remaining - 9), min(9, remaining)
            for i in range(i_min, i_max + 1):
                j = remaining - i
                tens_sum_dist[:, k] += p_a_tens[:, i] * p_b_tens[:, j] * p_c

    # Expected tens result: modulo 10
    p_tens_result_expected = torch.zeros(B, 10, device=device)
    for d in range(10):
        if d < 20:
            p_tens_result_expected[:, d] += tens_sum_dist[:, d]
        if d + 10 < 20:
            p_tens_result_expected[:, d] += tens_sum_dist[:, d + 10]

    p_tens_result_expected = p_tens_result_expected / p_tens_result_expected.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-8)

    loss_tens = F.kl_div(
        torch.log(p_s_tens.clamp(min=1e-8)),
        p_tens_result_expected,
        reduction="batchmean",
    )

    # --- Hundreds: carry_tens ---
    p_carry_tens = tens_sum_dist[:, 10:].sum(dim=-1)  # [B]
    p_hund_expected = torch.zeros(B, 10, device=device)
    p_hund_expected[:, 0] = 1 - p_carry_tens
    p_hund_expected[:, 1] = p_carry_tens

    loss_hund = F.kl_div(
        torch.log(p_s_hund.clamp(min=1e-8)),
        p_hund_expected,
        reduction="batchmean",
    )

    total_loss = loss_ones + loss_tens + loss_hund

    info = {
        "loss_ones": loss_ones.item(),
        "loss_tens": loss_tens.item(),
        "loss_hund": loss_hund.item(),
        "p_carry_ones_mean": p_carry_ones.mean().item(),
        "p_carry_tens_mean": p_carry_tens.mean().item(),
    }

    return total_loss, info


def verify_multi_digit(
    pred_a_tens: torch.Tensor,
    pred_a_ones: torch.Tensor,
    pred_b_tens: torch.Tensor,
    pred_b_ones: torch.Tensor,
    pred_s_ones: torch.Tensor,
    pred_s_tens: torch.Tensor,
    pred_s_hund: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Verify arithmetic constraints on predictions (hard check).

    Returns:
        violations: [B] boolean tensor (True = violated).
        csr: constraint satisfaction rate.
    """
    a = pred_a_tens * 10 + pred_a_ones
    b = pred_b_tens * 10 + pred_b_ones
    s_pred = pred_s_hund * 100 + pred_s_tens * 10 + pred_s_ones
    s_true = a + b

    violations = s_pred != s_true
    csr = 1.0 - violations.float().mean().item()
    return violations, csr


def find_counterexamples(
    model,
    dataloader,
    device: str,
    max_ce: int = 500,
) -> list[dict]:
    """Run the model on a dataset and find counterexamples (constraint violations).

    Args:
        model: MultiDigitModel with predict() method.
        dataloader: DataLoader yielding multi-digit batches.
        device: computation device.
        max_ce: maximum counterexamples to collect.

    Returns:
        List of counterexample dicts (raw batch items where model is wrong).
    """
    model.eval()
    counterexamples = []

    with torch.no_grad():
        for batch in dataloader:
            if len(counterexamples) >= max_ce:
                break

            preds = model.predict(
                batch["img_a"].to(device),
                batch["img_b"].to(device),
            )

            violations, _ = verify_multi_digit(
                preds["pred_a_tens"], preds["pred_a_ones"],
                preds["pred_b_tens"], preds["pred_b_ones"],
                preds["pred_s_ones"], preds["pred_s_tens"],
                preds["pred_s_hund"],
            )

            # Collect violating samples
            viol_indices = violations.nonzero(as_tuple=True)[0]
            for idx in viol_indices:
                if len(counterexamples) >= max_ce:
                    break
                i = idx.item()
                ce = {k: v[i] for k, v in batch.items()}
                counterexamples.append(ce)

    model.train()
    return counterexamples
