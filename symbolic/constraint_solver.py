"""Constraint solver: soft differentiable constraints and optional Z3 hard verification.

Soft constraints use product t-norm semantics from logic.logic.
Hard constraints use z3-solver (optional dependency) for SAT-based verification/repair.
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Soft differentiable constraint: digit addition  a + b = c
# ---------------------------------------------------------------------------


def sum_constraint_soft(
    p_a: torch.Tensor,
    p_b: torch.Tensor,
    p_c: torch.Tensor,
    max_val: int = 19,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the soft arithmetic constraint loss for digit addition.

    Given predicted probability distributions over digit classes for a, b,
    and the sum c, compute the expected sum distribution via discrete
    convolution and return the KL-divergence loss plus the expected sum
    distribution.

    Args:
        p_a: [B, 10] softmax probabilities for digit a (0-9).
        p_b: [B, 10] softmax probabilities for digit b (0-9).
        p_c: [B, num_sum_classes] predicted probabilities for sum c.
        max_val: maximum possible sum value (inclusive), default 19 (9+9).

    Returns:
        loss: scalar KL divergence loss between expected and predicted sum.
        p_c_expected: [B, num_sum_classes] expected sum distribution.
    """
    B = p_a.size(0)
    num_sum = max_val + 1  # 0..max_val

    # Discrete convolution: P(c=k) = sum_{i+j=k} P(a=i) * P(b=j)
    # Efficient via outer product + diagonal sums
    p_c_expected = torch.zeros(B, num_sum, device=p_a.device, dtype=p_a.dtype)
    for k in range(num_sum):
        # all (i, j) with i + j = k, 0 <= i <= 9, 0 <= j <= 9
        i_min = max(0, k - 9)
        i_max = min(9, k)
        for i in range(i_min, i_max + 1):
            j = k - i
            p_c_expected[:, k] += p_a[:, i] * p_b[:, j]

    # Clamp for numerical stability
    p_c_expected = p_c_expected.clamp(min=1e-8)
    p_c_expected = p_c_expected / p_c_expected.sum(dim=-1, keepdim=True)

    # If p_c has fewer classes (e.g., 10 for single-digit), truncate expected
    if p_c.size(1) < num_sum:
        p_c_expected = p_c_expected[:, : p_c.size(1)]
        p_c_expected = p_c_expected / p_c_expected.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # KL(expected || predicted) — we want predicted to match expected
    log_p_c = torch.log(p_c.clamp(min=1e-8))
    loss = F.kl_div(log_p_c, p_c_expected, reduction="batchmean")

    return loss, p_c_expected


def sum_constraint_violation(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    pred_c: torch.Tensor,
) -> torch.Tensor:
    """Binary violation check: does argmax(a) + argmax(b) == argmax(c)?

    Returns a boolean tensor of shape [B] where True = violated.
    """
    a = pred_a.argmax(dim=-1)
    b = pred_b.argmax(dim=-1)
    c = pred_c.argmax(dim=-1)
    return (a + b) != c


def constraint_satisfaction_rate(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    pred_c: torch.Tensor,
) -> float:
    """Fraction of samples where argmax(a) + argmax(b) == argmax(c)."""
    violations = sum_constraint_violation(pred_a, pred_b, pred_c)
    return 1.0 - violations.float().mean().item()


# ---------------------------------------------------------------------------
# Generalised soft constraint: arbitrary body => head with t-norm
# ---------------------------------------------------------------------------


def soft_rule_loss(
    body_values: list[torch.Tensor],
    head_value: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Differentiable Horn clause violation loss.

    Uses product t-norm for conjunction and Reichenbach implication.

    Args:
        body_values: list of [B] tensors in [0,1] — soft truth of body atoms.
        head_value: [B] tensor in [0,1] — soft truth of head atom.
        weight: rule weight multiplier.

    Returns:
        Scalar mean violation loss.
    """
    if len(body_values) == 0:
        return torch.tensor(0.0, device=head_value.device)

    # Product t-norm for body conjunction
    body = body_values[0].clamp(0, 1)
    for bv in body_values[1:]:
        body = body * bv.clamp(0, 1)

    # Violation = body * (1 - head) — zero when head is true or body is false
    violation = body * (1.0 - head_value.clamp(0, 1))
    return weight * violation.mean()


# ---------------------------------------------------------------------------
# Hard constraint verification via Z3 (optional)
# ---------------------------------------------------------------------------

_z3_available = None


def _check_z3():
    global _z3_available
    if _z3_available is None:
        try:
            import z3  # noqa: F401
            _z3_available = True
        except ImportError:
            _z3_available = False
    return _z3_available


def hard_constraint_verify(
    a_pred: int, b_pred: int, c_pred: int
) -> tuple[bool, tuple[int, int, int]]:
    """Verify a + b == c using Z3 and repair if violated.

    If z3 is not installed, falls back to Python arithmetic check + repair.

    Args:
        a_pred: predicted digit a.
        b_pred: predicted digit b.
        c_pred: predicted sum c.

    Returns:
        (satisfied, (a, b, c_repaired)):
            satisfied: True if original prediction is correct.
            c_repaired: the corrected sum (= a + b) if violated.
    """
    correct_sum = a_pred + b_pred
    if c_pred == correct_sum:
        return True, (a_pred, b_pred, c_pred)

    if _check_z3():
        import z3

        a = z3.Int("a")
        b = z3.Int("b")
        c = z3.Int("c")

        solver = z3.Solver()
        solver.add(a == a_pred)
        solver.add(b == b_pred)
        solver.add(c == a + b)
        solver.add(a >= 0, a <= 9)
        solver.add(b >= 0, b <= 9)
        solver.add(c >= 0, c <= 18)

        if solver.check() == z3.sat:
            model = solver.model()
            c_repaired = model[c].as_long()
            return False, (a_pred, b_pred, c_repaired)
        else:
            # Should not happen for valid digit inputs
            return False, (a_pred, b_pred, correct_sum)
    else:
        # Fallback: arithmetic repair without Z3
        return False, (a_pred, b_pred, correct_sum)


def hard_constraint_batch(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    pred_c: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Apply hard constraint verification to a batch.

    Args:
        pred_a: [B, 10] logits/probs for digit a.
        pred_b: [B, 10] logits/probs for digit b.
        pred_c: [B, num_sum] logits/probs for sum c.

    Returns:
        c_repaired: [B] tensor of corrected sum predictions.
        repair_rate: fraction of samples that needed repair.
    """
    a_vals = pred_a.argmax(dim=-1)
    b_vals = pred_b.argmax(dim=-1)
    c_vals = pred_c.argmax(dim=-1)

    c_repaired = []
    repairs = 0
    for i in range(len(a_vals)):
        satisfied, (_, _, c_rep) = hard_constraint_verify(
            a_vals[i].item(), b_vals[i].item(), c_vals[i].item()
        )
        c_repaired.append(c_rep)
        if not satisfied:
            repairs += 1

    return (
        torch.tensor(c_repaired, device=pred_a.device),
        repairs / max(1, len(a_vals)),
    )
