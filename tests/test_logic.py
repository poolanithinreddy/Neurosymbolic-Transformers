import torch
from logic.logic import horn_violation, imply


def test_implication_reichenbach():
    a = torch.tensor([0.0, 0.5, 1.0])
    b = torch.tensor([0.0, 0.5, 1.0])
    v = imply(a, b)
    assert torch.all((v >= 0) & (v <= 1))


def test_violation_sane():
    body = torch.tensor([[1.0, 1.0]])
    head = torch.tensor([0.0])
    v = horn_violation(body, head)
    assert torch.isclose(v, torch.tensor(1.0))
