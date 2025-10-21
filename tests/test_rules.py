import torch
from logic.logic import horn_truth, horn_violation


def test_horn_no_violation_when_head_true():
    body = torch.tensor([[1.0, 0.8, 0.9]])  # body truth >0
    head = torch.tensor([1.0])
    v = horn_violation(body, head)
    assert torch.all(v <= 1e-6)


def test_horn_truth_bounds():
    body = torch.tensor([[0.4, 0.5]])
    head = torch.tensor([0.7])
    t = horn_truth(body, head)
    assert float(t) >= 0.0 and float(t) <= 1.0
