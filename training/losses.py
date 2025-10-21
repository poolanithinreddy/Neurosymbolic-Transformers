import torch
import torch.nn as nn

bce = nn.BCELoss()


def task_loss_classification(logits, gold_ids):
    return nn.CrossEntropyLoss()(logits, gold_ids)


def concept_bce(pred, target, weight=1.0):
    return weight * bce(pred, target)


def logic_loss(rule_violations):
    # rule_violations: list of (violation_value * weight)
    if len(rule_violations) == 0:
        return torch.tensor(0.0, device="cpu")
    return torch.stack(rule_violations).mean()
