import torch

EPS = 1e-8


def neg(x):
    return 1.0 - x.clamp(0, 1)


def t_and(a, b):
    return a.clamp(0, 1) * b.clamp(0, 1)


def t_or(a, b):  # probabilistic sum
    a = a.clamp(0, 1)
    b = b.clamp(0, 1)
    return a + b - a * b


def imply(a, b):  # Reichenbach
    a = a.clamp(0, 1)
    b = b.clamp(0, 1)
    return 1 - a + a * b


def prod_many(xs):
    # log-space product for stability
    xs = torch.clamp(xs, EPS, 1.0)
    return torch.exp(torch.log(xs).sum(dim=-1))


def forall_meaningful(xs):  # product over a sampled set
    return prod_many(xs)


def exists_meaningful(xs):
    xs = torch.clamp(xs, 0, 1)
    return 1 - prod_many(1 - xs)


def horn_truth(body_vals, head_val):
    t_body = prod_many(body_vals) if body_vals.ndim > 0 else body_vals
    return imply(t_body, head_val)


def horn_violation(body_vals, head_val):
    t_body = prod_many(body_vals) if body_vals.ndim > 0 else body_vals
    return t_body * (1 - head_val.clamp(0, 1))
