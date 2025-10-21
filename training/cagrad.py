import torch


def cagrad(grad_list, c=0.3):
    # grad_list: list of flattened gradients for different losses
    # very small, illustrative combiner: average + conflict correction
    G = torch.stack(grad_list, dim=0)  # [T, P]
    g_avg = G.mean(dim=0)
    # project each onto g_avg and cap negative conflicts
    adjusted = []
    for g in G:
        cos = torch.dot(g, g_avg) / (g.norm() * g_avg.norm() + 1e-8)
        if cos < c:
            # pull towards g_avg
            g = (c / g.norm().clamp(min=1e-8)) * g_avg
        adjusted.append(g)
    g_comb = torch.stack(adjusted, dim=0).mean(dim=0)
    return g_comb
