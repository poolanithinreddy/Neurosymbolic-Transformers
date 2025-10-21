import torch
import torch.nn as nn


class UnaryHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 1))

    def forward(self, h):  # [B,D] -> [B]
        return torch.sigmoid(self.mlp(h).squeeze(-1))


class BinaryHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 1))

    def forward(self, hx, hy):  # [B,D], [B,D] -> [B]
        z = torch.cat([hx, hy], dim=-1)
        return torch.sigmoid(self.mlp(z).squeeze(-1))
