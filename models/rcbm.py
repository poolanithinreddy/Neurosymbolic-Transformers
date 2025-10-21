import torch
import torch.nn as nn

from .heads import BinaryHead, UnaryHead


class RCBM(nn.Module):
    def __init__(self, hidden_dim, pred_info):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.unary = nn.ModuleDict()
        self.binary = nn.ModuleDict()
        for p in pred_info:
            name, arity = p["name"], p["arity"]
            if arity == 1:
                self.unary[name] = UnaryHead(hidden_dim)
            else:
                self.binary[name] = BinaryHead(hidden_dim)

    def forward(self, pooled, ent_reps, pairs):
        """
        pooled: [B,D] global mean-pooled H*
        ent_reps: dict[eid]->[B,D] entity embeddings
        pairs: dict[pred_name]->list of (eid1,eid2) tuples
        returns: dict[predicate]->tensor of scores (B or [B, K])
        """
        out = {}
        # Unary from pooled OR ent reps
        for name, head in self.unary.items():
            if name in ["TrueClaim", "FalseClaim"]:
                out[name] = head(pooled)
            else:
                # if entities exist for this predicate, average
                if ent_reps:
                    ent_stack = torch.stack(list(ent_reps.values()), dim=1)  # [B,N,D]
                    v = head(ent_stack.mean(dim=1))
                else:
                    v = head(pooled)
                out[name] = v
        # Binary using pairs
        for name, head in self.binary.items():
            tuples = pairs.get(name, [])
            if len(tuples) == 0:
                out[name] = torch.zeros(pooled.size(0), device=pooled.device)
                continue
            vals = []
            for e1, e2 in tuples:
                vals.append(head(ent_reps[e1], ent_reps[e2]))
            out[name] = torch.stack(vals, dim=1).mean(dim=1)  # average over candidate groundings
        return out
