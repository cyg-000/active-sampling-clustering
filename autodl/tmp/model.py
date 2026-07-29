"""
Point-cloud -> soft assignment matrix P (B, N, K).
Design: soft assignment works for both lasso (integrate contour from assignments)
and voronoi (argmax). Loss is pairwise co-assignment BCE -- invariant to cluster
label permutation and isomorphic to FM. K is emergent (softmax over K_MAX slots),
not a hyperparameter.
"""
import torch
import torch.nn as nn

import config as C


class Block(nn.Module):
    """Point-cloud -> soft assignment matrix P (B, N, K).
Design: soft assignment works for both lasso (integrate contour from assignments)
and voronoi (argmax). Loss is pairwise co-assignment BCE -- invariant to cluster
label permutation and isomorphic to FM. K is emergent (softmax over K_MAX slots),
not a hyperparameter."""

    def __init__(self, d, p):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(),
                               nn.Dropout(p), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, mask):
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1, keepdim=True).values
        z = torch.cat([h, mean.expand_as(h), mx.expand_as(h)], -1)
        return self.norm(h + self.f(z))


class AssignNet(nn.Module):
    def __init__(self, d=C.D_MODEL, k=C.K_MAX, n_blocks=C.N_BLOCKS,
                 use_gru=True, dropout=C.DROPOUT):
        super().__init__()
        self.use_gru = use_gru
        self.embed = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        if use_gru:
            self.gru = nn.GRU(d, C.GRU_HIDDEN, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(2 * C.GRU_HIDDEN, d)
        self.blocks = nn.ModuleList([Block(d, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, k))
        last = self.head[-1]
        nn.init.normal_(last.weight, std=0.5 / d ** 0.5)
        nn.init.uniform_(last.bias, -0.5, 0.5)

    def forward(self, x, mask, order=None):
        """Point-cloud -> soft assignment matrix P (B, N, K).
Design: soft assignment works for both lasso (integrate contour from assignments)
and voronoi (argmax). Loss is pairwise co-assignment BCE -- invariant to cluster
label permutation and isomorphic to FM. K is emergent (softmax over K_MAX slots),
not a hyperparameter."""
        h = self.embed(x)
        if self.use_gru and order is not None:
            idx = order.unsqueeze(-1).expand(-1, -1, h.size(-1))
            hs = torch.gather(h, 1, idx)
            g, _ = self.gru(hs)
            g = self.proj(g)
            back = torch.zeros_like(g).scatter_(1, idx, g)
            h = h + back
        for b in self.blocks:
            h = b(h, mask)
        logits = self.head(h)
        return logits.masked_fill(~mask.unsqueeze(-1), -1e9)


def soft_assign(logits, tau=1.0):
    return torch.softmax(logits / tau, dim=-1)
