"""
Architecture ablation: separate two candidate causes of slope_k failure.
Background: tmp/ Model-H reproduces four human statistics never in the loss
but fails to learn slope_k. 3x2 ablation crosses {meanmax, dpool, attn} x
{with/without explicit point count}. All variants permutation-equivariant.
"""
import torch
import torch.nn as nn

import config as C


class MeanMaxBlock(nn.Module):
    """Architecture ablation: separate two candidate causes of slope_k failure.
Background: tmp/ Model-H reproduces four human statistics never in the loss
but fails to learn slope_k. 3x2 ablation crosses {meanmax, dpool, attn} x
{with/without explicit point count}. All variants permutation-equivariant."""

    def __init__(self, d, p):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(),
                               nn.Dropout(p), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, mask, dist):
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1, keepdim=True).values
        z = torch.cat([h, mean.expand_as(h), mx.expand_as(h)], -1)
        return self.norm(h + self.f(z))


class DPoolBlock(nn.Module):
    """
    Architecture ablation: separate two candidate causes of slope_k failure.
    Background: tmp/ Model-H reproduces four human statistics never in the loss
    but fails to learn slope_k. 3x2 ablation crosses {meanmax, dpool, attn} x
    {with/without explicit point count}. All variants permutation-equivariant.
    """

    def __init__(self, d, p):
        super().__init__()
        S = C.N_SIGMA
        self.register_buffer("sig", torch.tensor(C.SIGMAS).view(1, -1, 1, 1))
        self.f = nn.Sequential(nn.Linear((3 + S) * d + S, 2 * d), nn.GELU(),
                               nn.Dropout(p), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, mask, dist):
        B, N, D = h.shape
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1, keepdim=True).values
        w = torch.exp(-(dist.unsqueeze(1) ** 2) / (2 * self.sig ** 2))    # (B,S,N,N)
        w = w * mask[:, None, None, :].float()
        loc = (w @ h.unsqueeze(1)) / w.sum(-1, keepdim=True).clamp(min=1e-6)  # (B,S,N,D)
        dens = w.sum(-1) / mask.sum(1).view(B, 1, 1).clamp(min=1).float()     # (B,S,N)
        z = torch.cat([h, mean.expand_as(h), mx.expand_as(h),
                       loc.permute(0, 2, 1, 3).reshape(B, N, -1),
                       dens.permute(0, 2, 1)], -1)
        return self.norm(h + self.f(z))


class AttnBlock(nn.Module):
    """Architecture ablation: separate two candidate causes of slope_k failure.
Background: tmp/ Model-H reproduces four human statistics never in the loss
but fails to learn slope_k. 3x2 ablation crosses {meanmax, dpool, attn} x
{with/without explicit point count}. All variants permutation-equivariant."""

    def __init__(self, d, p, heads=4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.dbias = nn.Sequential(nn.Linear(C.N_RBF, 32), nn.GELU(), nn.Linear(32, heads))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p),
                                nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(p)
        centers = torch.linspace(0, 1, C.N_RBF).view(1, 1, 1, -1)
        self.register_buffer("centers", centers)

    def forward(self, h, mask, dist):
        B, N, D = h.shape
        q, k, v = self.qkv(self.n1(h)).chunk(3, -1)
        sh = lambda t: t.view(B, N, self.h, self.dk).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(-1, -2)) / self.dk ** 0.5
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * C.RBF_W ** 2))
        att = att + self.dbias(rbf).permute(0, 3, 1, 2)
        att = att.masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, N, D)
        h = h + self.o(out)
        return h + self.ff(self.n2(h))


class AssignNet2(nn.Module):
    def __init__(self, pairing="meanmax", use_ncount=False, d=C.D_MODEL,
                 k=C.K_MAX, n_blocks=C.N_BLOCKS, use_gru=False, dropout=C.DROPOUT):
        super().__init__()
        self.pairing, self.use_ncount, self.use_gru = pairing, use_ncount, use_gru
        in_dim = 4 + (2 if use_ncount else 0)
        self.embed = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        if use_gru:
            self.gru = nn.GRU(d, C.GRU_HIDDEN, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(2 * C.GRU_HIDDEN, d)
        B = {"attn": AttnBlock, "dpool": DPoolBlock}.get(pairing, MeanMaxBlock)
        self.blocks = nn.ModuleList([B(d, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, k))
        last = self.head[-1]
        nn.init.normal_(last.weight, std=0.5 / d ** 0.5)
        nn.init.uniform_(last.bias, -0.5, 0.5)

    def forward(self, x, mask, order=None, return_hidden=False):
        n = mask.sum(1, keepdim=True).float()
        if self.use_ncount:
            f = torch.cat([n / C.MAX_N, torch.log(n) / 4.0], -1)   # (B,2)
            x = torch.cat([x, f.unsqueeze(1).expand(-1, x.size(1), -1)], -1)
        h = self.embed(x)
        if self.use_gru and order is not None:
            idx = order.unsqueeze(-1).expand(-1, -1, h.size(-1))
            g = self.proj(self.gru(torch.gather(h, 1, idx))[0])
            h = h + torch.zeros_like(g).scatter_(1, idx, g)
        p = x[..., :2]
        dist = torch.cdist(p, p) if self.pairing in ("attn", "dpool") else None
        hs = []
        for b in self.blocks:
            h = b(h, mask, dist)
            if return_hidden:
                hs.append(h)
        logits = self.head(h).masked_fill(~mask.unsqueeze(-1), -1e9)
        return (logits, hs) if return_hidden else logits


def soft_assign(logits, tau=1.0):
    return torch.softmax(logits / tau, dim=-1)
