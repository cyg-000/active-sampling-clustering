"""
Loss functions.
Primary: pairwise co-assignment BCE. A_ij = sum_k p_ik * p_jk = model probability
that i and j belong to the same cluster. BCE against 1[y_i == y_j].
PAIR_BALANCE must be OFF -- pos_weight rewards merging.
Three human-prior regularisation terms implemented but default to weight=0.
"""
import torch
import torch.nn.functional as F

import config as C


def pair_targets(y, mask):
    """Loss functions.
Primary: pairwise co-assignment BCE. A_ij = sum_k p_ik * p_jk = model probability
that i and j belong to the same cluster. BCE against 1[y_i == y_j].
PAIR_BALANCE must be OFF -- pos_weight rewards merging.
Three human-prior regularisation terms implemented but default to weight=0."""
    valid = mask & (y >= 0)
    V = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(y.size(1), dtype=torch.bool, device=y.device).unsqueeze(0)
    V = V & ~eye
    T = (y.unsqueeze(2) == y.unsqueeze(1)).float()
    return T, V


def pair_loss(P, y, mask, balance=None, eps=1e-7):
    """
    Loss functions.
    Primary: pairwise co-assignment BCE. A_ij = sum_k p_ik * p_jk = model probability
    that i and j belong to the same cluster. BCE against 1[y_i == y_j].
    PAIR_BALANCE must be OFF -- pos_weight rewards merging.
    Three human-prior regularisation terms implemented but default to weight=0.
    """
    balance = C.PAIR_BALANCE if balance is None else balance
    T, V = pair_targets(y, mask)
    A = torch.bmm(P, P.transpose(1, 2)).clamp(eps, 1 - eps)
    Vf = V.float()
    if balance:
        npos = (T * Vf).sum((1, 2)).clamp(min=1)
        nneg = ((1 - T) * Vf).sum((1, 2)).clamp(min=1)
        w = torch.where(T > 0, (nneg / npos).view(-1, 1, 1), torch.ones_like(T))
    else:
        w = torch.ones_like(T)
    bce = -(T * torch.log(A) + (1 - T) * torch.log(1 - A)) * w * Vf
    return (bce.sum((1, 2)) / (w * Vf).sum((1, 2)).clamp(min=1)).mean()


def entropy_loss(P, mask, eps=1e-7):
    """Loss functions.
Primary: pairwise co-assignment BCE. A_ij = sum_k p_ik * p_jk = model probability
that i and j belong to the same cluster. BCE against 1[y_i == y_j].
PAIR_BALANCE must be OFF -- pos_weight rewards merging.
Three human-prior regularisation terms implemented but default to weight=0."""
    H = -(P * torch.log(P + eps)).sum(-1)
    m = mask.float()
    return ((H * m).sum(1) / m.sum(1).clamp(min=1)).mean()


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════

def reg_cluster_count(P, mask, lo=3.0, hi=5.0):
    """Loss functions.
Primary: pairwise co-assignment BCE. A_ij = sum_k p_ik * p_jk = model probability
that i and j belong to the same cluster. BCE against 1[y_i == y_j].
PAIR_BALANCE must be OFF -- pos_weight rewards merging.
Three human-prior regularisation terms implemented but default to weight=0."""
    m = mask.float().unsqueeze(-1)
    q = (P * m).sum(1)
    q = q / q.sum(1, keepdim=True).clamp(min=1e-7)
    k_eff = torch.exp(-(q * torch.log(q + 1e-7)).sum(1))
    return (F.relu(lo - k_eff) + F.relu(k_eff - hi)).mean()


def reg_centroid(P, x, mask):
    """Loss functions.
Primary: pairwise co-assignment BCE. A_ij = sum_k p_ik * p_jk = model probability
that i and j belong to the same cluster. BCE against 1[y_i == y_j].
PAIR_BALANCE must be OFF -- pos_weight rewards merging.
Three human-prior regularisation terms implemented but default to weight=0."""
    m = mask.float().unsqueeze(-1)
    w = P * m
    cen = torch.einsum("bnk,bnd->bkd", w, x[..., :2]) / \
        w.sum(1).unsqueeze(-1).clamp(min=1e-6)
    used = (w.sum(1) > 1e-3).float().unsqueeze(-1)
    pred = (cen * used).sum(1) / used.sum(1).clamp(min=1e-6)
    parr = (x[..., :2] * m).sum(1) / m.sum(1).clamp(min=1e-6)
    return ((pred - parr) ** 2).sum(-1).mean()


def reg_silhouette(P, x, mask, target=0.35):
    """Loss functions.
Primary: pairwise co-assignment BCE. A_ij = sum_k p_ik * p_jk = model probability
that i and j belong to the same cluster. BCE against 1[y_i == y_j].
PAIR_BALANCE must be OFF -- pos_weight rewards merging.
Three human-prior regularisation terms implemented but default to weight=0."""
    d = torch.cdist(x[..., :2], x[..., :2])
    A = torch.bmm(P, P.transpose(1, 2))
    V = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
    intra = (d * A * V).sum((1, 2)) / (A * V).sum((1, 2)).clamp(min=1e-6)
    inter = (d * (1 - A) * V).sum((1, 2)) / ((1 - A) * V).sum((1, 2)).clamp(min=1e-6)
    s = (inter - intra) / torch.maximum(inter, intra).clamp(min=1e-6)
    return ((s - target) ** 2).mean()


def total_loss(P, x, y, mask, cfg=C):
    parts = {"pair": cfg.W_PAIR * pair_loss(P, y, mask)}
    if cfg.W_ENTROPY:
        parts["ent"] = cfg.W_ENTROPY * entropy_loss(P, mask)
    if cfg.W_REG_K:
        parts["reg_k"] = cfg.W_REG_K * reg_cluster_count(P, mask)
    if cfg.W_REG_CENTROID:
        parts["reg_cen"] = cfg.W_REG_CENTROID * reg_centroid(P, x, mask)
    if cfg.W_REG_SIL:
        parts["reg_sil"] = cfg.W_REG_SIL * reg_silhouette(P, x, mask)
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
