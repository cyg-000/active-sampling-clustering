"""
Pre-load entire dataset into GPU memory for pure tensor indexing.
Model is tiny (N<=40, d=128). Full dataset (~15 MB) fits in VRAM.
"""
import numpy as np
import torch

import config as C


def build_tensors(d, idx, target="human", scanpath="full", seed=0, device="cpu"):
    idx = np.asarray(idx)
    B, N = len(idx), C.MAX_N
    rng = np.random.default_rng(seed)

    X = np.zeros((B, N, 4), np.float32)
    Y = np.full((B, N), -1, np.int64)
    M = np.zeros((B, N), bool)
    O = np.tile(np.arange(N), (B, 1)).astype(np.int64)
    F = np.zeros(B, bool)

    src = "human" if target == "shuffle" else target
    for b, j in enumerate(idx):
        pts = np.asarray(d["pts"][j], np.float32)
        n = len(pts)
        lab = np.asarray(d[src][j], np.int64).copy()
        if target == "shuffle":
            m = np.where(lab >= 0)[0]
            lab[m] = lab[m][rng.permutation(len(m))]
        rev = np.asarray(d["reveal"][j], np.int64)
        is_f = bool(d["visibility"][j] == "funnel") and (rev >= 0).any() \
            and scanpath != "none"

        X[b, :n, 0] = pts[:, 0] / C.W
        X[b, :n, 1] = pts[:, 1] / C.H
        if is_f:
            t = np.where(rev >= 0, rev, rev.max()).astype(np.float32)
            X[b, :n, 2] = t / max(t.max(), 1.0)
            X[b, :n, 3] = 1.0
            O[b, :n] = np.argsort(np.where(rev >= 0, rev, 10 ** 6), kind="stable")
        Y[b, :n] = lab
        M[b, :n] = True
        F[b] = is_f

    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return dict(X=t(X, torch.float32), Y=t(Y, torch.int64), M=t(M, torch.bool),
                O=t(O, torch.int64), F=t(F, torch.bool), n=B)


def epoch_order(T, gen):
    """
    Pre-load entire dataset into GPU memory for pure tensor indexing.
    Model is tiny (N<=40, d=128). Full dataset (~15 MB) fits in VRAM.
    """
    B, N = T["O"].shape
    key = torch.rand(B, N, device=T["O"].device) + (~T["M"]).float() * 10.0
    rnd = key.argsort(dim=1)
    return torch.where(T["F"].view(-1, 1), T["O"], rnd)


def batches(T, batch, shuffle, gen=None):
    n = T["n"]
    order = torch.randperm(n, device=T["X"].device, generator=gen) if shuffle \
        else torch.arange(n, device=T["X"].device)
    for i in range(0, n, batch):
        yield order[i:i + batch]
