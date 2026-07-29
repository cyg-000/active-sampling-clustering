"""
Dataset / train-val-test split / collate.
Split by **base_uuid** (stimulus), NOT by trial at random.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


def load_npz(path=None):
    z = np.load(path or C.DATASET, allow_pickle=True)
    return {k: z[k] for k in z.files}


def make_splits(d, split_by=None, seed=None):
    split_by = split_by or C.SPLIT_BY
    keys = d[split_by]
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed if seed is not None else C.SPLIT_SEED)
    rng.shuffle(uniq)
    n = len(uniq)
    n_te = int(round(n * C.TEST_FRAC))
    n_va = int(round(n * C.VAL_FRAC))
    grp = {"test": set(uniq[:n_te]), "val": set(uniq[n_te:n_te + n_va]),
           "train": set(uniq[n_te + n_va:])}
    return {s: np.where(np.isin(keys, list(g)))[0] for s, g in grp.items()}


def shuffle_labels(lab, rng):
    """
    Dataset / train-val-test split / collate.
    Split by **base_uuid** (stimulus), NOT by trial at random.
    """
    out = lab.copy()
    m = np.where(lab >= 0)[0]
    out[m] = lab[m][rng.permutation(len(m))]
    return out


class Trials(Dataset):
    """Dataset / train-val-test split / collate.
Split by **base_uuid** (stimulus), NOT by trial at random."""

    def __init__(self, d, idx, target="human", scanpath="full", train=False, seed=0):
        self.d, self.idx, self.target = d, np.asarray(idx), target
        self.scanpath, self.train = scanpath, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        d = self.d
        pts = np.asarray(d["pts"][j], np.float32)
        n = len(pts)
        lab = np.asarray(d[self.target if self.target != "shuffle" else "human"][j],
                         np.int64).copy()
        if self.target == "shuffle":
            lab = shuffle_labels(lab, self.rng)
        rev = np.asarray(d["reveal"][j], np.int64)
        is_funnel = d["visibility"][j] == "funnel"

        if is_funnel and (rev >= 0).any():
            t = rev.astype(np.float32)
            t[t < 0] = t[t >= 0].max() if (t >= 0).any() else 0.0
            t = t / max(t.max(), 1.0)
        else:
            t = np.zeros(n, np.float32)
        has_order = np.float32(1.0 if (is_funnel and self.scanpath != "none") else 0.0)
        if self.scanpath == "none":
            t = np.zeros(n, np.float32)

        if is_funnel and (rev >= 0).any() and self.scanpath != "none":
            order = np.argsort(np.where(rev >= 0, rev, 10 ** 6), kind="stable")
        else:
            order = self.rng.permutation(n) if self.train else np.arange(n)

        N = C.MAX_N
        x = np.zeros((N, 4), np.float32)
        x[:n, 0] = pts[:, 0] / C.W
        x[:n, 1] = pts[:, 1] / C.H
        x[:n, 2] = t
        x[:n, 3] = has_order
        y = np.full(N, -1, np.int64)
        y[:n] = lab
        mask = np.zeros(N, bool)
        mask[:n] = True
        ordr = np.arange(N)
        ordr[:n] = order
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask),
                torch.from_numpy(ordr), int(j))


def make_loader(ds, batch=None, shuffle=False):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch or C.BATCH, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)
