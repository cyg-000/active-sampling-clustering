"""Shared, dependency-light helpers for the revision analyses.

The neural-network code deliberately keeps its own training utilities.  This
module is CPU-only and provides one canonical definition of condition labels,
partition summaries, stimulus-disjoint splits, and stimulus-equal slopes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO / "data" / "processed_trials.npz"
EXPERIMENTS = ("exp1", "exp2", "exp3", "exp4")
CONDITION_META = {
    "exp1": ("lasso", "full"),
    "exp2": ("lasso", "funnel"),
    "exp3": ("voronoi", "full"),
    "exp4": ("voronoi", "funnel"),
}


def load_dataset(path: str | Path = DEFAULT_DATASET) -> dict[str, np.ndarray]:
    z = np.load(Path(path), allow_pickle=True)
    return {k: z[k] for k in z.files}


def stimulus_split(
    d: Mapping[str, np.ndarray],
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    """Reproduce autodl/tmp/data.py::make_splits for base_uuid."""
    keys = np.asarray(d["base_uuid"])
    uniq = np.unique(keys).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    n_val = int(round(len(uniq) * val_frac))
    groups = {
        "test": set(uniq[:n_test]),
        "val": set(uniq[n_test:n_test + n_val]),
        "train": set(uniq[n_test + n_val:]),
    }
    return {
        name: np.where(np.isin(keys, list(group)))[0]
        for name, group in groups.items()
    }


def valid_labels(labels: Sequence[int]) -> np.ndarray:
    x = np.asarray(labels, int)
    return x[x >= 0]


def cluster_count(labels: Sequence[int]) -> float:
    x = valid_labels(labels)
    return float(len(np.unique(x))) if len(x) else np.nan


def mean_cluster_size(labels: Sequence[int]) -> float:
    x = valid_labels(labels)
    if not len(x):
        return np.nan
    _, counts = np.unique(x, return_counts=True)
    return float(np.mean(counts))


def ols_slope(x: Sequence[float], y: Sequence[float], weights=None) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        w = np.asarray(weights, float)
        ok &= np.isfinite(w) & (w > 0)
        w = w[ok]
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) <= 0:
        return np.nan
    X = np.column_stack([np.ones(len(x)), x])
    if weights is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return float(beta[1])


def condition_mask(d: Mapping[str, np.ndarray], exp: str, idx=None) -> np.ndarray:
    base = np.arange(len(d["exp"])) if idx is None else np.asarray(idx, int)
    return base[np.asarray(d["exp"])[base] == exp]


def stimulus_equal_summary(
    d: Mapping[str, np.ndarray],
    idx: Sequence[int],
    labels: Sequence[Sequence[int]] | None = None,
    trial_weights: Sequence[float] | None = None,
    stimulus_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Estimate slopes after giving each base stimulus equal weight.

    Human trials are first averaged within base_uuid.  Model predictions can be
    passed through ``labels`` using the same row order as ``idx``.  Optional
    trial/stimulus weights support the participant x stimulus bootstrap.
    """
    idx = np.asarray(idx, int)
    labs = [d["human"][j] for j in idx] if labels is None else list(labels)
    if len(labs) != len(idx):
        raise ValueError("labels must have the same length/order as idx")
    tw = np.ones(len(idx), float) if trial_weights is None else np.asarray(trial_weights, float)
    by_base: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for pos, j in enumerate(idx):
        k = cluster_count(labs[pos])
        size = mean_cluster_size(labs[pos])
        if np.isfinite(k) and np.isfinite(size) and tw[pos] > 0:
            by_base[str(d["base_uuid"][j])].append(
                (float(d["n_points"][j]), k, size, float(tw[pos]))
            )
    npts, ks, sizes, sw = [], [], [], []
    for base, rows in by_base.items():
        a = np.asarray(rows, float)
        w = a[:, 3]
        npts.append(float(np.average(a[:, 0], weights=w)))
        ks.append(float(np.average(a[:, 1], weights=w)))
        sizes.append(float(np.average(a[:, 2], weights=w)))
        sw.append(float(stimulus_weights.get(base, 1.0)) if stimulus_weights else 1.0)
    return {
        "n_trials": int(np.sum(tw > 0)),
        "n_stimuli": len(npts),
        "k_mean": float(np.average(ks, weights=sw)) if ks else np.nan,
        "cluster_size_mean": float(np.average(sizes, weights=sw)) if sizes else np.nan,
        "slope_k_vs_n": ols_slope(npts, ks, sw),
        "slope_size_vs_n": ols_slope(npts, sizes, sw),
    }


def participant_stimulus_bootstrap(
    d: Mapping[str, np.ndarray],
    idx: Sequence[int],
    n_boot: int = 1000,
    seed: int = 20260814,
) -> dict[str, tuple[float, float]]:
    """Bootstrap participants and stimuli independently within one condition."""
    idx = np.asarray(idx, int)
    sids = np.asarray(d["sid"])[idx].astype(str)
    bases = np.asarray(d["base_uuid"])[idx].astype(str)
    unique_sids, sid_code = np.unique(sids, return_inverse=True)
    unique_bases, base_code = np.unique(bases, return_inverse=True)
    k_trial = np.asarray([cluster_count(d["human"][j]) for j in idx], float)
    size_trial = np.asarray([mean_cluster_size(d["human"][j]) for j in idx], float)
    n_trial = np.asarray(d["n_points"])[idx].astype(float)
    valid = np.isfinite(k_trial) & np.isfinite(size_trial)
    # n_points is constant within a base stimulus; keep one value per base.
    n_by_base = np.zeros(len(unique_bases), float)
    for b in range(len(unique_bases)):
        n_by_base[b] = np.nanmedian(n_trial[base_code == b])
    rng = np.random.default_rng(seed)
    vals = {"slope_k_vs_n": [], "slope_size_vs_n": []}
    for _ in range(n_boot):
        sid_draw = rng.integers(0, len(unique_sids), len(unique_sids))
        sid_counts = np.bincount(sid_draw, minlength=len(unique_sids)).astype(float)
        stim_draw = rng.integers(0, len(unique_bases), len(unique_bases))
        stim_counts = np.bincount(stim_draw, minlength=len(unique_bases)).astype(float)
        trial_w = sid_counts[sid_code] * valid
        den = np.bincount(base_code, weights=trial_w, minlength=len(unique_bases))
        ok = (den > 0) & (stim_counts > 0)
        if ok.sum() < 3:
            continue
        k_by_base = np.divide(
            np.bincount(base_code, weights=trial_w * np.nan_to_num(k_trial),
                        minlength=len(unique_bases)),
            den, out=np.full(len(unique_bases), np.nan), where=den > 0,
        )
        size_by_base = np.divide(
            np.bincount(base_code, weights=trial_w * np.nan_to_num(size_trial),
                        minlength=len(unique_bases)),
            den, out=np.full(len(unique_bases), np.nan), where=den > 0,
        )
        sk = ols_slope(n_by_base[ok], k_by_base[ok], stim_counts[ok])
        ss = ols_slope(n_by_base[ok], size_by_base[ok], stim_counts[ok])
        if np.isfinite(sk):
            vals["slope_k_vs_n"].append(sk)
        if np.isfinite(ss):
            vals["slope_size_vs_n"].append(ss)
    return {
        key: tuple(np.quantile(v, [0.025, 0.975])) if v else (np.nan, np.nan)
        for key, v in vals.items()
    }
