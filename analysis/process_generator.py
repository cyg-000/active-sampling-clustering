"""Process-generative granularity (Instrument 2, data-driven).

Simulate the STOPPING dynamics fitted from the human anchor-construction
trajectories, and test whether they generate the observed cluster-count scaling
(k vs N) per condition -- without any hand-designed quality-optimal add policy.

Mechanism: the fitted hazard's P(stop) at a fixed k declines with array size
(participants stop less readily on larger arrays), so the stopping dynamics
alone generate k growing with N.

Walk: start at k = 2 (the design floor -- a single-group response was rejected).
At step k, sample stop ~ Bernoulli(P_stop(k, N, condition)); if stop, k_at_stop =
k; else k -> k+1.  P_stop is the fitted penalized logistic hazard evaluated at
the empirical mean-covariate state for each (k, N-bin, condition).  Averaging
walks per N yields predicted k_at_stop(N) and a predicted scaling slope.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).resolve().parent / "outputs"
STATES = OUT_DIR / "anchor_states.csv"

COV = ["event_index", "log_elapsed", "k", "revisions_so_far", "is_delete",
       "n_points", "is_exp4", "Q", "dQ"]
CONDS = ("exp3", "exp4")
NBINS = [(10, 15), (20, 25), (30, 35), (40, 40)]
NWALKS = 4000
RNG = np.random.default_rng(0)

# Column index of each covariate inside the full hazard design vector
# [event_index, log_elapsed, k, revisions_so_far, is_delete, n_points,
#  is_exp4, Q, dQ], used to walk with arbitrary covariate subsets (ablation)
# without re-encoding the states.
_FULL_COLS = {"event_index": 0, "log_elapsed": 1, "k": 2, "revisions_so_far": 3,
              "is_delete": 4, "n_points": 5, "is_exp4": 6, "Q": 7, "dQ": 8}
_NBIN_LABELS = [f"{lo}-{hi}" for lo, hi in NBINS]


def mean_state_table(z: pd.DataFrame) -> pd.DataFrame:
    """Empirical mean covariates of human *non-terminal* states per (exp, k, nbin).

    The walk fixes its auxiliary covariates (elapsed time, revisions so far,
    current quality and quality change) to these values at every cluster
    count, so generation needs no hand-designed add policy.  Only non-stop
    states are used, matching the states at which a decision to continue is
    actually observed.
    """
    z = z.copy()
    z["nbin"] = z["n_points"].map(dict(zip([lo for lo, _ in NBINS], _NBIN_LABELS)))
    for lo, hi in NBINS:
        z.loc[(z.n_points >= lo) & (z.n_points <= hi), "nbin"] = f"{lo}-{hi}"
    return (z[z.stop == 0].groupby(["exp", "k", "nbin"])
            [["log_elapsed", "revisions_so_far", "Q", "dQ"]].mean().reset_index())


def walk_k_matrix(clf, scaler, mean_state, covs, cond, nmid, n_walks=4000,
                  seed=0, max_k=16, start_k=2) -> np.ndarray:
    """Hazard-walk stop counts: (n_bins, n_walks) matrix of k_at_stop.

    For every N-bin, start at ``start_k`` clusters; at cluster count ``k``,
    stop with probability P_stop(k) from the fitted hazard evaluated at the
    empirical mean state for (exp=cond, k, nbin), otherwise increment k.
    P_stop depends only on (k, nbin, cond, covs), so it is precomputed once
    per (k, nbin) and the per-walk loop is pure RNG -- fast and reproducible.
    Cells with no empirical state fall back to the (k, condition) marginal
    mean; if that is also missing the walk stops at the current k.
    """
    is_exp4 = 1.0 if cond == "exp4" else 0.0
    sel = [_FULL_COLS[c] for c in covs]
    P = np.full((max_k + 1, len(nmid)), np.nan, float)
    fallback = {k: g[["log_elapsed", "revisions_so_far", "Q", "dQ"]].mean()
                for k, g in mean_state[mean_state.exp == cond].groupby("k")}
    for bi, (lo, hi) in enumerate(NBINS):
        for k in range(start_k, max_k + 1):
            ms = mean_state[(mean_state.exp == cond) & (mean_state.k == k)
                            & (mean_state.nbin == _NBIN_LABELS[bi])]
            if len(ms):
                m = ms.iloc[0]
            elif k in fallback:
                m = fallback[k]
            else:
                continue
            full = np.array([k, m.log_elapsed, k, m.revisions_so_far, 0.0,
                             nmid[bi], is_exp4, m.Q, m.dQ], float)
            P[k, bi] = float(clf.predict_proba(
                scaler.transform(full[sel].reshape(1, -1)))[0, 1])
    rng = np.random.default_rng(seed)
    out = np.zeros((len(nmid), n_walks), float)
    for bi in range(len(nmid)):
        for _ in range(n_walks):
            k = start_k
            while k <= max_k:
                p = P[k, bi]
                if not np.isfinite(p):
                    break
                if rng.random() < p:
                    break
                k += 1
            out[bi, _] = min(k, max_k)
    return out


def walk_slope(nmid: np.ndarray, mat: np.ndarray, n_boot: int = 1000,
               seed: int = 1) -> tuple[float, tuple[float, float]]:
    """OLS slope of mean k_at_stop across N-bins + bootstrap CI over walks."""
    point = float(np.polyfit(nmid, mat.mean(1), 1)[0])
    rng = np.random.default_rng(seed)
    n_walks = mat.shape[1]
    slopes = []
    for _ in range(n_boot):
        ks = np.asarray([mat[bi, rng.integers(0, n_walks, n_walks)].mean()
                         for bi in range(mat.shape[0])], float)
        slopes.append(float(np.polyfit(nmid, ks, 1)[0]))
    return point, (float(np.percentile(slopes, 2.5)),
                   float(np.percentile(slopes, 97.5)))


def prep(z: pd.DataFrame) -> pd.DataFrame:
    z = z.copy()
    z["log_elapsed"] = np.log1p(z["elapsed_s"].clip(lower=0))
    z["is_delete"] = (z["action"] == "delete").astype(int)
    z["is_exp4"] = (z["exp"] == "exp4").astype(int)
    z["dQ"] = z["delta_silhouette"].fillna(0.0)
    z = z.rename(columns={"silhouette": "Q"})
    return z.replace([np.inf, -np.inf], np.nan).dropna(subset=["Q", "dQ"])


def main() -> None:
    z = prep(pd.read_csv(STATES))
    X = z[COV].to_numpy(float)
    y = z["stop"].to_numpy(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(sc.transform(X), y)
    print("hazard AUC (in-sample):",
          round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"])
                     .roc_auc_score(y, clf.predict_proba(sc.transform(X))[:, 1])), 3))

    # empirical mean covariates per (exp, k, N-bin)
    z["nbin"] = z["n_points"].map({lo: f"{lo}-{hi}" for lo, hi in NBINS})
    for lo, hi in NBINS:
        z.loc[(z.n_points >= lo) & (z.n_points <= hi), "nbin"] = f"{lo}-{hi}"
    mean_state = (z[z.stop == 0].groupby(["exp", "k", "nbin"])
                  [["log_elapsed", "revisions_so_far", "Q", "dQ"]]
                  .mean().reset_index())

    pred_rows = []
    nmid = np.asarray([0.5 * (lo + hi) for lo, hi in NBINS], float)
    for ci, cond in enumerate(CONDS):
        mat = walk_k_matrix(clf, sc, mean_state, COV, cond, nmid,
                            n_walks=NWALKS, seed=ci)
        for bi, value in enumerate(nmid):
            pred_rows.append({"condition": cond, "n_mid": value,
                              "k_at_stop": float(mat[bi].mean())})
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(OUT_DIR / "process_generator_predictions.csv", index=False)

    print("\n===== process-generative k_at_stop vs N (hazard walk) =====")
    for cond in CONDS:
        p = pred[pred.condition == cond]
        slope = float(np.polyfit(p.n_mid, p.k_at_stop, 1)[0])
        print(f"{cond}: slope = {slope:+.4f}   k(N): "
              + ", ".join(f"N={int(r.n_mid)}→{r.k_at_stop:.2f}"
                          for _, r in p.iterrows()))
    print("\nhuman anchor slopes: exp3 b = 0.0328   exp4 b = 0.0296")
    print(f"\n-> {OUT_DIR / 'process_generator_predictions.csv'}")


if __name__ == "__main__":
    main()
