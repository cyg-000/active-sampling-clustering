"""Which hazard components generate the granularity scaling? (P0 / reviewer C5).

The sequential-construction result is best read as a *generative sufficiency*
test: once the empirically fitted stopping dynamics are dropped into a minimal
hazard-walk framework, do they reproduce the human cluster-count scaling?  This
module answers the attribution question that framing raises -- which predictors
in the fitted stopping hazard actually carry the generated slope.

Diagnostic pre-publication check, run on the committed anchor states. For each
covariate set the hazard is refit on all 42,161 states, the
walk regenerates k_at_stop per N-bin, and the scaling slope is compared with the
participant-level human slopes (BF, participant as the sampling unit):

* ``full``                  process + Q + dQ (the published generator)
* ``process_only``          process variables only (no quality terms)
* ``process_no_npoints``    process minus the array-size term
* ``process_plus_Q``        process + Q (no dQ)
* ``process_plus_dQ``       process + dQ (no Q)
* ``fixed_k=3``             null reference: a construction that stops at 3
                            groups regardless of the hazard (slope 0)

Interpretation discipline: the scaling-generation claim concerns the *empirical
stopping dynamics as a whole* (participants stop less readily on larger arrays,
so k grows with N); the *quality-guided* claim concerns stopping prediction and
is evidenced separately by the out-of-fold log-loss improvement in
:mod:`stopping_hazard`.  The two claims do not rest on the same test, and this
attribution makes that division explicit rather than leaving it implicit.

Outputs
  process_ablation_slopes.csv  covariate_set x condition: walk slope + CI + BF
                                against participant human slopes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .process_generator import COV, NBINS, mean_state_table, prep, walk_k_matrix, walk_slope
from .stats_process_slopes import participant_slopes, one_sample_against

OUT_DIR = Path(__file__).resolve().parent / "outputs"
STATES = OUT_DIR / "anchor_states.csv"
NPZ = Path(__file__).resolve().parent.parent / "data" / "processed_trials.npz"

CONDS = ("exp3", "exp4")
NBIN_MIDS = np.asarray([0.5 * (lo + hi) for lo, hi in NBINS], float)

PROCESS = ["event_index", "log_elapsed", "k", "revisions_so_far",
           "is_delete", "n_points", "is_exp4"]
SETS = {
    "full": COV,
    "process_only": PROCESS,
    "process_no_npoints": [c for c in PROCESS if c != "n_points"],
    "process_plus_Q": PROCESS + ["Q"],
    "process_plus_dQ": PROCESS + ["dQ"],
}


def run(n_walks: int = 4000, seed: int = 0) -> pd.DataFrame:
    z = prep(pd.read_csv(STATES))
    ms = mean_state_table(z)
    human = participant_slopes()  # per-condition participant slopes

    rows = []
    for name, covs in SETS.items():
        X = z[covs].to_numpy(float)
        y = z["stop"].to_numpy(int)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X), y)
        auc = float(roc_auc_score(y, clf.predict_proba(sc.transform(X))[:, 1]))
        for ci, cond in enumerate(CONDS):
            mat = walk_k_matrix(clf, sc, ms, covs, cond, NBIN_MIDS,
                                n_walks=n_walks, seed=seed + ci)
            slope, (lo, hi) = walk_slope(NBIN_MIDS, mat)
            r = one_sample_against(human[cond], slope)
            rows.append({"covariate_set": name, "n_covariates": len(covs),
                         "auc": auc, "condition": cond,
                         "slope": slope, "ci_lo": lo, "ci_hi": hi,
                         "human_mean_slope": float(human[cond].mean()),
                         "mean_diff": r["mean_diff"], "t": r["t"], "df": r["df"],
                         "p": r["p"], "dz": r["dz"], "BF10": r["BF10"],
                         "BF01": r["BF01"]})
    # null reference: stop at k = 3 regardless of the hazard
    for cond in CONDS:
        r = one_sample_against(human[cond], 0.0)
        rows.append({"covariate_set": "fixed_k=3", "n_covariates": 0, "auc": np.nan,
                     "condition": cond, "slope": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                     "human_mean_slope": float(human[cond].mean()),
                     "mean_diff": r["mean_diff"], "t": r["t"], "df": r["df"],
                     "p": r["p"], "dz": r["dz"], "BF10": r["BF10"],
                     "BF01": r["BF01"]})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-walks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tab = run(args.n_walks, args.seed)
    tab.to_csv(OUT_DIR / "process_ablation_slopes.csv", index=False)

    pd.set_option("display.width", 220)
    print("===== hazard component attribution (walk slope per covariate set) =====")
    print(tab[["covariate_set", "condition", "slope", "ci_lo", "ci_hi",
               "auc", "human_mean_slope", "mean_diff", "BF10", "BF01"]]
          .round(4).to_string(index=False))
    print("\nhuman anchor slopes: exp3 b = 0.0328  exp4 b = 0.0292")
    full = tab[tab.covariate_set == "full"].set_index("condition").slope
    print(f"full-data walk slopes ({args.n_walks} walks): "
          f"exp3 {full['exp3']:.4f}  exp4 {full['exp4']:.4f}")
    print(f"\n-> {OUT_DIR / 'process_ablation_slopes.csv'}")


if __name__ == "__main__":
    main()
