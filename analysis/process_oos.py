"""Repeated out-of-sample generative tests of the stopping-hazard walk.

The hazard and empirical mean-state table are estimated from training groups
only. Repeated participant-grouped CV is primary and repeated stimulus-grouped
CV is a robustness test. Slope agreement, absolute K calibration, and stimulus
agreement beyond their common N trend are reported separately.

The default slope equivalence margin is +/-0.01 K/point (about +/-0.3 groups
across N=10--40). It is configurable and is not estimated from these data.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .inference import one_sample
from .process_generator import COV, NBINS, mean_state_table, prep, walk_k_matrix, walk_slope

OUT_DIR = Path(__file__).resolve().parent / "outputs"
STATES = OUT_DIR / "anchor_states.csv"
CONDS = ("exp3", "exp4")
NBIN_MIDS = np.asarray([0.5 * (lo + hi) for lo, hi in NBINS], float)
NBIN_LABELS = [f"{lo}-{hi}" for lo, hi in NBINS]


def assign_nbin(z: pd.DataFrame) -> pd.DataFrame:
    z = z.copy(); z["nbin"] = pd.NA
    for lo, hi in NBINS:
        z.loc[z["n_points"].between(lo, hi), "nbin"] = f"{lo}-{hi}"
    return z


def fit_hazard(states: pd.DataFrame):
    z = prep(states); x = z[COV].to_numpy(float); y = z["stop"].to_numpy(int)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(scaler.transform(x), y)
    return clf, scaler, z


def grouped_folds(values, folds: int, repeats: int, seed: int):
    values = np.asarray(sorted(map(str, np.unique(values))))
    for repeat in range(repeats):
        shuffled = values.copy()
        np.random.default_rng(seed + repeat * 1009).shuffle(shuffled)
        for fold, test in enumerate(np.array_split(shuffled, folds), 1):
            yield repeat + 1, fold, set(map(str, test))


def slope_xy(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) <= 0:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def participant_slopes(term: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, person), g in term.groupby(["exp", "person_hash"]):
        b = slope_xy(g["n_points"], g["k"])
        if np.isfinite(b):
            rows.append({"condition": cond, "person_hash": person,
                         "observed_slope": b, "n_trials": len(g)})
    return pd.DataFrame(rows)


def tost_equivalence(x: np.ndarray, margin: float) -> dict:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / math.sqrt(n)
    df = n - 1
    p_lower = float(sps.t.sf((mean + margin) / se, df))
    p_upper = float(sps.t.cdf((mean - margin) / se, df))
    crit = float(sps.t.ppf(.95, df)); p = max(p_lower, p_upper)
    return {"equivalence_margin": margin, "tost_p_lower": p_lower,
            "tost_p_upper": p_upper, "tost_p": p,
            "equivalent_05": bool(p < .05),
            "difference_ci90_lo": mean - crit * se,
            "difference_ci90_hi": mean + crit * se}


def bootstrap_mean(x: np.ndarray, n_boot: int, seed: int, margin: float) -> dict:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return {"bootstrap_ci_lo": float(np.quantile(boot, .025)),
            "bootstrap_ci_hi": float(np.quantile(boot, .975)),
            "bootstrap_rope_probability": float(np.mean(np.abs(boot) <= margin))}


def error_summary(rows: pd.DataFrame, cluster: str, n_boot: int, seed: int) -> dict:
    d = rows.dropna(subset=["observed_k", "predicted_k"]).copy()
    d["error"] = d.observed_k - d.predicted_k
    out = {"k_bias": float(d.error.mean()), "k_mae": float(d.error.abs().mean()),
           "k_rmse": float(np.sqrt(np.mean(d.error ** 2))),
           "n_k_units": int(d[cluster].nunique())}
    groups = [g.error.to_numpy(float) for _, g in d.groupby(cluster)]
    rng = np.random.default_rng(seed); vals = {k: [] for k in ("k_bias", "k_mae", "k_rmse")}
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), len(groups))
        e = np.concatenate([groups[i] for i in draw])
        vals["k_bias"].append(e.mean()); vals["k_mae"].append(np.abs(e).mean())
        vals["k_rmse"].append(np.sqrt(np.mean(e ** 2)))
    for name, v in vals.items():
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = map(float, np.quantile(v, [.025, .975]))
    return out


def participant_level(states, folds, repeats, n_walks, n_boot, margin, seed):
    fold_rows, slope_rows, k_rows, kcurve = [], [], [], []
    for repeat, fold, test_ids in grouped_folds(states.person_hash, folds, repeats, seed):
        is_test = states.person_hash.astype(str).isin(test_ids)
        train, test = states[~is_test], states[is_test]
        clf, scaler, ztrain = fit_hazard(train); mean_state = mean_state_table(ztrain)
        terminal = test[test.stop == 1]; observed_slopes = participant_slopes(terminal)
        for ci, cond in enumerate(CONDS):
            mat = walk_k_matrix(clf, scaler, mean_state, COV, cond, NBIN_MIDS,
                                n_walks=n_walks, seed=seed + repeat * 10007 + fold * 101 + ci)
            generated_slope, generated_ci = walk_slope(NBIN_MIDS, mat, seed=seed + repeat + fold)
            obs = observed_slopes[observed_slopes.condition == cond]
            fold_rows.append({"repeat": repeat, "fold": fold, "condition": cond,
                              "n_train_identities": train.person_hash.nunique(),
                              "n_test_identities": len(obs), "generated_slope": generated_slope,
                              "generated_ci_lo": generated_ci[0], "generated_ci_hi": generated_ci[1],
                              "observed_mean_slope": float(obs.observed_slope.mean())})
            for r in obs.itertuples(index=False):
                slope_rows.append({"repeat": repeat, "fold": fold, "condition": cond,
                                   "person_hash": r.person_hash, "observed_slope": r.observed_slope,
                                   "generated_slope": generated_slope})
            tcond = terminal[terminal.exp == cond]
            for bi, label in enumerate(NBIN_LABELS):
                predicted = float(mat[bi].mean()); tb = tcond[tcond.nbin == label]
                kcurve.append({"split": "participant", "repeat": repeat, "fold": fold,
                               "condition": cond, "n_mid": NBIN_MIDS[bi], "generated_k": predicted,
                               "observed_k": float(tb.k.mean()) if len(tb) else np.nan})
                for person, g in tb.groupby("person_hash"):
                    k_rows.append({"repeat": repeat, "condition": cond, "person_hash": person,
                                   "n_mid": NBIN_MIDS[bi], "observed_k": float(g.k.mean()),
                                   "predicted_k": predicted})
    slopes = pd.DataFrame(slope_rows)
    person = (slopes.groupby(["condition", "person_hash"], as_index=False)
              .agg(observed_slope=("observed_slope", "mean"),
                   generated_slope=("generated_slope", "mean"),
                   n_oos_predictions=("repeat", "nunique")))
    person["slope_difference"] = person.observed_slope - person.generated_slope
    kdetail = (pd.DataFrame(k_rows).groupby(["condition", "person_hash", "n_mid"], as_index=False)
               .agg(observed_k=("observed_k", "mean"), predicted_k=("predicted_k", "mean"),
                    n_oos_predictions=("repeat", "nunique")))
    summaries = []
    for ci, cond in enumerate(CONDS):
        p = person[person.condition == cond]; inf = one_sample(p.slope_difference, 0.0)
        row = {"condition": cond, "repeats": repeats, "folds": folds,
               "n_participants": len(p), "observed_mean_slope": float(p.observed_slope.mean()),
               "generated_mean_slope": float(p.generated_slope.mean()),
               "slope_difference": float(p.slope_difference.mean()),
               "difference_ci95_lo": inf["ci_low"], "difference_ci95_hi": inf["ci_high"],
               "difference_p": inf["p_two_sided"], "BF10": inf["bf10"], "BF01": inf["bf01"]}
        row.update(tost_equivalence(p.slope_difference.to_numpy(float), margin))
        row.update(bootstrap_mean(p.slope_difference.to_numpy(float), n_boot, seed + ci, margin))
        row.update(error_summary(kdetail[kdetail.condition == cond], "person_hash", n_boot, seed + 20 + ci))
        summaries.append(row)
    return pd.DataFrame(fold_rows), person, kdetail, pd.DataFrame(summaries), kcurve


def residual_correlation(x, y, control) -> float:
    x, y, control = map(lambda v: np.asarray(v, float), (x, y, control))
    c = np.column_stack([np.ones(len(control)), control])
    rx = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
    ry = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    return float(sps.pearsonr(rx, ry).statistic)


def stimulus_bootstrap(d, n_boot, seed):
    rng = np.random.default_rng(seed); a = d[["n_points", "observed_k", "predicted_k"]].to_numpy(float)
    vals = {k: [] for k in ("slope_difference", "detrended_r", "mae", "rmse", "bias")}
    for _ in range(n_boot):
        b = a[rng.integers(0, len(a), len(a))]
        if np.ptp(b[:, 0]) <= 0: continue
        e = b[:, 1] - b[:, 2]
        vals["slope_difference"].append(slope_xy(b[:, 0], b[:, 1]) - slope_xy(b[:, 0], b[:, 2]))
        vals["detrended_r"].append(residual_correlation(b[:, 1], b[:, 2], b[:, 0]))
        vals["mae"].append(np.abs(e).mean()); vals["rmse"].append(np.sqrt(np.mean(e ** 2)))
        vals["bias"].append(e.mean())
    out = {}
    for name, v in vals.items():
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = map(float, np.nanquantile(v, [.025, .975]))
    return out


def stimulus_level(states, folds, repeats, n_walks, n_boot, seed):
    fold_rows, pred_rows, kcurve = [], [], []
    for repeat, fold, test_ids in grouped_folds(states.base_uuid, folds, repeats, seed + 500003):
        is_test = states.base_uuid.astype(str).isin(test_ids)
        train, test = states[~is_test], states[is_test]
        clf, scaler, ztrain = fit_hazard(train); mean_state = mean_state_table(ztrain)
        terminal = test[test.stop == 1]
        for ci, cond in enumerate(CONDS):
            mat = walk_k_matrix(clf, scaler, mean_state, COV, cond, NBIN_MIDS,
                                n_walks=n_walks, seed=seed + 700001 + repeat * 10007 + fold * 101 + ci)
            generated_slope, generated_ci = walk_slope(NBIN_MIDS, mat, seed=seed + repeat + fold)
            tcond = terminal[terminal.exp == cond]
            per_stim = (tcond.groupby("base_uuid", as_index=False)
                        .agg(n_points=("n_points", "mean"), observed_k=("k", "mean"), n_trials=("k", "size")))
            fold_rows.append({"repeat": repeat, "fold": fold, "condition": cond,
                              "n_train_stimuli": train.base_uuid.nunique(), "n_test_stimuli": len(per_stim),
                              "generated_slope": generated_slope, "generated_ci_lo": generated_ci[0],
                              "generated_ci_hi": generated_ci[1],
                              "observed_slope": slope_xy(per_stim.n_points, per_stim.observed_k)})
            for bi, label in enumerate(NBIN_LABELS):
                predicted = float(mat[bi].mean()); tb = tcond[tcond.nbin == label]
                kcurve.append({"split": "stimulus", "repeat": repeat, "fold": fold,
                               "condition": cond, "n_mid": NBIN_MIDS[bi], "generated_k": predicted,
                               "observed_k": float(tb.k.mean()) if len(tb) else np.nan})
                lo, hi = NBINS[bi]
                for r in per_stim[per_stim.n_points.between(lo, hi)].itertuples(index=False):
                    pred_rows.append({"repeat": repeat, "condition": cond, "base_uuid": r.base_uuid,
                                      "n_points": r.n_points, "observed_k": r.observed_k,
                                      "predicted_k": predicted})
    predictions = (pd.DataFrame(pred_rows).groupby(["condition", "base_uuid"], as_index=False)
                   .agg(n_points=("n_points", "mean"), observed_k=("observed_k", "mean"),
                        predicted_k=("predicted_k", "mean"), n_oos_predictions=("repeat", "nunique")))
    summaries = []
    for ci, cond in enumerate(CONDS):
        d = predictions[predictions.condition == cond]; e = d.observed_k - d.predicted_k
        row = {"condition": cond, "repeats": repeats, "folds": folds, "n_stimuli": len(d),
               "observed_slope": slope_xy(d.n_points, d.observed_k),
               "generated_slope": slope_xy(d.n_points, d.predicted_k),
               "slope_difference": slope_xy(d.n_points, d.observed_k) - slope_xy(d.n_points, d.predicted_k),
               "k_bias": float(e.mean()), "k_mae": float(e.abs().mean()),
               "k_rmse": float(np.sqrt(np.mean(e ** 2))),
               "raw_curve_r": float(sps.pearsonr(d.observed_k, d.predicted_k).statistic),
               "detrended_curve_r": residual_correlation(d.observed_k, d.predicted_k, d.n_points)}
        row.update(stimulus_bootstrap(d, n_boot, seed + 100 + ci)); summaries.append(row)
    return pd.DataFrame(fold_rows), predictions, pd.DataFrame(summaries), kcurve


def in_sample_kcurve(states, n_walks, seed):
    clf, scaler, z = fit_hazard(states); mean_state = mean_state_table(z); rows = []
    for ci, cond in enumerate(CONDS):
        mat = walk_k_matrix(clf, scaler, mean_state, COV, cond, NBIN_MIDS,
                            n_walks=n_walks, seed=seed + ci)
        for bi, nmid in enumerate(NBIN_MIDS):
            rows.append({"split": "in_sample", "repeat": 0, "fold": 0, "condition": cond,
                         "n_mid": nmid, "generated_k": float(mat[bi].mean()), "observed_k": np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20); ap.add_argument("--n-walks", type=int, default=4000)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--equivalence-margin", type=float, default=.01)
    ap.add_argument("--seed", type=int, default=20260823); args = ap.parse_args()
    states = assign_nbin(pd.read_csv(STATES))
    pfold, pdetail, pk, psum, pkc = participant_level(states, args.folds, args.repeats,
        args.n_walks, args.n_bootstrap, args.equivalence_margin, args.seed)
    sfold, sdetail, ssum, skc = stimulus_level(states, args.folds, args.repeats,
        args.n_walks, args.n_bootstrap, args.seed)
    outputs = {"process_oos_participant.csv": psum,
               "process_oos_participant_folds.csv": pfold,
               "process_oos_participant_predictions.csv": pdetail,
               "process_oos_participant_k_errors.csv": pk,
               "process_oos_stimulus.csv": ssum,
               "process_oos_stimulus_folds.csv": sfold,
               "process_oos_stimulus_predictions.csv": sdetail,
               "process_oos_kcurve.csv": pd.concat([pd.DataFrame(pkc), pd.DataFrame(skc),
                   pd.DataFrame(in_sample_kcurve(states, args.n_walks, args.seed))], ignore_index=True)}
    for name, frame in outputs.items(): frame.to_csv(OUT_DIR / name, index=False)
    pd.set_option("display.width", 260)
    print("===== repeated participant-grouped OOS ====="); print(psum.round(4).to_string(index=False))
    print("\n===== repeated stimulus-grouped OOS ====="); print(ssum.round(4).to_string(index=False))
    print(f"\nEquivalence margin: +/-{args.equivalence_margin:.3f} K/point")


if __name__ == "__main__": main()
