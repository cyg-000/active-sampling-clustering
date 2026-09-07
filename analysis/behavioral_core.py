"""Clean-sample reliability and condition-specific behavioral scaling.

This module supplies the population-level behavioral results that are distinct
from the stimulus-equal model targets in :mod:`condition_targets`. It writes
only aggregate, identity-free tables suitable for a public repository.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score

from .common import DEFAULT_DATASET, EXPERIMENTS, load_dataset, ols_slope
from .inference import independent, one_sample


OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _ci(values, seed=20260815, n_boot=5000):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    b = np.mean(rng.choice(x, (n_boot, len(x)), replace=True), axis=1)
    return np.quantile(b, [.025, .975])


def _cluster_count(labels):
    x = np.asarray(labels, int)
    return len(np.unique(x[x >= 0]))


def _agreement(a, b):
    """Partition agreement on points assigned by both responses."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    ok = (a >= 0) & (b >= 0)
    if ok.sum() < 2:
        return np.nan, np.nan
    return fowlkes_mallows_score(a[ok], b[ok]), adjusted_rand_score(a[ok], b[ok])


def _frame(d):
    rows = []
    for j in range(len(d["exp"])):
        k = _cluster_count(d["human"][j])
        rows.append({
            "row": j, "exp": str(d["exp"][j]), "sid": str(d["sid"][j]),
            "base_uuid": str(d["base_uuid"][j]), "presentation": int(d["presentation"][j]),
            "flipped": bool(d["flipped"][j]), "n_points": int(d["n_points"][j]),
            "k": float(k), "mean_size": float(d["n_points"][j]) / k,
        })
    return pd.DataFrame(rows)


def reliability(d, frame, seed=20260815):
    internal = []
    for (exp, sid, base), g in frame.groupby(["exp", "sid", "base_uuid"]):
        if len(g) < 2:
            continue
        # The two presentations preserve point identity even when mirrored.
        a, b = g.sort_values("presentation").iloc[:2].row.astype(int)
        fm, ari = _agreement(d["human"][a], d["human"][b])
        internal.append({"exp": exp, "sid": sid, "base_uuid": base,
                         "FM": fm, "ARI": ari})
    internal = pd.DataFrame(internal)
    internal_p = internal.groupby(["exp", "sid"], as_index=False)[["FM", "ARI"]].mean()

    # One response per participant and base stimulus prevents repeated trials
    # from inflating between-person agreement.
    first = frame.sort_values("presentation").drop_duplicates(["exp", "sid", "base_uuid"])
    between_records = []
    for (exp, base), g in first.groupby(["exp", "base_uuid"]):
        vals = []
        for ia, ib in combinations(g.row.astype(int), 2):
            fm, ari = _agreement(d["human"][ia], d["human"][ib])
            vals.append((str(d["sid"][ia]), str(d["sid"][ib]), fm, ari))
        for sa, sb, fm, ari in vals:
            between_records.append((exp, base, sa, fm, ari))
            between_records.append((exp, base, sb, fm, ari))
    between = pd.DataFrame(between_records, columns=["exp", "base_uuid", "sid", "FM", "ARI"])
    between_p = between.groupby(["exp", "sid"], as_index=False)[["FM", "ARI"]].mean()

    summary = []
    contrasts = []
    for offset, exp in enumerate(EXPERIMENTS):
        for kind, x in (("internal", internal_p.query("exp == @exp")),
                        ("interparticipant", between_p.query("exp == @exp"))):
            for metric in ("FM", "ARI"):
                lo, hi = _ci(x[metric], seed + offset + (metric == "ARI") * 20)
                summary.append({"experiment": exp, "agreement": kind, "metric": metric,
                                "n_participants": len(x), "mean": x[metric].mean(),
                                "ci_low": lo, "ci_high": hi})
        paired = internal_p.query("exp == @exp").merge(
            between_p.query("exp == @exp"), on=["exp", "sid"], suffixes=("_internal", "_between"))
        for metric in ("FM", "ARI"):
            test = one_sample(paired[f"{metric}_internal"] - paired[f"{metric}_between"])
            contrasts.append({"experiment": exp, "metric": metric,
                              "estimand": "internal minus interparticipant agreement",
                              "sampling_unit": "participant", **test})

    # Stability of participant-level granularity across the two presentations.
    stability = []
    for exp in EXPERIMENTS:
        x = frame.query("exp == @exp")
        for outcome in ("k", "mean_size"):
            wide = x.groupby(["sid", "presentation"])[outcome].median().unstack()
            wide = wide.dropna()
            if len(wide) < 4 or wide.shape[1] < 2:
                continue
            r, p = stats.pearsonr(wide.iloc[:, 0], wide.iloc[:, 1])
            z = np.arctanh(np.clip(r, -.999999, .999999)); se = 1 / np.sqrt(len(wide) - 3)
            lo, hi = np.tanh(z + np.array([-1, 1]) * 1.96 * se)
            stability.append({"experiment": exp, "outcome": outcome,
                              "sampling_unit": "participant", "n": len(wide),
                              "pearson_r": r, "p_two_sided": p, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(summary), pd.DataFrame(contrasts), pd.DataFrame(stability)


def _mixed_slope(x, outcome):
    x = x.copy()
    x["n_center"] = x.n_points - x.n_points.mean()
    model = smf.mixedlm(f"{outcome} ~ n_center", x, groups=x["sid"],
                        re_formula="1", vc_formula={"stimulus": "0 + C(base_uuid)"})
    fit = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    if not fit.converged:
        raise RuntimeError("MixedLM did not converge")
    beta = float(fit.params["n_center"]); se = float(fit.bse["n_center"])
    return beta, se, float(fit.pvalues["n_center"]), beta - 1.96 * se, beta + 1.96 * se
    

def scaling(frame):
    mixed_rows, participant_rows = [], []
    for exp in EXPERIMENTS:
        x = frame.query("exp == @exp")
        for outcome in ("k", "mean_size"):
            try:
                beta, se, p, lo, hi = _mixed_slope(x, outcome)
                status = "converged"
            except Exception as exc:
                # A participant/stimulus fixed-effects model is a transparent
                # fallback if the crossed random-effects optimizer is singular.
                fit = smf.ols(f"{outcome} ~ n_points + C(sid) + C(base_uuid)", x).fit(
                    cov_type="cluster", cov_kwds={"groups": x["sid"]})
                beta = float(fit.params["n_points"]); se = float(fit.bse["n_points"])
                p = float(fit.pvalues["n_points"]); lo, hi = beta - 1.96 * se, beta + 1.96 * se
                status = f"participant/stimulus FE fallback: {type(exc).__name__}"
            slopes = x.groupby("sid").apply(
                lambda g: ols_slope(g.n_points, g[outcome]), include_groups=False).dropna()
            bf = one_sample(slopes)
            mixed_rows.append({"experiment": exp, "outcome": outcome,
                               "estimator": "crossed mixed model", "n_trials": len(x),
                               "n_participants": x.sid.nunique(), "n_stimuli": x.base_uuid.nunique(),
                               "slope": beta, "se": se, "ci_low": lo, "ci_high": hi,
                               "p_two_sided": p, "participant_slope_bf10": bf["bf10"],
                               "participant_slope_bf01": bf["bf01"], "fit_status": status})
            participant_rows.extend({"experiment": exp, "sid_hash": hashlib.sha256(s.encode()).hexdigest()[:16],
                                     "outcome": outcome, "slope": v}
                                    for s, v in slopes.items())
    return pd.DataFrame(mixed_rows), pd.DataFrame(participant_rows)


def scaling_contrasts(participant_slopes):
    pairs = [("exp2", "exp1", "funnel minus full within lasso"),
             ("exp4", "exp3", "funnel minus full within anchor"),
             ("exp3", "exp1", "anchor minus lasso under full view"),
             ("exp4", "exp2", "anchor minus lasso under funnel view")]
    rows = []
    for outcome in ("k", "mean_size"):
        z = participant_slopes.query("outcome == @outcome")
        for a, b, name in pairs:
            xa = z.query("experiment == @a").slope
            xb = z.query("experiment == @b").slope
            test = independent(xa, xb)
            rows.append({"outcome": outcome, "contrast": name,
                         "group1": a, "group2": b,
                         "sampling_unit": "participant slope", **test})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = load_dataset(args.dataset); frame = _frame(d)
    rel, rel_con, stability = reliability(d, frame)
    slopes, participant_slopes = scaling(frame)
    contrasts = scaling_contrasts(participant_slopes)
    outputs = {"behavioral_reliability.csv": rel,
               "behavioral_reliability_contrasts.csv": rel_con,
               "granularity_stability.csv": stability,
               "behavioral_population_slopes.csv": slopes,
               "behavioral_slope_contrasts.csv": contrasts}
    for name, table in outputs.items():
        table.to_csv(args.out_dir / name, index=False)
        print(f"wrote {name}: {len(table)} rows")


if __name__ == "__main__":
    main()
