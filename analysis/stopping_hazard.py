"""Out-of-fold tests of process, quality, and aspiration-threshold stopping."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_STATES = OUT_DIR / "anchor_states.csv"


PROCESS = ["event_index", "log_elapsed", "k", "revisions_so_far", "is_delete", "n_points", "is_exp4"]


def design(frame: pd.DataFrame, model: str, tau: float | None = None) -> np.ndarray:
    x = frame[PROCESS].to_numpy(float)
    if model == "process":
        return x
    q = frame["silhouette"].to_numpy(float)
    delta = frame["delta_silhouette"].fillna(0).to_numpy(float)
    if model == "linear_quality":
        return np.column_stack([x, q, delta])
    if tau is None:
        raise ValueError("threshold model requires tau")
    # A segmented hazard: separate below-threshold shortfall from the jump and
    # slope above the aspiration level. Tau is estimated only on training data.
    return np.column_stack([x, np.minimum(q - tau, 0), (q >= tau).astype(float),
                            np.maximum(q - tau, 0), delta])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str, tau=None):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
    clf.fit(design(train, model, tau), train["stop"].to_numpy(int))
    return clf, clf.predict_proba(design(test, model, tau))[:, 1]


def select_tau(train: pd.DataFrame) -> float:
    q = train["silhouette"].dropna().to_numpy(float)
    candidates = np.unique(np.quantile(q, np.linspace(0.15, 0.85, 15)))
    best_tau, best_loss = float(np.median(q)), np.inf
    # Selection uses the outer training data only; all reported performance is
    # on stimulus-disjoint outer folds.
    for tau in candidates:
        clf, pred = fit_predict(train, train, "threshold", float(tau))
        score = log_loss(train["stop"], pred)
        if score < best_loss:
            best_tau, best_loss = float(tau), float(score)
    return best_tau


def bootstrap_logloss_difference(pred: pd.DataFrame, a: str, b: str, n_boot=2000, seed=20260814):
    """Paired stimulus bootstrap of log-loss(a) - log-loss(b)."""
    eps = 1e-12
    y = pred["stop"].to_numpy(float)
    loss_a = -(y * np.log(np.clip(pred[a], eps, 1-eps)) + (1-y) * np.log(np.clip(1-pred[a], eps, 1-eps)))
    loss_b = -(y * np.log(np.clip(pred[b], eps, 1-eps)) + (1-y) * np.log(np.clip(1-pred[b], eps, 1-eps)))
    temp = pd.DataFrame({"base": pred["base_uuid"], "d": loss_a - loss_b}).groupby("base")["d"].mean()
    values = temp.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.mean(values)), *map(float, np.quantile(boot, [0.025, 0.975]))


def split_half_threshold_reliability(final: pd.DataFrame, exp: str, n_boot=2000, seed=20260814):
    z = final[final["exp"] == exp].copy()
    # Remove stimulus difficulty before estimating each person's aspiration
    # offset, so stability cannot be driven by an easy/hard stimulus split.
    z["q_resid"] = z["silhouette"] - z.groupby("base_uuid")["silhouette"].transform("mean")
    z["half"] = z["base_uuid"].map(lambda s: int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16) % 2)
    wide = z.groupby(["sid", "half"])["q_resid"].median().unstack()
    wide = wide.dropna()
    r = float(np.corrcoef(wide[0], wide[1])[0, 1])
    rng = np.random.default_rng(seed)
    arr = wide[[0, 1]].to_numpy(float)
    vals = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        vals.append(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    corrected = 2 * r / (1 + r) if r > -0.999 else np.nan
    return {"exp": exp, "n_participants": len(wide), "split_half_r": r,
            "ci_low": float(lo), "ci_high": float(hi), "spearman_brown": float(corrected)}


def confidence_analysis(final: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    z = final[(final["exp"] == "exp4") & final["confidence"].notna()].copy()
    model = smf.ols("confidence ~ silhouette + k + n_points + C(sid) + C(base_uuid)", data=z).fit()
    robust = model.get_robustcov_results(cov_type="cluster", groups=z["sid"])
    pos = list(model.params.index).index("silhouette")
    ci = robust.conf_int()[pos]
    return {"n_trials": len(z), "silhouette_beta": float(robust.params[pos]),
            "cluster_se": float(robust.bse[pos]), "p_value": float(robust.pvalues[pos]),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]), "r_squared": float(model.rsquared)}


def search_cost_adjustment(final: pd.DataFrame) -> dict:
    """Same response format, higher acquisition cost: Exp.4 funnel vs Exp.3 full."""
    import statsmodels.formula.api as smf
    model = smf.ols("silhouette ~ C(exp) + k + n_points + C(base_uuid)", data=final).fit()
    # person_hash links repeat participants across Exp.3/4, unlike experiment-
    # specific sid. Cluster on identity so the cost contrast does not assume
    # fully independent condition samples.
    group = final["person_hash"] if "person_hash" in final else final["sid"]
    robust = model.get_robustcov_results(cov_type="cluster", groups=group)
    name = "C(exp)[T.exp4]"; pos = list(model.params.index).index(name); ci = robust.conf_int()[pos]
    return {"contrast": "funnel_minus_full_within_anchor", "n_trials": len(final),
            "silhouette_difference": float(robust.params[pos]), "cluster_se": float(robust.bse[pos]),
            "p_value": float(robust.pvalues[pos]), "ci_low": float(ci[0]), "ci_high": float(ci[1])}


def run(states_path: Path, folds: int, seed: int) -> None:
    z = pd.read_csv(states_path)
    z["log_elapsed"] = np.log1p(z["elapsed_s"].clip(lower=0))
    z["is_delete"] = (z["action"] == "delete").astype(int)
    z["is_exp4"] = (z["exp"] == "exp4").astype(int)
    z = z.replace([np.inf, -np.inf], np.nan).dropna(subset=["silhouette", "compactness"])
    splitter = GroupKFold(n_splits=folds)
    out = z[["exp", "sid", "base_uuid", "stop"]].copy()
    for name in ("process", "linear_quality", "threshold"):
        out[name] = np.nan
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter.split(z, z["stop"], groups=z["base_uuid"]), 1):
        train, test = z.iloc[tr], z.iloc[te]
        tau = select_tau(train)
        for model in ("process", "linear_quality", "threshold"):
            _, prediction = fit_predict(train, test, model, tau if model == "threshold" else None)
            out.iloc[te, out.columns.get_loc(model)] = prediction
        fold_rows.append({"fold": fold, "tau": tau, "n_train": len(train), "n_test": len(test),
                          "n_test_stops": int(test["stop"].sum())})
        print(f"fold {fold}/{folds}: tau={tau:.3f}", flush=True)
    metric_rows = []
    for model in ("process", "linear_quality", "threshold"):
        metric_rows.append({"model": model, "log_loss": log_loss(out["stop"], out[model]),
                            "auc": roc_auc_score(out["stop"], out[model])})
    contrasts = []
    for a, b in (("linear_quality", "process"), ("threshold", "process"), ("threshold", "linear_quality")):
        estimate, lo, hi = bootstrap_logloss_difference(out, a, b, seed=seed)
        contrasts.append({"contrast": f"{a}_minus_{b}", "logloss_difference": estimate,
                          "ci_low": lo, "ci_high": hi})
    final = z[z["stop"] == 1]
    reliability = [split_half_threshold_reliability(final, exp, seed=seed) for exp in ("exp3", "exp4")]
    confidence = confidence_analysis(final)
    cost_adjustment = search_cost_adjustment(final)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "stopping_oof_predictions.csv": out,
        "stopping_model_metrics.csv": pd.DataFrame(metric_rows),
        "stopping_model_contrasts.csv": pd.DataFrame(contrasts),
        "stopping_fold_thresholds.csv": pd.DataFrame(fold_rows),
        "aspiration_split_half.csv": pd.DataFrame(reliability),
        "confidence_shared_quality.csv": pd.DataFrame([confidence]),
        "aspiration_cost_adjustment.csv": pd.DataFrame([cost_adjustment]),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT_DIR / filename, index=False)
        print(frame.to_string(index=False) if filename != "stopping_oof_predictions.csv" else f"wrote {len(frame)} OOF rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=DEFAULT_STATES)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    run(args.states, args.folds, args.seed)


if __name__ == "__main__":
    main()
