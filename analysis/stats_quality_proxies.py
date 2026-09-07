"""Cluster-robust convergence checks for geometric quality proxies.

Silhouette and compactness are alternative geometric summaries of the same
partition, not independent psychological measurements. This module therefore
reports their raw state-level association only descriptively and bases
uncertainty on participant- and stimulus-cluster summaries. It also refits all
stopping models on exactly the same stimulus-disjoint folds, separating current
quality from change in quality.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).resolve().parent / "outputs"
STATES = OUT_DIR / "anchor_states.csv"
PROCESS = ["event_index", "log_elapsed", "k", "revisions_so_far",
           "is_delete", "n_points", "is_exp4"]


def prepare() -> pd.DataFrame:
    z = pd.read_csv(STATES)
    z["log_elapsed"] = np.log1p(z.elapsed_s.clip(lower=0))
    z["is_delete"] = (z.action == "delete").astype(int)
    z["is_exp4"] = (z.exp == "exp4").astype(int)
    z["delta_compactness"] = z.groupby(["sid", "base_uuid", "flipped"])["compactness"].diff()
    return z.replace([np.inf, -np.inf], np.nan).dropna(subset=["silhouette", "compactness"]).reset_index(drop=True)


def design(z: pd.DataFrame, model: str) -> np.ndarray:
    x = z[PROCESS].to_numpy(float)
    if model == "process": return x
    if model == "silhouette_current": return np.column_stack([x, z.silhouette])
    if model == "silhouette_full":
        return np.column_stack([x, z.silhouette, z.delta_silhouette.fillna(0)])
    if model == "compactness_current": return np.column_stack([x, z.compactness])
    if model == "compactness_full":
        return np.column_stack([x, z.compactness, z.delta_compactness.fillna(0)])
    raise ValueError(model)


def weighted_partial_r(df: pd.DataFrame, controls: list[str]) -> float:
    if len(df) < 8: return np.nan
    c = np.column_stack([np.ones(len(df)), df[controls].to_numpy(float)])
    x = df.silhouette.to_numpy(float); y = df.compactness.to_numpy(float)
    rx = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
    ry = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(sps.pearsonr(rx, ry).statistic)


def cluster_correlations(z: pd.DataFrame, cluster: str, controls: list[str],
                         n_boot: int, seed: int) -> dict:
    vals = []
    for _, g in z.groupby(cluster):
        r = weighted_partial_r(g, controls)
        if np.isfinite(r) and abs(r) < 1: vals.append(r)
    vals = np.asarray(vals, float); fisher = np.arctanh(vals)
    estimate = float(np.tanh(fisher.mean()))
    rng = np.random.default_rng(seed)
    boot = np.tanh(np.mean(rng.choice(fisher, size=(n_boot, len(fisher)), replace=True), axis=1))
    return {"estimate": estimate, "ci_lo": float(np.quantile(boot, .025)),
            "ci_hi": float(np.quantile(boot, .975)), "n_clusters": len(vals)}


def fit_oof(z: pd.DataFrame, folds: int) -> pd.DataFrame:
    models = ("process", "silhouette_current", "silhouette_full",
              "compactness_current", "compactness_full")
    out = z[["person_hash", "base_uuid", "exp", "stop"]].copy()
    for model in models: out[model] = np.nan
    splitter = GroupKFold(n_splits=folds)
    for tr, te in splitter.split(z, z.stop, groups=z.base_uuid):
        train, test = z.iloc[tr], z.iloc[te]
        for model in models:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
            clf.fit(design(train, model), train.stop.to_numpy(int))
            out.loc[out.index[te], model] = clf.predict_proba(design(test, model))[:, 1]
    return out


def row_logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12); y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered_loss_contrast(pred: pd.DataFrame, model: str, cluster: str,
                            n_boot: int, seed: int) -> dict:
    d = row_logloss(pred.stop, pred[model]) - row_logloss(pred.stop, pred.process)
    per_cluster = pd.DataFrame({"cluster": pred[cluster].astype(str), "d": d}).groupby("cluster").d.mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(per_cluster, size=(n_boot, len(per_cluster)), replace=True), axis=1)
    return {"estimate": float(per_cluster.mean()), "ci_lo": float(np.quantile(boot, .025)),
            "ci_hi": float(np.quantile(boot, .975)), "n_clusters": len(per_cluster)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260823); args = ap.parse_args()
    z = prepare(); rows = []
    raw = float(sps.pearsonr(z.silhouette, z.compactness).statistic)
    rows.append({"analysis": "raw_state_level_descriptive", "cluster": "none",
                 "controls": "none", "estimate": raw, "n_states": len(z)})
    for i, (cluster, controls) in enumerate((("person_hash", ["k", "n_points"]),
                                             ("base_uuid", ["k"]))):
        r = cluster_correlations(z, cluster, controls, args.n_bootstrap, args.seed + i)
        rows.append({"analysis": "within_cluster_partial_correlation", "cluster": cluster,
                     "controls": "+".join(controls), "estimate": r["estimate"],
                     "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "n_clusters": r["n_clusters"], "n_states": len(z)})
    correlations = pd.DataFrame(rows)

    pred = fit_oof(z, args.folds)
    models = ("process", "silhouette_current", "silhouette_full",
              "compactness_current", "compactness_full")
    metrics = pd.DataFrame([{"model": m, "log_loss": log_loss(pred.stop, pred[m]),
                             "auc": roc_auc_score(pred.stop, pred[m])} for m in models])
    contrasts = []
    for mi, model in enumerate(models[1:]):
        for ci, cluster in enumerate(("person_hash", "base_uuid")):
            r = clustered_loss_contrast(pred, model, cluster, args.n_bootstrap,
                                        args.seed + 100 + mi * 10 + ci)
            contrasts.append({"contrast": f"{model}_minus_process", "cluster": cluster,
                              "logloss_difference": r["estimate"], "ci_lo": r["ci_lo"],
                              "ci_hi": r["ci_hi"], "n_clusters": r["n_clusters"]})
    contrasts = pd.DataFrame(contrasts)
    correlations.to_csv(OUT_DIR / "quality_proxy_cluster_correlations.csv", index=False)
    metrics.to_csv(OUT_DIR / "quality_proxy_oof_metrics.csv", index=False)
    contrasts.to_csv(OUT_DIR / "quality_proxy_oof_cluster_contrasts.csv", index=False)
    pred.to_csv(OUT_DIR / "quality_proxy_oof_predictions.csv", index=False)
    print("===== cluster-aware silhouette x compactness convergence =====")
    print(correlations.round(4).to_string(index=False))
    print("\n===== same-fold OOF stopping metrics ====="); print(metrics.round(5).to_string(index=False))
    print("\n===== clustered OOF log-loss differences (negative favors quality) =====")
    print(contrasts.round(5).to_string(index=False))


if __name__ == "__main__": main()
