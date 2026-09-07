"""Tidy K(N) data for revised Fig 4 (panel d) and slope forest (panel e).

Outputs (into figures_v2/outputs/):
  kn_human.csv    per condition x base_uuid: human mean K (per-stimulus)
  kn_walk.csv     per condition x N-bin: stopping-walk mean K + 95% percentile CI
  kn_greedy.csv   per condition x base_uuid: greedy k_at_stop
  kn_modelh.csv   per condition x base_uuid: Model-H mean predicted K (OOF)
  kn_kauto.csv    per condition x base_uuid: kmeans_auto silhouette-optimal K
  slopes.csv      per model x condition: slope point + 95% CI (for the forest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "revision_analysis"))
from revision_analysis.process_generator import prep, COV, NBINS  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "revision_analysis"
NPZ = ROOT / "data" / "processed_trials.npz"
PKG = ROOT / "revision_analysis"
STATES = PKG / "outputs" / "anchor_states.csv"
OUT = HERE.parent / "assets"
CONDS = ("exp3", "exp4")
ALL = ("exp1", "exp2", "exp3", "exp4")


def binned_mean_ci(ks: np.ndarray) -> tuple[float, float, float]:
    q = np.percentile(ks, [2.5, 97.5])
    return float(ks.mean()), float(q[0]), float(q[1])


def human_kn() -> pd.DataFrame:
    z = np.load(NPZ, allow_pickle=True)
    exp = np.asarray([str(e) for e in z["exp"]])
    bu = np.asarray([str(u) for u in z["base_uuid"]])
    npts = np.asarray(z["n_points"], int)
    human = z["human"]
    rows = []
    for e in ALL:
        m = exp == e
        for u in np.unique(bu[m]):
            kk = []
            for lab in human[m & (bu == u)]:
                lab = np.asarray(lab, int)
                lab = lab[lab >= 0]
                kk.append(len(np.unique(lab)))
            n = int(np.median(npts[m & (bu == u)]))
            rows.append({"condition": e, "base_uuid": u, "n_points": n,
                         "k": float(np.mean(kk))})
    return pd.DataFrame(rows)


def walk_kn(n_walks: int = 5000, seed: int = 0) -> pd.DataFrame:
    z = prep(pd.read_csv(STATES))
    X = z[COV].to_numpy(float)
    y = z["stop"].to_numpy(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(sc.transform(X), y)
    z["nbin"] = z["n_points"].map({lo: f"{lo}-{hi}" for lo, hi in NBINS})
    for lo, hi in NBINS:
        z.loc[(z.n_points >= lo) & (z.n_points <= hi), "nbin"] = f"{lo}-{hi}"
    ms = (z[z.stop == 0].groupby(["exp", "k", "nbin"])
          [["log_elapsed", "revisions_so_far", "Q", "dQ"]].mean().reset_index())
    nmid = np.asarray([0.5 * (lo + hi) for lo, hi in NBINS], float)
    rows = []
    for cond in CONDS:
        rng = np.random.default_rng(seed)
        is4 = 1.0 if cond == "exp4" else 0.0
        for bi, (lo, hi) in enumerate(NBINS):
            nm = nmid[bi]
            ks = np.empty(n_walks)
            for w in range(n_walks):
                k = 2
                while k <= 16:
                    row = ms[(ms.exp == cond) & (ms.k == k) & (ms.nbin == f"{lo}-{hi}")]
                    if len(row):
                        m = row.iloc[0]
                        xv = np.array([k, m.log_elapsed, k, m.revisions_so_far, 0.0,
                                       nm, is4, m.Q, m.dQ], float)
                    else:
                        row2 = ms[(ms.exp == cond) & (ms.k == k)]
                        if len(row2):
                            m = row2[["log_elapsed", "revisions_so_far", "Q", "dQ"]].mean()
                            xv = np.array([k, m.log_elapsed, k, m.revisions_so_far, 0.0,
                                           nm, is4, m.Q, m.dQ], float)
                        else:
                            break
                    p = float(clf.predict_proba(sc.transform(xv.reshape(1, -1)))[0, 1])
                    if rng.random() < p:
                        break
                    k += 1
                ks[w] = min(k, 16)
            # CI = sampling distribution of the MEAN (bootstrap over walks)
            bmean = np.array([ks[rng.integers(0, n_walks, n_walks)].mean() for _ in range(1000)])
            rows.append({"condition": cond, "n_mid": nm, "k": float(ks.mean()),
                         "ci_lo": float(np.percentile(bmean, 2.5)),
                         "ci_hi": float(np.percentile(bmean, 97.5))})
    return pd.DataFrame(rows)


def greedy_kn() -> pd.DataFrame:
    r = pd.read_csv(PKG / "outputs" / "process_simulator_results.csv")
    return r[["condition", "base_uuid", "n_points", "k_at_stop"]].rename(columns={"k_at_stop": "k"})


def modelh_kn() -> pd.DataFrame:
    """Per-stimulus Model-H mean predicted K from OOF hard partitions."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "autodl" / "tmp"))
    import torch
    import config as C
    from data import load_npz, make_cv_split
    from evaluate import predict
    from gpu_data import build_tensors
    from model import AssignNet

    d = load_npz()
    exp = np.asarray([str(e) for e in d["exp"]])
    bu = np.asarray([str(u) for u in d["base_uuid"]])
    npts = np.asarray(d["n_points"], int)
    pred_k = {}
    for fold in range(5):
        ck = torch.load(C.OUTPUT.parent / "runs_revision_full" / f"HC_cv{fold}" / "best.pt",
                        map_location="cpu")
        a = ck.get("args", {})
        mode = a.get("condition_mode", "none")
        net = AssignNet(use_gru=not a.get("no_gru", False),
                        in_dim=6 if mode == "tokens" else 4,
                        pairing=a.get("pairing", "meanmax"),
                        use_ncount=bool(a.get("ncount", False))).to("cpu")
        net.load_state_dict(ck["model"])
        idx = make_cv_split(d, fold, 5, a.get("split_by", C.SPLIT_BY),
                            a.get("split_seed", C.SPLIT_SEED))["test"]
        T = build_tensors(d, idx, "human", a.get("scanpath", "full"), 0, "cpu", mode)
        for j, lab in zip(idx, predict(net, T)):
            lab = np.asarray(lab, int)
            pred_k[int(j)] = len(np.unique(lab[lab >= 0]))
    rows = []
    for j in range(len(exp)):
        rows.append({"condition": exp[j], "base_uuid": bu[j],
                     "n_points": int(npts[j]), "k": pred_k[j]})
    df = pd.DataFrame(rows)
    agg = df.groupby(["condition", "base_uuid"]).agg(
        n_points=("n_points", "first"), k=("k", "mean")).reset_index()
    return agg


def kauto_kn() -> pd.DataFrame:
    p = Path(__file__).resolve().parents[2] / "autodl" / "tmp" / "outputs" / "kauto_perstim.csv"
    if not p.exists():
        return pd.DataFrame(columns=["condition", "base_uuid", "n_points", "k"])
    return pd.read_csv(p)


def slopes(human: pd.DataFrame, walk: pd.DataFrame, greedy: pd.DataFrame,
           mh: pd.DataFrame, kauto: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # human: participant slopes
    z = np.load(NPZ, allow_pickle=True)
    exp = np.asarray([str(e) for e in z["exp"]])
    sid = np.asarray([str(s) for s in z["sid"]])
    npts = np.asarray(z["n_points"], float)
    hum = z["human"]
    for e in ("exp3", "exp4"):
        m = exp == e
        vals = []
        for s in np.unique(sid[m]):
            kk = np.asarray([len(np.unique(np.asarray(l, int)[np.asarray(l, int) >= 0]))
                             for l in hum[m & (sid == s)]], float)
            nn = npts[m & (sid == s)]
            if len(kk) >= 2:
                vals.append(np.polyfit(nn, kk, 1)[0])
        vals = np.asarray(vals, float)
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        rows.append({"model": "Human", "condition": e, "slope": float(vals.mean()),
                     "ci_lo": float(vals.mean() - 1.96 * se),
                     "ci_hi": float(vals.mean() + 1.96 * se), "n": len(vals)})
    # stopping walk: slope from the curve (n_mid), CI from bootstrap on the curve
    for e in CONDS:
        w = walk[walk.condition == e]
        nm = w.n_mid.to_numpy(float)
        kk = w.k.to_numpy(float)
        b = float(np.polyfit(nm, kk, 1)[0])
        rng = np.random.default_rng(0)
        slopes = []
        for _ in range(1000):
            ks = np.asarray([w.iloc[bi]["k"] + rng.normal(0, 1) * 0 for bi in range(len(w))])
            # sample k_at_stop within bin from its normal approx (mean=curve mean, sd=curve sd)
            sds = (w.ci_hi - w.ci_lo).to_numpy(float) / (2 * 1.96)
            ks = np.asarray([rng.normal(m, s) for m, s in zip(kk, sds)], float)
            slopes.append(np.polyfit(nm, ks, 1)[0])
        slopes = np.asarray(slopes, float)
        rows.append({"model": "Stopping walk", "condition": e, "slope": b,
                     "ci_lo": float(np.percentile(slopes, 2.5)),
                     "ci_hi": float(np.percentile(slopes, 97.5)), "n": len(w)})
    # greedy, Model-H: per-stimulus OLS
    for model, df in (("Greedy", greedy), ("Model-H", mh)):
        for e in CONDS:
            g = df[df.condition == e]
            if len(g) < 3:
                continue
            x = g.n_points.to_numpy(float)
            y = g.k.to_numpy(float)
            b, c = np.polyfit(x, y, 1)
            yhat = b * x + c
            se = float(np.sqrt(np.sum((y - yhat) ** 2) / (len(g) - 2)))
            sxx = float(np.sum((x - x.mean()) ** 2))
            rows.append({"model": model, "condition": e, "slope": float(b),
                         "ci_lo": float(b - 1.96 * se / np.sqrt(sxx)),
                         "ci_hi": float(b + 1.96 * se / np.sqrt(sxx)), "n": len(g)})
    for e in CONDS:
        g = kauto[kauto.condition == e]
        b = float(np.polyfit(g.n_points.to_numpy(float), g.k.to_numpy(float), 1)[0])
        rows.append({"model": "kmeans_auto", "condition": e, "slope": b,
                     "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    human = human_kn()
    walk = walk_kn()
    greedy = greedy_kn()
    mh = modelh_kn()
    kauto = kauto_kn()
    human.to_csv(OUT / "kn_human.csv", index=False)
    walk.to_csv(OUT / "kn_walk.csv", index=False)
    greedy.to_csv(OUT / "kn_greedy.csv", index=False)
    mh.to_csv(OUT / "kn_modelh.csv", index=False)
    kauto.to_csv(OUT / "kn_kauto.csv", index=False)
    sl = slopes(human, walk, greedy, mh, kauto)
    sl.to_csv(OUT / "kn_slopes.csv", index=False)
    pd.set_option("display.width", 200)
    print("walk K(N):")
    print(walk.round(3).to_string(index=False))
    print("\nslopes:")
    print(sl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
