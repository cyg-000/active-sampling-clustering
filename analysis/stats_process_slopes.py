"""BF + Cohen's d for the process-generator and greedy-builder slope claims.

Comparisons (all use participant-specific human slopes as the sampling unit,
the paper's standard inference unit):
  (a) hazard-walk generated slope  vs  human participant slopes (exp3, exp4)
  (b) greedy construction builder  vs  human participant slopes (exp3, exp4)
Model slopes are treated as fixed predictions, exactly as Model-H is treated
in neural_slope_inference.py; we additionally report a bootstrap CI of the
generated slope across walk resamples so the model's own uncertainty is visible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .inference import one_sample
from .process_generator import prep, COV, NBINS

NPZ = Path(__file__).resolve().parent.parent / "data" / "processed_trials.npz"
STATES = Path(__file__).resolve().parent / "outputs" / "anchor_states.csv"
OUT = Path(__file__).resolve().parent / "outputs" / "stats_process_slopes.csv"
CONDS = ("exp3", "exp4")


def participant_slopes() -> dict[str, np.ndarray]:
    z = np.load(NPZ, allow_pickle=True)
    exp = np.asarray([str(e) for e in z["exp"]])
    sid = np.asarray([str(s) for s in z["sid"]])
    npts = np.asarray(z["n_points"], float)
    human = z["human"]
    out = {}
    for e in CONDS:
        m = exp == e
        slopes = []
        for s in np.unique(sid[m]):
            kk = np.asarray([len(np.unique(np.asarray(lab, int))) for lab in human[m & (sid == s)]], float)
            nn = npts[m & (sid == s)]
            if len(kk) < 2:
                continue
            slopes.append(np.polyfit(nn, kk, 1)[0])
        out[e] = np.asarray(slopes, float)
    return out


def hazard_walk_slopes(n_walks: int = 5000, seed: int = 0) -> tuple[dict[str, float], dict[str, tuple]]:
    """Return (point slope per condition, (lo, hi) bootstrap CI).

    Each condition gets a dedicated RNG so the estimate is order-independent;
    n_walks is set high so the point estimate is stable across seeds (at 400
    walks the slope swings by +/-0.01 depending on the draw, which is why the
    earlier +0.034 headline was fragile).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z = prep(pd.read_csv(STATES))
    X = z[COV].to_numpy(float)
    y = z["stop"].to_numpy(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(sc.transform(X), y)

    z["nbin"] = z["n_points"].map({lo: f"{lo}-{hi}" for lo, hi in NBINS})
    for lo, hi in NBINS:
        z.loc[(z.n_points >= lo) & (z.n_points <= hi), "nbin"] = f"{lo}-{hi}"
    mean_state = (z[z.stop == 0].groupby(["exp", "k", "nbin"])
                  [["log_elapsed", "revisions_so_far", "Q", "dQ"]].mean().reset_index())

    # per-walk k_at_stop: [condition][nbin_idx][walk]
    walks = {e: np.zeros((len(NBINS), n_walks)) for e in CONDS}
    for cond in CONDS:
        rng = np.random.default_rng(seed)  # dedicated RNG per condition
        is_exp4 = 1.0 if cond == "exp4" else 0.0
        for bi, (lo, hi) in enumerate(NBINS):
            nmid = 0.5 * (lo + hi)
            for w in range(n_walks):
                k = 2
                while k <= 16:
                    ms = mean_state[(mean_state.exp == cond) & (mean_state.k == k)
                                    & (mean_state.nbin == f"{lo}-{hi}")]
                    if len(ms):
                        m = ms.iloc[0]
                        xv = np.array([k, m.log_elapsed, k, m.revisions_so_far, 0.0,
                                       nmid, is_exp4, m.Q, m.dQ], float)
                    else:
                        mm = mean_state[(mean_state.exp == cond) & (mean_state.k == k)]
                        if len(mm):
                            m = mm[["log_elapsed", "revisions_so_far", "Q", "dQ"]].mean()
                            xv = np.array([k, m.log_elapsed, k, m.revisions_so_far, 0.0,
                                           nmid, is_exp4, m.Q, m.dQ], float)
                        else:
                            break
                    p = float(clf.predict_proba(sc.transform(xv.reshape(1, -1)))[0, 1])
                    if rng.random() < p:
                        break
                    k += 1
                walks[cond][bi, w] = min(k, 16)

    nmid = np.asarray([0.5 * (lo + hi) for lo, hi in NBINS], float)
    point = {e: float(np.polyfit(nmid, walks[e].mean(1), 1)[0]) for e in CONDS}
    rng2 = np.random.default_rng(1)
    cis = {}
    for e in CONDS:
        slopes = []
        for _ in range(1000):
            ks = np.asarray([walks[e][bi, rng2.integers(0, n_walks, n_walks)].mean()
                             for bi in range(len(NBINS))], float)
            slopes.append(np.polyfit(nmid, ks, 1)[0])
        cis[e] = (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
    return point, cis


def one_sample_against(human: np.ndarray, model: float) -> dict:
    diff = human - model
    out = one_sample(diff, 0.0)
    bf10 = float(out["bf10"])
    return {
        "mean_diff": float(diff.mean()), "sd": float(diff.std(ddof=1)),
        "t": float(out["t"]), "df": int(out["df"]),
        "p": float(out["p_two_sided"]),
        "dz": float(out["cohen_dz"]),
        "BF10": bf10, "BF01": 1.0 / bf10,
    }


def main() -> None:
    human = participant_slopes()
    gen_point, gen_ci = hazard_walk_slopes()
    print("hazard-walk generated slopes:", {k: round(v, 4) for k, v in gen_point.items()},
          "\nbootstrap CI:", {k: (round(v[0], 4), round(v[1], 4)) for k, v in gen_ci.items()})

    greedy = {"exp3": -0.008928571428571423, "exp4": -0.008928571428571428}  # aspiration x1.0
    rows = []
    for e in CONDS:
        h = human[e]
        print(f"\n{ e } human participant slopes: n={len(h)} mean={h.mean():.4f} SD={h.std(ddof=1):.4f}")
        for label, model in (("hazard_walk", gen_point[e]), ("greedy", greedy[e])):
            r = one_sample_against(h, model)
            r.update({"condition": e, "comparison": label, "model_slope": model,
                      "human_mean_slope": float(h.mean()),
                      "gen_slope_ci_lo": gen_ci[e][0] if label == "hazard_walk" else "",
                      "gen_slope_ci_hi": gen_ci[e][1] if label == "hazard_walk" else ""})
            rows.append(r)
            print(f"  {label:10s} model={model:+.4f}  human-model={r['mean_diff']:+.4f}  "
                  f"t={r['t']:.2f}  dz={r['dz']:.2f}  BF10={r['BF10']:.3g}")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
