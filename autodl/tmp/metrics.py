"""
Evaluation metrics.
Two levels: (1) goodness-of-fit -- FM and ARI (primary; FM rewards merging)
vs human on held-out stimuli; (2) emergent properties (never in loss) --
cluster count, silhouette, convex-hull%, two invariant slopes.
HUMAN_REF provides measured human baselines on the test set.
"""
import numpy as np


def _contingency(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = (a >= 0) & (b >= 0)
    if m.sum() < 4:
        return None
    a, b = a[m], b[m]
    ua, ub = np.unique(a), np.unique(b)
    n = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            n[i, j] = np.sum((a == x) & (b == y))
    return n


def fowlkes_mallows(a, b):
    n = _contingency(a, b)
    if n is None:
        return np.nan
    c2 = lambda z: (z * (z - 1) / 2).sum()
    tp, tf, tn = c2(n), c2(n.sum(1)), c2(n.sum(0))
    return np.nan if tf <= 0 or tn <= 0 else tp / np.sqrt(tf * tn)


def adjusted_rand(a, b):
    """
    Evaluation metrics.
    Two levels: (1) goodness-of-fit -- FM and ARI (primary; FM rewards merging)
    vs human on held-out stimuli; (2) emergent properties (never in loss) --
    cluster count, silhouette, convex-hull%, two invariant slopes.
    HUMAN_REF provides measured human baselines on the test set.
    """
    n = _contingency(a, b)
    if n is None:
        return np.nan
    c2 = lambda z: (z * (z - 1) / 2).sum()
    sij, si, sj = c2(n), c2(n.sum(1)), c2(n.sum(0))
    N = n.sum()
    exp = si * sj / max(c2(np.array([N])), 1e-9)
    mx = (si + sj) / 2
    return np.nan if abs(mx - exp) < 1e-12 else (sij - exp) / (mx - exp)


def silhouette(pts, lab):
    m = lab >= 0
    p, l = pts[m], lab[m]
    if len(np.unique(l)) < 2 or len(p) < 3:
        return np.nan
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    out = []
    for i in range(len(p)):
        same = (l == l[i]); same[i] = False
        if same.sum() == 0:
            out.append(0.0); continue
        a = d[i, same].mean()
        b = min(d[i, l == c].mean() for c in np.unique(l) if c != l[i])
        out.append((b - a) / max(a, b))
    return float(np.mean(out))


def hull_pct(pts, lab):
    """
    Evaluation metrics.
    Two levels: (1) goodness-of-fit -- FM and ARI (primary; FM rewards merging)
    vs human on held-out stimuli; (2) emergent properties (never in loss) --
    cluster count, silhouette, convex-hull%, two invariant slopes.
    HUMAN_REF provides measured human baselines on the test set.
    """
    from scipy.spatial import ConvexHull
    out = []
    for c in np.unique(lab[lab >= 0]):
        P = pts[lab == c]
        U = np.unique(P, axis=0)
        if len(U) < 3:
            out.append(1.0)
            continue
        try:
            out.append(len(ConvexHull(U).vertices) / len(P))
        except Exception:
            out.append(1.0)
    return float(np.mean(out)) if out else np.nan


def trial_stats(pts, lab):
    """Evaluation metrics.
Two levels: (1) goodness-of-fit -- FM and ARI (primary; FM rewards merging)
vs human on held-out stimuli; (2) emergent properties (never in loss) --
cluster count, silhouette, convex-hull%, two invariant slopes.
HUMAN_REF provides measured human baselines on the test set."""
    ok = lab >= 0
    cl = np.unique(lab[ok])
    if len(cl) < 1:
        return None
    nums = [int((lab == c).sum()) for c in cl]
    cen = np.array([pts[lab == c].mean(0) for c in cl])
    g = pts.mean(0)
    return dict(k=len(cl), numerosity=float(np.mean(nums)),
                sil=silhouette(pts, lab), ch=hull_pct(pts, lab),
                radial=float(np.mean(np.hypot(*(cen - g).T))),
                unassigned=float((~ok).mean()))


def slope(x, y):
    """Evaluation metrics.
Two levels: (1) goodness-of-fit -- FM and ARI (primary; FM rewards merging)
vs human on held-out stimuli; (2) emergent properties (never in loss) --
cluster count, silhouette, convex-hull%, two invariant slopes.
HUMAN_REF provides measured human baselines on the test set."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    x, y = x[m], y[m]
    return float(np.polyfit(x, y, 1)[0])


def summarize(pts_list, lab_list, n_list, human_list=None):
    rows = [trial_stats(p, l) for p, l in zip(pts_list, lab_list)]
    keep = [i for i, r in enumerate(rows) if r is not None]
    rows = [rows[i] for i in keep]
    n = [n_list[i] for i in keep]
    def med(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")
    out = {f"{k}_median": med(k) for k in ("k", "numerosity", "sil", "ch", "radial")}
    out["k_mean"] = float(np.nanmean([r["k"] for r in rows]))
    out["slope_k_vs_n"] = slope(n, [r["k"] for r in rows])
    out["slope_num_vs_n"] = slope(n, [r["numerosity"] for r in rows])
    out["unassigned"] = float(np.nanmean([r["unassigned"] for r in rows]))
    if human_list is not None:
        fm = [fowlkes_mallows(human_list[i], lab_list[i]) for i in keep]
        ar = [adjusted_rand(human_list[i], lab_list[i]) for i in keep]
        out["ARI_vs_human"] = float(np.nanmean(ar))
        out["FM_vs_human"] = float(np.nanmean(fm))
        out["FM_sd"] = float(np.nanstd(fm))
    out["n_trials"] = len(rows)
    return out


#              k_mean   sil     ch%    slope_k  slope_num
#   exp1 lasso   4.54   0.375   0.855   0.088    0.242
#   exp2 lasso   4.52   0.341   0.887   0.097    0.149
#   exp3 voronoi 3.17   0.376   0.750   0.036    0.317
#   exp4 voronoi 3.32   0.352   0.769   0.027    0.283
HUMAN_REF = {
    "exp1": dict(k_mean=4.54, sil=0.375, ch=0.855, slope_k=0.088, slope_num=0.242),
    "voronoi": dict(k_mean=3.2, sil=0.365, ch=0.76, slope_k=0.03, slope_num=0.30),
    "M&V_exp1": dict(ch=0.89, sil=0.35, slope_k=0.13, slope_num=0.12),
}
