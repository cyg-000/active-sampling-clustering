"""
Frontier analysis: re-frame "no sweet spot" as positive difference test.
Per seed: interpolate lambda* where slope_k equals human value, then test
slope_num gap against zero + pre-registered equivalence margin.
"""
import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np

try:
    from scipy import stats, integrate
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

HERE = Path(__file__).parent

DELTA_ARI = 0.02
DELTA_NUM = 0.05
HUMAN_ROW = "human (ref)"
TAG_RE = re.compile(r"^CAP([0-9.]+)_s(\d+)")


def read_comparison(path):
    """-> (human dict, {seed: [(lambda, slope_k, slope_num, ARI), ...]})"""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    def col(name):
        return head.index(name)
    ci_sk, ci_num, ci_ari = col("slope_k_vs_n"), col("slope_num_vs_n"), col("ARI_vs_human")
    human = None
    per_seed = {}
    for r in rows[1:]:
        if not r:
            continue
        name = r[0].strip()
        if name.startswith(HUMAN_ROW):
            human = dict(slope_k=float(r[ci_sk]), slope_num=float(r[ci_num]))
            continue
        m = TAG_RE.match(name)
        if not m:
            continue
        lam, seed = float(m.group(1)), int(m.group(2))
        try:
            rec = (lam, float(r[ci_sk]), float(r[ci_num]), float(r[ci_ari]))
        except ValueError:
            continue
        per_seed.setdefault(seed, []).append(rec)
    return human, per_seed


def interp_at_slope(curve, target_sk):
    """
    Frontier analysis: re-frame "no sweet spot" as positive difference test.
    Per seed: interpolate lambda* where slope_k equals human value, then test
    slope_num gap against zero + pre-registered equivalence margin.
    """
    c = sorted(curve, key=lambda t: t[0])
    for i in range(len(c) - 1):
        lo, hi = c[i], c[i + 1]
        sk_lo, sk_hi = lo[1], hi[1]
        if (sk_lo <= target_sk <= sk_hi) and sk_hi > sk_lo:
            frac = (target_sk - sk_lo) / (sk_hi - sk_lo)
            lam_star = lo[0] + frac * (hi[0] - lo[0])
            num_star = lo[2] + frac * (hi[2] - lo[2])
            ari_star = lo[3] + frac * (hi[3] - lo[3])
            return lam_star, num_star, ari_star
    return None


def baseline_ari(curve):
    for lam, _, _, ari in curve:
        if abs(lam) < 1e-9:
            return ari
    return None


def jzs_bf10_onesample(t, n, r=0.7071067811865476):
    """
    Frontier analysis: re-frame "no sweet spot" as positive difference test.
    Per seed: interpolate lambda* where slope_k equals human value, then test
    slope_num gap against zero + pre-registered equivalence margin.
    """
    if not HAVE_SCIPY:
        return float("nan")
    nu = n - 1

    def integrand(g):
        num = (1.0 + n * g) ** (-0.5) * \
              (1.0 + t * t / ((1.0 + n * g) * nu)) ** (-(nu + 1) / 2.0)
        prior = (r ** 2) ** 0.5 / (math.sqrt(2 * math.pi) * g ** 1.5) * \
                math.exp(-(r ** 2) / (2.0 * g))
        return num * prior

    m1, _ = integrate.quad(integrand, 1e-8, np.inf, limit=200)
    m0 = (1.0 + t * t / nu) ** (-(nu + 1) / 2.0)
    return m1 / m0


def one_sided_test(diffs, margin, direction="greater"):
    """
    Frontier analysis: re-frame "no sweet spot" as positive difference test.
    Per seed: interpolate lambda* where slope_k equals human value, then test
    slope_num gap against zero + pre-registered equivalence margin.
    """
    x = np.asarray([d for d in diffs if np.isfinite(d)], float)
    n = len(x)
    out = dict(n=n, mean=float(np.mean(x)) if n else float("nan"))
    if n < 2:
        return out
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    out["sd"] = sd; out["se"] = se
    if se == 0:
        se = 1e-12
    t = out["mean"] / se
    out["t"] = t
    if HAVE_SCIPY:
        tcrit = stats.t.ppf(0.975, n - 1)
        out["ci_lo"] = out["mean"] - tcrit * se
        out["ci_hi"] = out["mean"] + tcrit * se
        out["p_one_sided"] = float(stats.t.sf(t, n - 1)) if direction == "greater" \
            else float(stats.t.cdf(t, n - 1))
        out["bf10"] = jzs_bf10_onesample(t, n)
    else:
        out["ci_lo"] = out["mean"] - 2.78 * se   # t_.975,df4 ≈ 2.78 (n=5)
        out["ci_hi"] = out["mean"] + 2.78 * se
        out["p_one_sided"] = float("nan"); out["bf10"] = float("nan")
    out["exceeds_margin"] = out["ci_lo"] > margin
    out["equiv_to_zero"] = out["ci_hi"] < margin
    return out


def fmt(v, p=4):
    return f"{v:.{p}f}" if isinstance(v, float) and np.isfinite(v) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "runs" / "comparison.csv"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    human, per_seed = read_comparison(a.csv)
    if human is None:
        raise SystemExit("Cannot find human reference row; cannot calibrate.")
    H_sk, H_num = human["slope_k"], human["slope_num"]

    star = []   # (seed, lam*, num*, ari*, base_ari)
    for seed, curve in sorted(per_seed.items()):
        res = interp_at_slope(curve, H_sk)
        base = baseline_ari(curve)
        if res is None:
            print(f"  [seed {seed}] slope_k did not cross human value {H_sk:.3f} -- skip")
            continue
        lam_s, num_s, ari_s = res
        star.append((seed, lam_s, num_s, ari_s, base))

    lam_vals = sorted({rec[0] for c in per_seed.values() for rec in c})
    summ = []
    for lam in lam_vals:
        sk = [rec[1] for c in per_seed.values() for rec in c if abs(rec[0] - lam) < 1e-9]
        nu = [rec[2] for c in per_seed.values() for rec in c if abs(rec[0] - lam) < 1e-9]
        ar = [rec[3] for c in per_seed.values() for rec in c if abs(rec[0] - lam) < 1e-9]
        def ms(v):
            v = np.asarray(v, float)
            return (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0)
        summ.append((lam, len(sk), *ms(sk), *ms(nu), *ms(ar)))
    with open(os.path.join(a.out, "frontier_summary.csv"), "w", encoding="utf-8") as f:
        f.write("lambda,n_seed,slope_k_mean,slope_k_sem,slope_num_mean,slope_num_sem,ARI_mean,ARI_sem\n")
        for row in summ:
            f.write(",".join(fmt(x, 5) for x in row) + "\n")

    if len(star) < 2:
        raise SystemExit("Not enough valid seeds for test (check whether slope_k covers human value).")

    d_num = [H_num - s[2] for s in star]
    d_ari = [s[4] - s[3] for s in star if s[4] is not None]
    lam_star = [s[1] for s in star]

    t_num = one_sided_test(d_num, DELTA_NUM)
    t_ari = one_sided_test(d_ari, DELTA_ARI)

    caseB = t_num.get("exceeds_margin", False)
    caseA = t_num.get("equiv_to_zero", False) and t_ari.get("equiv_to_zero", False)

    L = []
    L.append("=" * 72)
    L.append("Frontier analysis: lambda* = capacity penalty needed to reproduce human cluster-count slope")
    L.append("=" * 72)
    L.append(f"Human reference:  slope_k = {H_sk:.4f}   slope_num (cluster-size slope) = {H_num:.4f}")
    L.append(f"λ* (mean±sd across seeds) = {np.mean(lam_star):.4f} ± "
             f"{np.std(lam_star, ddof=1):.4f}   (n={len(star)} seeds)")
    base_mean = np.mean([s[4] for s in star if s[4] is not None])
    L.append(f"Fidelity baseline ARI (lambda=0, Model-H) = {base_mean:.4f}")
    L.append("")
    L.append("-- Primary endpoint: cluster-size slope gap  Delta_num = human - model(lambda*)  (>0 = model below human) --")
    L.append(f"   Δnum mean = {fmt(t_num['mean'])}   95% CI [{fmt(t_num.get('ci_lo'))}, "
             f"{fmt(t_num.get('ci_hi'))}]")
    L.append(f"   one-sided t = {fmt(t_num.get('t'))},  p = {fmt(t_num.get('p_one_sided'),5)},  "
             f"BF10 = {fmt(t_num.get('bf10'),3)}")
    L.append(f"   equivalence margin delta_num = {DELTA_NUM}:  gap exceeds margin (CI lower > delta)? "
             f"{t_num.get('exceeds_margin')}   equiv to zero (CI upper < delta)? {t_num.get('equiv_to_zero')}")
    L.append("")
    L.append("-- Secondary endpoint: fidelity cost  Delta_ARI = ARI(lambda=0) - ARI(lambda*)  (>0 = matching slope sacrifices fit) --")
    L.append(f"   ΔARI mean = {fmt(t_ari['mean'])}   95% CI [{fmt(t_ari.get('ci_lo'))}, "
             f"{fmt(t_ari.get('ci_hi'))}]")
    L.append(f"   单侧 t = {fmt(t_ari.get('t'))},  p = {fmt(t_ari.get('p_one_sided'),5)},  "
             f"BF10 = {fmt(t_ari.get('bf10'),3)}")
    L.append(f"   equivalence margin delta_ARI = {DELTA_ARI}:  cost exceeds margin? {t_ari.get('exceeds_margin')}   "
             f"与0等价? {t_ari.get('equiv_to_zero')}")
    L.append("")
    L.append("-- Verdict --")
    if caseB:
        L.append("* Case B (no sweet spot): at lambda* that reproduces human cluster-count slope,")
        L.append("  cluster-size slope is significantly below human, and the gap lower bound exceeds")
        L.append("  the pre-registered equivalence margin. => **the constraint is graded (soft), not a fixed cap**.")
        if t_ari.get("exceeds_margin"):
            L.append("  Secondary endpoint fidelity cost also significant, strengthening the argument.")
    elif caseA:
        L.append("* Case A (sweet spot exists): at lambda*, both slopes match human/baseline.")
        L.append("  => A single capacity strength can match both. The graded-constraint claim **does not hold**.")
    else:
        L.append("? Inconclusive: CI crosses equivalence margin (insufficient seeds or moderate gap).")
        L.append("  Suggestion: add seeds or densify around lambda* to tighten CI.")
    L.append("")
    L.append("Note: BF10>1 supports gap (the claim), <1 supports no-gap (counter-claim);")
    L.append("  JZS prior scale r=0.707. Equivalence margins are pre-registered values.")

    text = "\n".join(L)
    with open(os.path.join(a.out, "frontier_verdict.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", "replace").decode("utf-8", "replace"))
    print(f"\n-> {a.out}/frontier_summary.csv")
    print(f"-> {a.out}/frontier_verdict.txt")


if __name__ == "__main__":
    main()
