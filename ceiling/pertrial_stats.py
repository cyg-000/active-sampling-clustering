"""Per-trial export + inferential stats for the MODEL results (Results 6, 7) and
the satisficing silhouette comparison (Results 8 data).

Reuses the *same* test split, consensus machinery and ARI as ceiling.py / evaluate.py
(imported, not re-implemented) so every number is on the paper's existing scale.

Outputs
  outputs/pertrial.csv        one row per held-out test trial (for R mixed models)
  outputs/pertrial_stats.txt  printed inferential summary (effect sizes, CIs, tests)

Run:  python dataana/ceiling/pertrial_stats.py
"""
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ceiling as CE                       # reuse load_test, consensus_at_k, coassoc_counts, k_of
sys.path.insert(0, str(CE.TMP))
import torch                               # noqa: E402
from model import AssignNet                # noqa: E402
from gpu_data import build_tensors         # noqa: E402
from metrics import adjusted_rand, silhouette  # noqa: E402

OUT = CE.OUT
RNG = np.random.default_rng(0)
LOG = io.StringIO()


def emit(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n")


# ---------------------------------------------------------------- inference
@torch.no_grad()
def model_pertrial_ari(tag, d, rows):
    """Load runs/<tag>/best.pt, predict on the test rows, return per-trial ARI vs human."""
    ck_path = CE.TMP / "runs" / tag / "best.pt"
    if not ck_path.exists():
        return None
    ck = torch.load(ck_path, map_location="cpu")
    args = ck.get("args", {})
    net = AssignNet(use_gru=not args.get("no_gru", False)).eval()
    net.load_state_dict(ck["model"])
    idx = np.array([r["j"] for r in rows])
    T = build_tensors(d, idx, "human", args.get("scanpath", "full"), 0, "cpu")
    hard = net(T["X"], T["M"], T["O"]).argmax(-1).cpu().numpy()
    M = T["M"].cpu().numpy()
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        pred = np.where(M[i], hard[i], -1)[M[i]]
        out[i] = adjusted_rand(r["hum"], pred)
    return out


# ---------------------------------------------------------------- consensus / single (as §A)
def ceiling_pertrial(rows, by_base):
    """Per-trial single-other (within format) and leave-one-out consensus ARI."""
    grp = {}
    for b, rs in by_base.items():
        N = max(r["n"] for r in rs)
        for fmt in set(r["rt"] for r in rs):
            sub = [r for r in rs if r["rt"] == fmt]
            parts = [r["hum"] for r in sub]
            same, both = CE.coassoc_counts(parts, N)
            grp[(b, fmt)] = dict(same=same, both=both, sids=[r["sid"] for r in sub],
                                 parts=parts, ks=[CE.k_of(r["hum"]) for r in sub], N=N)
    single_f = np.full(len(rows), np.nan); cons = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        g = grp[(r["base"], r["rt"])]; N = g["N"]
        li = r["hum"][:N]; asg = (li >= 0)
        eq = (li[:, None] == li[None, :]) & np.outer(asg, asg)
        s2 = g["same"] - eq; b2 = g["both"] - np.outer(asg, asg)
        ks_o = [kk for sid, kk in zip(g["sids"], g["ks"]) if sid != r["sid"]]
        kstar = int(np.median(ks_o)) if ks_o else CE.k_of(r["hum"])
        cf = np.full(len(r["hum"]), -1); cf[:N] = CE.consensus_at_k(s2, b2, N, kstar)
        cons[i] = adjusted_rand(r["hum"], cf)
        fo = [p for sid, p in zip(g["sids"], g["parts"]) if sid != r["sid"]]
        if len(fo) > CE.N_OTHERS:
            fo = [fo[t] for t in RNG.permutation(len(fo))[:CE.N_OTHERS]]
        av = [adjusted_rand(r["hum"], o) for o in fo]
        av = [x for x in av if np.isfinite(x)]
        single_f[i] = np.mean(av) if av else np.nan
    return single_f, cons


# ---------------------------------------------------------------- small stats helpers
def cohen_dz(diff):
    diff = diff[np.isfinite(diff)]
    return diff.mean() / diff.std(ddof=1)


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    m = np.isfinite(a) & np.isfinite(b)
    try:
        return wilcoxon(a[m], b[m]).pvalue
    except ValueError:
        return np.nan


def cluster_boot_ci(vals, clusters, n_boot=3000):
    """95% CI for the mean, resampling whole clusters (base_uuid) with replacement."""
    vals = np.asarray(vals); clusters = np.asarray(clusters)
    uc = np.unique(clusters[np.isfinite(vals)])
    idx = {c: np.where((clusters == c) & np.isfinite(vals))[0] for c in uc}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(uc, len(uc), replace=True)
        take = np.concatenate([idx[c] for c in pick])
        means[b] = vals[take].mean()
    return np.nanpercentile(means, [2.5, 97.5])


def per_stimulus_paired(a, b, base):
    """Mean per base_uuid, then compare a vs b across stimuli (robust to trial dependence)."""
    ua = np.unique(base)
    ma = np.array([np.nanmean(a[base == u]) for u in ua])
    mb = np.array([np.nanmean(b[base == u]) for u in ua])
    wins = int(np.sum(ma > mb)); n = len(ua)
    p = wilcoxon_p(ma, mb)
    return wins, n, p, ma.mean(), mb.mean()


# ================================================================ main
def main():
    d, rows, by_base = CE.load_test()
    base = np.array([r["base"] for r in rows])
    emit(f"test trials = {len(rows)}   stimuli = {len(by_base)}   "
         f"(median {int(np.median([len(v) for v in by_base.values()]))} raters/stimulus)")

    hum = [r["hum"] for r in rows]
    gmm = [np.asarray(r["gmm"], int) for r in rows]
    dbs = [np.asarray(r["dbscan"], int) for r in rows]

    # ---- model per-trial ARI ----
    emit("\n[inference] running CPU inference for each model ...")
    ari = {}
    for tag in ["H_s0", "H_s1", "H_s2", "R_gmm", "R_dbscan", "R_shuffle"]:
        a = model_pertrial_ari(tag, d, rows)
        if a is not None:
            ari[tag] = a
            emit(f"   {tag:10s} mean ARI = {np.nanmean(a):.4f}")
    H = np.nanmean(np.vstack([ari["H_s0"], ari["H_s1"], ari["H_s2"]]), axis=0)   # per-trial, seed-avg

    # ---- reference algorithms' own agreement with humans ----
    ari_gmm_ref = np.array([adjusted_rand(hum[i], gmm[i]) for i in range(len(rows))])
    ari_dbs_ref = np.array([adjusted_rand(hum[i], dbs[i]) for i in range(len(rows))])

    # ---- ceiling ----
    emit("[consensus] computing single-other and leave-one-out consensus ARI ...")
    single_f, cons = ceiling_pertrial(rows, by_base)

    # ---- silhouettes (item 8) ----
    sil_h = np.array([silhouette(rows[i]["pts"], hum[i]) for i in range(len(rows))])
    sil_g = np.array([silhouette(rows[i]["pts"], gmm[i]) for i in range(len(rows))])
    sil_d = np.array([silhouette(rows[i]["pts"], dbs[i]) for i in range(len(rows))])
    k_h = np.array([CE.k_of(hum[i]) for i in range(len(rows))], float)
    k_g = np.array([CE.k_of(gmm[i]) for i in range(len(rows))], float)
    k_d = np.array([CE.k_of(dbs[i]) for i in range(len(rows))], float)

    # ---- export per-trial CSV (for R mixed models) ----
    cols = dict(j=[r["j"] for r in rows], base=base, sid=[r["sid"] for r in rows],
                rt=[r["rt"] for r in rows], n=[r["n"] for r in rows],
                ari_H=H, ari_H0=ari["H_s0"], ari_H1=ari["H_s1"], ari_H2=ari["H_s2"],
                ari_Rgmm=ari["R_gmm"], ari_Rdbscan=ari["R_dbscan"], ari_Rshuffle=ari["R_shuffle"],
                ari_gmm_ref=ari_gmm_ref, ari_dbscan_ref=ari_dbs_ref,
                single_within=single_f, consensus=cons,
                sil_human=sil_h, sil_gmm=sil_g, sil_dbscan=sil_d,
                k_human=k_h, k_gmm=k_g, k_dbscan=k_d)
    with open(OUT / "pertrial.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(list(cols))
        for i in range(len(rows)):
            w.writerow([cols[c][i] for c in cols])
    emit(f"[export] -> {OUT/'pertrial.csv'}  ({len(rows)} rows)")

    # ============================================================ Item 6: lesion
    emit("\n" + "=" * 70)
    emit("Result 6  Model-H vs lesioned targets (paired, per-trial ARI vs human)")
    emit("=" * 70)
    emit(f"  Model-H (seed-avg) mean ARI = {np.nanmean(H):.4f}  "
         f"[seeds {np.nanmean(ari['H_s0']):.3f}/{np.nanmean(ari['H_s1']):.3f}/{np.nanmean(ari['H_s2']):.3f}]")
    for tag, lab in [("R_gmm", "GMM target"), ("R_dbscan", "DBSCAN target"),
                     ("R_shuffle", "shuffled target")]:
        diff = H - ari[tag]
        dz = cohen_dz(diff); p = wilcoxon_p(H, ari[tag])
        wins, n, ps, mh, mt = per_stimulus_paired(H, ari[tag], base)
        emit(f"  H vs {lab:15s}: Δmean={np.nanmean(diff):+.4f}  dz={dz:+.2f}  "
             f"Wilcoxon p={p:.2e} | per-stimulus H>target in {wins}/{n} (p={ps:.3g})")
    for lab, ref in [("GMM-self", ari_gmm_ref), ("DBSCAN-self", ari_dbs_ref)]:
        diff = H - ref
        dz = cohen_dz(diff); p = wilcoxon_p(H, ref)
        wins, n, ps, mh, mt = per_stimulus_paired(H, ref, base)
        emit(f"  H vs {lab:15s}: Δmean={np.nanmean(diff):+.4f}  dz={dz:+.2f}  "
             f"Wilcoxon p={p:.2e} | per-stimulus H>ref in {wins}/{n} (p={ps:.3g}) "
             f"[ref mean {np.nanmean(ref):.3f}]")

    # ============================================================ Item 7: ceiling
    emit("\n" + "=" * 70)
    emit("Result 7  Calibration against the human consensus ceiling")
    emit("=" * 70)
    for nm, x in [("single-other (within format)", single_f), ("consensus ceiling", cons)]:
        ci = cluster_boot_ci(x, base)
        emit(f"  {nm:32s} mean={np.nanmean(x):.4f}  95% CI (stimulus-clustered) [{ci[0]:.3f}, {ci[1]:.3f}]")
    ciH = cluster_boot_ci(H, base)
    emit(f"  {'Model-H':32s} mean={np.nanmean(H):.4f}  95% CI [{ciH[0]:.3f}, {ciH[1]:.3f}]")
    emit(f"  Model-H / ceiling = {np.nanmean(H)/np.nanmean(cons)*100:.0f}%")
    # model below ceiling, above single
    for lab, other, direction in [("consensus ceiling", cons, "below"),
                                  ("single-other", single_f, "above")]:
        diff = H - other
        dz = cohen_dz(diff); p = wilcoxon_p(H, other)
        wins, n, ps, mh, mo = per_stimulus_paired(H, other, base)
        emit(f"  H {direction} {lab:22s}: Δmean={np.nanmean(diff):+.4f} dz={dz:+.2f} "
             f"Wilcoxon p={p:.2e} | per-stimulus H {'>' if direction=='above' else '<'} in "
             f"{wins if direction=='above' else n-wins}/{n} (p={ps:.3g})")

    # ============================================================ Item 8: silhouette (data + quick test)
    emit("\n" + "=" * 70)
    emit("Result 8  Human partitions are less separable (per-trial silhouette)")
    emit("=" * 70)
    emit(f"  median silhouette: human={np.nanmedian(sil_h):.4f}  GMM={np.nanmedian(sil_g):.4f}  "
         f"DBSCAN={np.nanmedian(sil_d):.4f}")
    emit(f"  mean   silhouette: human={np.nanmean(sil_h):.4f}  GMM={np.nanmean(sil_g):.4f}  "
         f"DBSCAN={np.nanmean(sil_d):.4f}")
    for lab, other in [("GMM", sil_g), ("DBSCAN", sil_d)]:
        diff = sil_h - other
        dz = cohen_dz(diff); p = wilcoxon_p(sil_h, other)
        wins, n, ps, mh, mo = per_stimulus_paired(sil_h, other, base)
        emit(f"  human vs {lab:7s}: Δmean={np.nanmean(diff):+.4f} dz={dz:+.2f} "
             f"Wilcoxon p={p:.2e} | per-stimulus human<{lab} in {n-wins}/{n} (p={ps:.3g})")
    emit(f"  k: human={np.nanmean(k_h):.3f}  GMM={np.nanmean(k_g):.3f}  DBSCAN={np.nanmean(k_d):.3f}  "
         "(mixed-model post-hoc on pertrial.csv done in R)")

    with open(OUT / "pertrial_stats.txt", "w", encoding="utf-8") as f:
        f.write(LOG.getvalue())
    emit(f"\n[done] stats -> {OUT/'pertrial_stats.txt'}")


if __name__ == "__main__":
    main()
