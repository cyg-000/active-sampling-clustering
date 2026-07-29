"""Per-trial statistics that Figure 4 needs but no existing export carries.

Two gaps in the published exports:

  1. `autodl/tmp/runs/comparison.csv` gives each model ONE summary k_mean / sil_median,
     so panel c had no error bars and no test. Here we re-run the same CPU inference
     `dataana/ceiling/pertrial_stats.py` already uses and keep the PER-TRIAL k and
     silhouette, which makes a paired test against the human values possible.
  2. `dataana/ceiling/outputs/coupling.csv` stores mean ARI and n per |Δk| bin but no
     dispersion, so panel d's decay curve had no error bars. Here we redo the same
     human-human pairwise sampling as `ceiling.py` and also keep the SEM.

Everything is imported from the analysis modules (never re-implemented) so the test
split, the ARI definition and the silhouette definition stay byte-identical to the paper.

!! Caveat carried into the figure: H_s0/s1/s2 are three seeds of the same model, but the
   lesioned targets R_gmm / R_shuffle exist as a SINGLE run each. Their per-trial tests
   against humans are valid at the trial level and carry no across-seed generalisation.

Run:  python figs_nhb/prep_fig4_stats.py     (CPU only, a few minutes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "dataana" / "ceiling"))
import ceiling as CE                                  # noqa: E402  load_test, k_of
sys.path.insert(0, str(CE.TMP))
import torch                                          # noqa: E402
from model import AssignNet                           # noqa: E402
from gpu_data import build_tensors                    # noqa: E402
from metrics import adjusted_rand, silhouette         # noqa: E402

OUT = HERE / "data_for_figs"
MODELS = ["H_s0", "H_s1", "H_s2", "R_gmm", "R_dbscan", "R_shuffle"]


@torch.no_grad()
def model_pertrial(tag, d, rows):
    """Predict on the test rows; return per-trial (k, silhouette, ARI vs human).

    Same load/predict path as dataana/ceiling/pertrial_stats.py:model_pertrial_ari,
    extended to also read out the two summary statistics panel c plots.
    """
    ck_path = CE.TMP / "runs" / tag / "best.pt"
    if not ck_path.exists():
        print(f"  skip {tag}: no best.pt")
        return None
    ck = torch.load(ck_path, map_location="cpu")
    args = ck.get("args", {})
    net = AssignNet(use_gru=not args.get("no_gru", False)).eval()
    net.load_state_dict(ck["model"])
    idx = np.array([r["j"] for r in rows])
    T = build_tensors(d, idx, "human", args.get("scanpath", "full"), 0, "cpu")
    hard = net(T["X"], T["M"], T["O"]).argmax(-1).cpu().numpy()
    M = T["M"].cpu().numpy()
    k = np.full(len(rows), np.nan)
    sil = np.full(len(rows), np.nan)
    ari = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        pred = np.where(M[i], hard[i], -1)[M[i]]
        k[i] = CE.k_of(pred)
        sil[i] = silhouette(r["pts"], pred)
        ari[i] = adjusted_rand(r["hum"], pred)
    print(f"  {tag:<10s} k_mean={np.nanmean(k):.3f}  sil_median={np.nanmedian(sil):.4f}"
          f"  ARI={np.nanmean(ari):.4f}")
    return k, sil, ari


def pertrial_table(d, rows):
    """One row per held-out trial: human + every model's k / silhouette."""
    tab = {
        "j": [r["j"] for r in rows],
        "base": [r["base"] for r in rows],
        "n": [r["n"] for r in rows],
        "k_human": [CE.k_of(r["hum"]) for r in rows],
        "sil_human": [silhouette(r["pts"], r["hum"]) for r in rows],
        "k_gmm": [CE.k_of(r["gmm"]) for r in rows],
        "sil_gmm": [silhouette(r["pts"], r["gmm"]) for r in rows],
        "k_dbscan": [CE.k_of(r["dbscan"]) for r in rows],
        "sil_dbscan": [silhouette(r["pts"], r["dbscan"]) for r in rows],
    }
    print("[inference] per-trial k / silhouette for each model ...")
    for tag in MODELS:
        got = model_pertrial(tag, d, rows)
        if got is None:
            continue
        k, sil, ari = got
        tab[f"k_{tag}"] = k
        tab[f"sil_{tag}"] = sil
        tab[f"ari_{tag}"] = ari
    return pd.DataFrame(tab)


def coupling_with_sem(rows, by_base):
    """Human-human pairwise ARI binned by |Δk|, WITH dispersion.

    !! Differs from ceiling.py:255 on purpose. That version subsamples 80 raters per
       stimulus from a module-level RNG that earlier sections have already drawn from,
       so its numbers are not reproducible in isolation (re-running the same seed here
       gives |Δk|=0 -> .505 rather than the stored .451). We enumerate ALL within-stimulus
       pairs instead: ~677k pairs, ~90 s, and a fixed answer with no seed dependence.

    Two error bars are computed. Pair-level SEM is meaningless as an error bar here -
    pairs sharing a rater or a stimulus are not independent - so the figure uses the
    STIMULUS-CLUSTERED SEM (sd of the 8 per-stimulus bin means / sqrt(8)).
    """
    ari, dk, base = [], [], []
    for b, rs in by_base.items():
        ks = [CE.k_of(r["hum"]) for r in rs]           # once per rater, not once per pair
        for i in range(len(rs)):
            for jj in range(i + 1, len(rs)):
                a = adjusted_rand(rs[i]["hum"], rs[jj]["hum"])
                if np.isfinite(a):
                    ari.append(a); dk.append(abs(ks[i] - ks[jj])); base.append(b)
    ari = np.asarray(ari, float); dk = np.asarray(dk, int); base = np.asarray(base)
    bases = np.unique(base)
    out = []
    for g in range(0, 6):
        m = dk == g
        if m.sum() == 0:
            continue
        per_stim = np.array([ari[m & (base == b)].mean() for b in bases
                             if (m & (base == b)).sum() > 0])
        out.append(dict(abs_dk=g, mean_ari=ari[m].mean(),
                        sem_stim=per_stim.std(ddof=1) / np.sqrt(len(per_stim)),
                        sem_pair=ari[m].std(ddof=1) / np.sqrt(m.sum()),
                        n_stim=len(per_stim), n=int(m.sum())))
    return pd.DataFrame(out)


def main():
    print("[load] test split (base_uuid, seed 42) ...")
    d, rows, by_base = CE.load_test()
    print(f"  {len(rows)} test trials over {len(by_base)} stimuli")

    tab = pertrial_table(d, rows)
    tab.to_csv(OUT / "f4_pertrial_stats.csv", index=False)
    print(f"-> {OUT / 'f4_pertrial_stats.csv'}  ({len(tab)} rows)")

    print("[coupling] human-human pairwise ARI by |dk| ...")
    cp = coupling_with_sem(rows, by_base)
    cp.to_csv(OUT / "f4_coupling_sem.csv", index=False)
    print(cp.to_string(index=False))
    print(f"-> {OUT / 'f4_coupling_sem.csv'}")


if __name__ == "__main__":
    main()
