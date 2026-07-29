"""
Lesion evaluation: compare all models on the SAME test set.
Output: runs/comparison.csv. Evidence comes from columns NEVER in the loss
(k_median, sil_median, ch_median, slope_k, slope_num), not from FM.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
from data import load_npz, make_splits
from gpu_data import batches, build_tensors
from metrics import HUMAN_REF, summarize
from model import AssignNet, soft_assign

COLS = ["n_trials", "ARI_vs_human", "FM_vs_human", "k_median", "k_mean", "sil_median", "ch_median",
        "slope_k_vs_n", "slope_num_vs_n", "radial_median", "unassigned"]


@torch.no_grad()
def predict(net, T, batch=1024):
    """
    Lesion evaluation: compare all models on the SAME test set.
    Output: runs/comparison.csv. Evidence comes from columns NEVER in the loss
    (k_median, sil_median, ch_median, slope_k, slope_num), not from FM.
    """
    net.eval()
    labs = [None] * T["n"]
    for sel in batches(T, batch, False):
        P = soft_assign(net(T["X"][sel], T["M"][sel], T["O"][sel]).float())
        hard = P.argmax(-1).cpu().numpy()
        mm = T["M"][sel].cpu().numpy()
        for k, b in enumerate(sel.cpu().numpy()):
            labs[b] = np.where(mm[k], hard[k], -1)[mm[k]]
    return labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--split-by", default=C.SPLIT_BY)
    a = ap.parse_args()
    dev = C.DEVICE if torch.cuda.is_available() else "cpu"
    d = load_npz()
    sp = make_splits(d, a.split_by)
    test = sp["test"]
    pts = [np.asarray(d["pts"][i], float) for i in test]
    npt = [int(d["n_points"][i]) for i in test]
    hum = [np.asarray(d["human"][i], int) for i in test]

    rows = {}
    rows["human (ref)"] = summarize(pts, hum, npt, hum)
    for nm, key in (("GMM (ref)", "gmm"), ("DBSCAN (ref)", "dbscan")):
        rows[nm] = summarize(pts, [np.asarray(d[key][i], int) for i in test], npt, hum)

    tags = a.tags or sorted(p.name for p in C.OUTPUT.iterdir()
                            if (p / "best.pt").exists())
    for tag in tags:
        p = C.OUTPUT / tag / "best.pt"
        st = C.OUTPUT / tag / "status.json"
        if not p.exists():
            print(f"  skip {tag}: no best.pt"); continue
        note = ""
        if st.exists():
            s = json.load(open(st)).get("status")
            if s == "stuck_saddle":
                note = "  <- stuck at saddle (for lesion models this IS the result)"
            elif s not in ("ok", "timeout", "interrupted"):
                print(f"  skip {tag}: status={s}"); continue
        ck = torch.load(p, map_location=dev)
        args = ck.get("args", {})
        net = AssignNet(use_gru=not args.get("no_gru", False)).to(dev)
        net.load_state_dict(ck["model"])
        T = build_tensors(d, test, "human", args.get("scanpath", "full"),
                          seed=0, device=dev)
        rows[tag + note] = summarize(pts, predict(net, T), npt, hum)

    w = max(len(k) for k in rows) + 2
    print("\n" + " " * w + "".join(f"{c:>16s}" for c in COLS))
    for k, v in rows.items():
        print(f"{k:<{w}s}" + "".join(
            f"{v.get(c, float('nan')):>16.4f}" if isinstance(v.get(c), float)
            else f"{v.get(c, ''):>16}" for c in COLS))
    print("\nHuman baselines (measured / M&V):")
    for k, v in HUMAN_REF.items():
        print(f"  {k:10s} {v}")
    print("""
How to read:
  * **ARI is primary; FM is kept only for comparability with M&V.**
    All-one-cluster FM = 0.5748 (human-vs-human ceiling only 0.6829),
    ARI = 0.000 -- evaluating by FM alone makes a degenerate model look normal.
  * Calibration baselines (test set): all-one-cluster FM .5748/ARI .000;
    median-x-split FM .5943/ARI .3143; human-vs-human ceiling FM .6829.
  * Look at k_median / sil_median / ch_median / two slopes -- **none of these
    entered the loss.** Model-H hitting human baselines while Model-R misses
    => the model learned a human prior, not a generic clustering algorithm.
  * If Model-H also deviates on these columns, the pairwise loss only captured
    local proximity without the "impose structure" drive -- only then consider
    enabling the regularisers in config (and declare them as built-in).""")

    out = C.OUTPUT / "comparison.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("model," + ",".join(COLS) + "\n")
        for k, v in rows.items():
            f.write(k + "," + ",".join(str(v.get(c, "")) for c in COLS) + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
