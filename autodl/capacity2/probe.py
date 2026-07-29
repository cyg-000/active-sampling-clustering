"""
Hidden-layer probe: at which layer does cluster structure emerge?
All 23,126 trials have human labels -- cluster each layer hidden representations
and compute ARI against human partitions. Three baselines: raw coordinates,
shuffled representations, output layer.
"""
import argparse
import json

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import config as C
from data import load_npz, make_splits
from gpu_data import build_tensors
from metrics import adjusted_rand
from model2 import AssignNet2

RNG = np.random.default_rng(0)


def cluster_rows(F, K):
    """Hidden-layer probe: at which layer does cluster structure emerge?
All 23,126 trials have human labels -- cluster each layer hidden representations
and compute ARI against human partitions. Three baselines: raw coordinates,
shuffled representations, output layer."""
    n = len(F)
    if n < 3 or K < 2 or K > n:
        return None
    F = F - F.mean(0)
    s = F.std(0)
    F = F / np.where(s > 1e-8, s, 1.0)
    D = 1.0 - np.corrcoef(F)
    D = np.nan_to_num((D + D.T) / 2.0)
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), "average"), K, "maxclust")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", required=True)
    ap.add_argument("--n", type=int, default=800, help="number of test trials to sample")
    a = ap.parse_args()
    dev = C.DEVICE if torch.cuda.is_available() else "cpu"

    d = load_npz()
    test = make_splits(d)["test"]
    sel = test[RNG.permutation(len(test))[:a.n]]
    hum = [np.asarray(d["human"][i], int) for i in sel]
    pts = [np.asarray(d["pts"][i], float) for i in sel]
    Ks = [len(np.unique(h[h >= 0])) for h in hum]

    b_xy = [adjusted_rand(h, cluster_rows(p, k))
            for h, p, k in zip(hum, pts, Ks) if k >= 2 and cluster_rows(p, k) is not None]
    print(f"\nBaseline-1  coordinate clustering (no model)  ARI = {np.nanmean(b_xy):.4f}   "
          f"<- hidden layers must beat this baseline to have learned anything")

    out = {}
    for tag in a.tags:
        p = C.OUTPUT / tag / "best.pt"
        if not p.exists():
            print(f"  skip {tag}: no best.pt"); continue
        ck = torch.load(p, map_location=dev)
        ar = ck.get("args", {})
        net = AssignNet2(pairing=ar.get("pairing", "meanmax"),
                         use_ncount=ar.get("ncount", False),
                         use_gru=not ar.get("no_gru", False)).to(dev).eval()
        net.load_state_dict(ck["model"])
        T = build_tensors(d, sel, "human", ar.get("scanpath", "full"), 0, dev)
        with torch.no_grad():
            logits, hs = net(T["X"], T["M"], T["O"], return_hidden=True)
        M = T["M"].cpu().numpy()
        hard = logits.argmax(-1).cpu().numpy()

        rows = {}
        for L, h_ in enumerate(hs):
            H = h_.detach().cpu().numpy()
            real, shuf = [], []
            for b in range(len(sel)):
                m = M[b]
                if Ks[b] < 2 or m.sum() < 4:
                    continue
                F = H[b][m]
                lab = cluster_rows(F, Ks[b])
                if lab is not None:
                    real.append(adjusted_rand(hum[b], lab))
                lab2 = cluster_rows(F[RNG.permutation(len(F))], Ks[b])
                if lab2 is not None:
                    shuf.append(adjusted_rand(hum[b], lab2))
            rows[L] = (np.nanmean(real), np.nanmean(shuf))
        o = [adjusted_rand(hum[b], np.where(M[b], hard[b], -1)[M[b]])
             for b in range(len(sel))]
        out[tag] = (rows, np.nanmean(o))

    print()
    for tag, (rows, o) in out.items():
        print(f"── {tag} ──")
        for L, (r, s) in rows.items():
            bar = "█" * int(max(r, 0) * 60)
            print(f"    block {L}:  ARI = {r:.4f}   (shuffle baseline {s:.4f})  {bar}")
        print(f"    output   :  ARI = {o:.4f}")
        peak = max(rows.items(), key=lambda kv: kv[1][0])
        gain = peak[1][0] - np.nanmean(b_xy)
        print(f"    => peak at block {peak[0]} (ARI {peak[1][0]:.4f}), "
              f"+{gain:+.4f} above coordinate baseline")
        if gain < 0.02:
            print("       !! Did NOT significantly exceed coordinate baseline."
                  " Correct conclusion: hidden layers do not directly encode clusters; clusters are a readout-layer product.")

    json.dump({k: {"layers": {str(i): list(v) for i, v in r.items()},
                   "output": float(o)} for k, (r, o) in out.items()},
              open(C.OUTPUT / "probe.json", "w"), indent=2)
    print(f"\n-> {C.OUTPUT / 'probe.json'}")


if __name__ == "__main__":
    main()
