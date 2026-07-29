"""Per-trial layer-probe ARIs for Figure 5b, across all three seeds per architecture.

`autodl/arch/runs/probe.json` only stores, for seed s0, the MEAN ARI of each layer -
no dispersion and no per-trial values - so panel b could show neither error bars nor a
test against the raw-coordinate baseline (the baseline is printed by probe.py and then
thrown away). This script re-runs the same probe for s0/s1/s2 of each family and keeps
the per-trial arrays.

Reuses `autodl/arch/probe.py`'s own `cluster_rows` and the real `model2.AssignNet2`, so
the numbers land on the same scale as the published probe.json (verify: the s0 layer
means should match it).

Output: data_for_figs/f5_probe_pertrial.csv, long format
        trial, base, tag, fam, seed, layer ('0'..'3' | 'out' | 'coord'), ari

`base` (the stimulus) is carried through because the held-out set spans only 8 stimuli:
tests on this file have to be paired at the stimulus level, exactly as in figures 3 and 4.

Run:  python figs_nhb/prep_fig5_probe.py     (CPU only, several minutes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent / "autodl" / "arch"
sys.path.insert(0, str(ARCH))
import config as C                                    # noqa: E402
import probe as PROBE                                 # noqa: E402  cluster_rows
from data import load_npz, make_splits                # noqa: E402
from gpu_data import build_tensors                    # noqa: E402
from metrics import adjusted_rand                     # noqa: E402
from model2 import AssignNet2                         # noqa: E402

OUT = HERE / "data_for_figs"
RNG = np.random.default_rng(0)                        # same seed/selection as probe.py
N_TRIALS = 800
FAMS = {"A_base": "base", "A_dpool_nc": "dist-pool", "A_attn_nc": "attention"}
SEEDS = [0, 1, 2]


def main():
    dev = "cpu"
    d = load_npz()
    test = make_splits(d)["test"]
    sel = test[RNG.permutation(len(test))[:N_TRIALS]]
    hum = [np.asarray(d["human"][i], int) for i in sel]
    pts = [np.asarray(d["pts"][i], float) for i in sel]
    Ks = [len(np.unique(h[h >= 0])) for h in hum]
    bases = [str(d["base_uuid"][i]) for i in sel]
    print(f"[probe] {len(sel)} test trials over {len(set(bases))} stimuli")

    recs = []

    # baseline (1): cluster the raw POINT COORDINATES at the human k. The hidden layers
    # have to beat this or they add nothing -- the structure was already in the input.
    for b in range(len(sel)):
        if Ks[b] < 2:
            continue
        lab = PROBE.cluster_rows(pts[b], Ks[b])
        if lab is not None:
            recs.append(dict(trial=b, base=bases[b], tag="coord", fam="coord",
                             seed=-1, layer="coord",
                             ari=adjusted_rand(hum[b], lab)))
    base_mean = np.nanmean([r["ari"] for r in recs])
    print(f"  raw-coordinate baseline ARI = {base_mean:.4f}")

    for fam in FAMS:
        for s in SEEDS:
            tag = f"{fam}_s{s}"
            p = C.OUTPUT / tag / "best.pt"
            if not p.exists():
                print(f"  skip {tag}: no best.pt")
                continue
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

            means = []
            for L, h_ in enumerate(hs):
                H = h_.detach().cpu().numpy()
                vals = []
                for b in range(len(sel)):
                    m = M[b]
                    if Ks[b] < 2 or m.sum() < 4:
                        continue
                    lab = PROBE.cluster_rows(H[b][m], Ks[b])
                    if lab is None:
                        continue
                    a = adjusted_rand(hum[b], lab)
                    vals.append(a)
                    recs.append(dict(trial=b, base=bases[b], tag=tag, fam=fam,
                                     seed=s, layer=str(L), ari=a))
                means.append(np.nanmean(vals))
            for b in range(len(sel)):
                a = adjusted_rand(hum[b], np.where(M[b], hard[b], -1)[M[b]])
                recs.append(dict(trial=b, base=bases[b], tag=tag, fam=fam, seed=s,
                                 layer="out", ari=a))
            print(f"  {tag:<14s} layers " +
                  " ".join(f"{v:.4f}" for v in means) +
                  f"   out {np.nanmean([r['ari'] for r in recs if r['tag'] == tag and r['layer'] == 'out']):.4f}")

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "f5_probe_pertrial.csv", index=False)
    print(f"-> {OUT / 'f5_probe_pertrial.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
