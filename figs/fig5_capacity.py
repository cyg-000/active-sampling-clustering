"""Figure 5 | The cluster-count slope reflects a graded capacity limit, not architecture.

a  architecture ablation: richer interaction raises fit (ARI) but the recovered
   cluster-count slope stays near zero, far below the human value
b  layer-wise probe: human-like grouping is present early in the hidden layers,
   above both the raw-coordinate baseline and the network's own output
c  two-slope Pareto frontier from a dense capacity dose-response: raising the
   cluster-count slope with a per-group penalty forces the cluster-size slope down;
   the human point (both slopes) lies OUTSIDE the achievable frontier -- no single
   fixed capacity reproduces both, so the constraint is graded, not a fixed cap
   (gap = .085, 95% CI [.082,.088], BF10 ~ 1.9e4 vs a pre-registered margin .05)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import nhbstyle as S

S.apply_style()
HERE = os.path.dirname(__file__)
cmp = pd.read_csv(os.path.join(HERE, "..", "autodl", "arch", "runs", "comparison.csv"))
# per-trial probe ARIs for all three seeds + raw-coordinate baseline
# (prep_fig5_probe.py; probe.json only holds seed-s0 means, no SEM possible)
PROBE = pd.read_csv(os.path.join(HERE, "data_for_figs", "f5_probe_pertrial.csv"))
# dense λ re-sweep (capacity3_dense): 10 λ x 5 seeds, per-λ mean +/- SEM
FRONT = pd.read_csv(os.path.join(HERE, "..", "autodl", "capacity3_dense",
                                 "outputs", "frontier_summary.csv"))
_hum = cmp[cmp.model.str.contains("human")]
HUMAN_SLOPE = float(_hum.filter(like="slope_k_vs_n").iloc[0, 0])   # 0.061 cluster-count slope
HUMAN_NUM = float(_hum.filter(like="slope_num_vs_n").iloc[0, 0])   # 0.259 cluster-size slope

fig = plt.figure(figsize=(S.COL2, 66 * S.MM))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                       left=0.065, right=0.925, top=0.82, bottom=0.17)

# ============================================================ a: arch ablation
axa = fig.add_subplot(gs[0, 0])
arch = cmp[~cmp.model.str.contains("ref")].copy()
arch["fam"] = arch.model.str.replace(r"_s\d+$", "", regex=True)
fams = ["A_base", "A_ncount", "A_dpool", "A_dpool_nc", "A_attn", "A_attn_nc"]
famlab = {"A_base": "base", "A_ncount": "+count", "A_dpool": "dist-pool",
          "A_dpool_nc": "dist-pool+c", "A_attn": "attention", "A_attn_nc": "attn+c"}
# sequential blue ramp = increasing interaction richness
ramp = ["#BBD6EA", "#9CC3E0", "#6BA7D2", "#3E86BE", "#1F6BA6", "#0B4C7E"]
for f, col in zip(fams, ramp):
    g = arch[arch.fam == f]
    axa.scatter(g.slope_k_vs_n, g.ARI_vs_human, s=16, color=col,
                edgecolor="white", lw=0.4, zorder=3, label=famlab[f])
axa.axvline(HUMAN_SLOPE, color=S.C["accent"], lw=1.0, ls=(0, (4, 2)), zorder=2)
axa.text(HUMAN_SLOPE - 0.002, 0.447, "human\nslope", color=S.C["accent"],
         fontsize=5.2, ha="right", va="center")
axa.set_xlim(-0.005, 0.068); axa.set_xlabel("Recovered cluster-count slope")
axa.set_ylabel("Agreement with humans (ARI)")
axa.legend(loc="lower left", fontsize=5.0, ncol=2, columnspacing=0.8,
           handletextpad=0.2)
S.panel_label(axa, "a")

# ============================================================ b: layer probe
axb = fig.add_subplot(gs[0, 1])
pcol = {"A_base": S.C["human"], "A_dpool_nc": S.C["dbscan"],
        "A_attn_nc": S.C["model"]}
pname = {"A_base": "base", "A_dpool_nc": "dist-pool", "A_attn_nc": "attention"}
XOUT = 4.4


def probe_ms(fam, layer):
    """Mean and SEM across the 3 training seeds."""
    per_seed = PROBE[(PROBE.fam == fam) & (PROBE.layer == layer)] \
        .groupby("seed")["ari"].mean()
    return per_seed.mean(), per_seed.std(ddof=1) / np.sqrt(len(per_seed))


for i, (fam, col) in enumerate(pcol.items()):
    ms = [probe_ms(fam, L) for L in ["0", "1", "2", "3"]]
    axb.errorbar(range(4), [m[0] for m in ms], yerr=[m[1] for m in ms],
                 marker="o", ms=3.5, color=col, lw=1.2, capsize=1.6,
                 elinewidth=0.7, label=pname[fam])
    mo, so = probe_ms(fam, "out")
    axb.errorbar([XOUT + (i - 1) * 0.16], [mo], yerr=[so], marker="s", ms=3.5,
                 color=col, capsize=1.6, elinewidth=0.7, lw=0, zorder=3)

b_coord = PROBE[PROBE.layer == "coord"]["ari"].mean()
axb.axhline(b_coord, color="#999999", lw=0.8, ls=(0, (3, 2)))
axb.text(0.0, b_coord + 0.012, "raw-coordinate baseline", fontsize=5.0, color="#888")
axb.set_xticks([0, 1, 2, 3, XOUT])
axb.set_xticklabels(["0", "1", "2", "3", "out"])
axb.set_xlabel("Network block (hidden) → output")
axb.set_ylabel("Probe ARI vs humans")
axb.set_ylim(0.15, 0.58); axb.set_xlim(-0.2, 4.9)
axb.legend(loc="center right", fontsize=5.4)
S.panel_label(axb, "b")

# ============================================================ c: dose-response
axc = fig.add_subplot(gs[0, 2])
d = FRONT.sort_values("slope_k_mean")
xk, yn = d.slope_k_mean.values, d.slope_num_mean.values
# achievable region = at/below the frontier (best cluster-size slope per count slope)
axc.fill_between(xk, 0, yn, color="#BBBBBB", alpha=0.30, zorder=1, lw=0)
axc.plot(xk, yn, color="#8C8C8C", lw=0.8, zorder=2)
axc.errorbar(FRONT.slope_k_mean, FRONT.slope_num_mean,
             xerr=FRONT.slope_k_sem, yerr=FRONT.slope_num_sem, fmt="none",
             ecolor="#AAAAAA", elinewidth=0.6, capsize=1.2, zorder=3)
sc = axc.scatter(FRONT.slope_k_mean, FRONT.slope_num_mean, c=FRONT["lambda"],
                 cmap="viridis", s=24, edgecolor="white", lw=0.4, zorder=4)
# human point: both slopes, outside (above) the frontier
axc.axhline(HUMAN_NUM, ls=(0, (4, 2)), lw=0.7, color=S.C["accent"], zorder=2)
axc.scatter([HUMAN_SLOPE], [HUMAN_NUM], marker="*", s=120, color=S.C["accent"],
            edgecolor="white", lw=0.5, zorder=5)
yfront = float(np.interp(HUMAN_SLOPE, xk, yn))
axc.annotate("", xy=(HUMAN_SLOPE, HUMAN_NUM - 0.004), xytext=(HUMAN_SLOPE, yfront + 0.004),
             arrowprops=dict(arrowstyle="<->", color=S.C["accent"], lw=0.8))
axc.text(HUMAN_SLOPE + 0.005, HUMAN_NUM + 0.004, "human\n(both slopes)", fontsize=5.0,
         color=S.C["accent"], va="bottom")
axc.text(HUMAN_SLOPE + 0.005, (HUMAN_NUM + yfront) / 2, "gap = .085\nBF10 ~ 1.9e4",
         fontsize=5.0, color=S.C["accent"], va="center")
axc.set_xlim(0, 0.14); axc.set_ylim(0, 0.31)
axc.set_xlabel("Cluster-count slope")
axc.set_ylabel("Cluster-size slope")
cb = fig.colorbar(sc, ax=axc, fraction=0.046, pad=0.02)
cb.set_label("penalty weight λ", fontsize=5.5); cb.ax.tick_params(labelsize=5)
S.panel_label(axc, "c")

S.save(fig, "fig5_capacity")
