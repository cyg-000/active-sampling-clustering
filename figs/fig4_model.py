"""Figure 4 | A neural model reverse-engineers the learned human prior.

a  architecture: permutation-invariant assignment network (+GRU over reveal order)
b  lesion: only training on human partitions recovers human agreement; a model
   trained on human data even beats the clustering algorithms' own agreement
c  Model-H matches human summary statistics that were never in the loss
d  Model-H reaches 85% of the human consensus ceiling; agreement decays with |Δk|
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch
import nhbstyle as S

S.apply_style()
HERE = os.path.dirname(__file__)
DAT = os.path.join(HERE, "data_for_figs")
LES = pd.read_csv(os.path.join(HERE, "..", "autodl", "tmp", "runs", "comparison.csv"))
CEIL = pd.read_csv(os.path.join(HERE, "..", "dataana", "ceiling", "outputs",
                                 "ceiling_summary.csv"))
# use f4_coupling_sem.csv instead of coupling.csv: it carries per-stimulus SEM
COUP = pd.read_csv(os.path.join(DAT, "f4_coupling_sem.csv"))
# per-trial ARI (4b, 4d-top) and per-trial k/silhouette (4c)
PT = pd.read_csv(os.path.join(HERE, "..", "dataana", "ceiling", "outputs", "pertrial.csv"))
PS = pd.read_csv(os.path.join(DAT, "f4_pertrial_stats.csv"))


def row(name):
    return LES[LES.model == name].iloc[0]


def sem_stim(df, col):
    """SEM of per-stimulus means (n = at most 8 stimuli)."""
    g = df.groupby("base")[col].mean()
    return g.std(ddof=1) / np.sqrt(len(g))


mh = LES[LES.model.isin(["H_s0", "H_s1", "H_s2"])]
mh_ari = mh.ARI_vs_human.mean(); mh_sd = mh.ARI_vs_human.std()
mh_k = mh.k_mean.mean(); mh_sil = mh.sil_median.mean()
# per-stimulus bars for 4b / 4c / 4d-top
PT["ari_H"] = PT[["ari_H0", "ari_H1", "ari_H2"]].mean(axis=1)
PS["k_MH"] = PS[["k_H_s0", "k_H_s1", "k_H_s2"]].mean(axis=1)
PS["sil_MH"] = PS[["sil_H_s0", "sil_H_s1", "sil_H_s2"]].mean(axis=1)

# ---------------------------------------------------------- isometric schematic helpers
DV = np.array([0.34, 0.46])                    # one unit of depth -> screen offset
A_BLUE = "#DCE9F5"; A_TAN = "#F5E6CC"; A_GREY = "#ECECEC"; A_OUTB = "#CFE0F0"


def _shade(hexc, f):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(min(1, (c / 255) * f) for c in (r, g, b))


def _slab(ax, x, y, w, h, depth, fc, z=1, lw=0.8, ec="#5A5A5A"):
    d = DV * depth
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + d[0], y + h + d[1]), (x + d[0], y + h + d[1])]
    side = [(x + w, y), (x + w, y + h), (x + w + d[0], y + h + d[1]), (x + w + d[0], y + d[1])]
    ax.add_patch(Polygon(top, closed=True, fc=_shade(fc, 1.08), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon(side, closed=True, fc=_shade(fc, 0.82), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon(front, closed=True, fc=fc, ec=ec, lw=lw, zorder=z + 0.1))


def _arw(ax, p0, p1, color="#5A5A5A", lw=1.0, style="-|>", z=5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7, lw=lw,
                                 color=color, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def architecture_panel(ax):
    yb, h = 2.2, 1.8
    # 1 point array
    _slab(ax, 0.4, yb, 1.15, h, 0.55, A_GREY)
    rng = np.random.default_rng(4)
    ax.scatter(0.58 + 0.8 * rng.random(9), yb + 0.22 + (h - 0.44) * rng.random(9),
               s=6, color="#555", zorder=3)
    ax.text(0.975, yb - 0.3, "Point\narray", ha="center", va="top", fontsize=5.0)
    # 2 shared MLP
    _slab(ax, 2.1, yb, 1.15, h, 0.55, A_BLUE)
    for i in range(5):
        yy = yb + 0.3 + i * (h - 0.6) / 4
        ax.plot([2.28, 3.06], [yy, yy], color="#7FA8CC", lw=1.3, zorder=3)
    ax.text(2.675, yb - 0.3, "Shared\nMLP embed", ha="center", va="top", fontsize=5.0)
    _arw(ax, (1.72, yb + h / 2), (2.05, yb + h / 2))
    # 3 set-interaction blocks (L receding slabs)
    bx = 4.05
    for k in range(3):
        off = DV * (2 - k) * 0.9
        _slab(ax, bx + off[0], yb + off[1], 1.2, h, 0.55, A_BLUE, z=1 + k)
    fcx = bx + 0.6
    for i in range(5):
        yy = yb + 0.3 + i * (h - 0.6) / 4
        ax.plot([bx + 0.16, bx + 1.04], [yy, yy], color="#7FA8CC", lw=1.3, zorder=4)
    ax.text(fcx, yb - 0.3, "Set-interaction\nblocks  (× L)", ha="center", va="top",
            fontsize=5.0)
    # aggregation motif: pool from all points, broadcast back (permutation-equivariant)
    agx, agy = fcx, yb + h + 1.35
    ax.add_patch(Polygon([(agx - 0.5, agy), (agx, agy + 0.42), (agx + 0.5, agy),
                          (agx, agy - 0.42)], closed=True, fc="#FBEAD1", ec="#B98A3E",
                         lw=0.9, zorder=6))
    for yy in (yb + 0.45, yb + h / 2, yb + h - 0.45):
        _arw(ax, (bx + 1.02, yy), (agx - 0.4, agy - 0.05), color="#D8BE8C", lw=0.6,
             style="-", z=5, rad=-0.10)
    _arw(ax, (agx + 0.15, agy - 0.4), (bx + 1.05, yb + h - 0.15), color="#B98A3E",
         lw=0.9, style="-|>", z=6, rad=-0.35)
    ax.text(agx - 1.2, agy + 0.5, "self + global pool → point", ha="center", va="bottom",
            fontsize=5.0, color="#8A6A2E")
    _arw(ax, (3.15, yb + h / 2), (bx - 0.02, yb + h / 2))
    # 4 GRU optional branch
    gx, gyr = 6.55, yb + 1.95
    _slab(ax, gx, gyr, 1.25, 1.35, 0.5, A_TAN, z=2)
    lx, ly = gx + 0.62, gyr + 0.68
    ax.add_patch(FancyArrowPatch((lx + 0.2, ly + 0.3), (lx - 0.2, ly + 0.3),
                 connectionstyle="arc3,rad=1.6", arrowstyle="-|>", mutation_scale=7,
                 lw=1.0, color="#B98A3E", zorder=6))
    ax.text(gx + 0.62, gyr - 0.28, "GRU over\nreveal order", ha="center", va="top",
            fontsize=5.0)
    ax.text(gx + 0.62, gyr + 1.78, "masked (funnel)\nexperiments only", ha="center",
            va="bottom", fontsize=5.0, style="italic", color="#8A6A2E")
    _arw(ax, (fcx + 0.78, yb + h - 0.12), (gx + 0.08, gyr + 0.12), color="#B98A3E",
         lw=0.9, style="-|>", rad=0.20)
    _arw(ax, (gx + 1.3, gyr + 0.12), (8.75, yb + h - 0.12), color="#B98A3E", lw=0.9,
         style="-|>", rad=0.20)
    _arw(ax, (fcx + 0.9, yb + h / 2), (8.7, yb + h / 2))
    # 5 per-point softmax -> slots (emergent K)
    ox = 8.8
    _slab(ax, ox, yb, 1.5, h, 0.55, A_OUTB, z=3)
    nK, nN = 4, 5
    gx0, gy0 = ox + 0.18, yb + 0.26
    cw, ch = (1.5 - 0.36) / nK, (h - 0.52) / nN
    asg = [0, 0, 1, 2, 1]
    for i in range(nN):
        for j in range(nK):
            col = S.CLUSTER_CYCLE[j] if asg[i] == j else "white"
            ax.add_patch(Polygon([(gx0 + j * cw, gy0 + i * ch),
                                  (gx0 + (j + 1) * cw, gy0 + i * ch),
                                  (gx0 + (j + 1) * cw, gy0 + (i + 1) * ch),
                                  (gx0 + j * cw, gy0 + (i + 1) * ch)], closed=True,
                                 fc=col, ec="#B9C7D6", lw=0.4, zorder=4))
    ax.text(ox + 0.75, yb - 0.3, "Per-point softmax\n→ K slots", ha="center", va="top",
            fontsize=5.0)
    _arw(ax, (10.35, yb + h / 2), (10.95, yb + h / 2))
    ax.text(11.02, yb + h / 2, "soft partition\neach point →\none soft slot", ha="left",
            va="center", fontsize=5.0)


fig = plt.figure(figsize=(S.COL2, 118 * S.MM))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.28,
                       left=0.090, right=0.975, top=0.92, bottom=0.09,
                       width_ratios=[1.4, 1])

# ============================================================ a: architecture (isometric)
axa = fig.add_subplot(gs[0, 0]); axa.axis("off")
axa.set_xlim(0, 13.2); axa.set_ylim(0, 7.4); axa.set_aspect("equal")
architecture_panel(axa)
S.panel_label(axa, "a", dx=0.0, dy=1.0)

# ============================================================ b: lesion
axb = fig.add_subplot(gs[0, 1])
items = [("Model-H\n(human target)", "ari_H", S.C["model"]),
         ("DBSCAN target", "ari_Rdbscan", S.C["dbscan_t"]),
         ("GMM target", "ari_Rgmm", S.C["gmm_t"]),
         ("Shuffled target", "ari_Rshuffle", S.C["shuffle"])]
# manuscript mixed-model: each lesion vs Model-H, all p < 10⁻¹⁶
dz_lesion = {"ari_Rdbscan": 0.44, "ari_Rgmm": 1.01, "ari_Rshuffle": 1.07}
y = np.arange(len(items))[::-1]
ekw = dict(elinewidth=0.8, capsize=2, ecolor="#333333")
for yi, (lab, col, c) in zip(y, items):
    v = PT.groupby("base")[col].mean().mean()
    e = sem_stim(PT, col)
    axb.barh(yi, v, color=c, height=0.62, xerr=e, error_kw=ekw)
    t = f"{v:.3f}"
    if col in dz_lesion:
        t += "   p $< 10^{-16}$, dz = " + str(dz_lesion[col])
    axb.text(v + e + 0.010, yi, t, va="center", fontsize=5.0)
# algorithms' own agreement with humans (reference lines)
axb.axvline(row("GMM (ref)").ARI_vs_human, color=S.C["gmm"], lw=0.9, ls=(0, (3, 2)))
axb.axvline(row("DBSCAN (ref)").ARI_vs_human, color=S.C["dbscan"], lw=0.9, ls=(0, (3, 2)))
axb.text(row("GMM (ref)").ARI_vs_human + 0.004, 3.55, "GMM self", fontsize=5.0,
         color=S.C["gmm"], ha="left", va="center")
axb.text(row("DBSCAN (ref)").ARI_vs_human + 0.004, 3.55, "DBSCAN self",
         fontsize=5.0, color=S.C["dbscan"], ha="left", va="center")
axb.set_yticks(y); axb.set_yticklabels([it[0] for it in items], fontsize=5.8)
axb.set_xlabel("Agreement with held-out humans (ARI)")
axb.set_xlim(0, 0.72)
S.panel_label(axb, "b", dx=-0.30)

# ============================================================ c: untrained stats
# Bars = mean over the 8 held-out stimuli, dots = the stimuli, whisker = 95% CI, and each
# model is tested against Human by a paired Wilcoxon over those same 8 stimuli. Model-H
# matching Human is a NULL result here, so the n it rests on has to be visible.
gsc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.5)
agents = [("Human", "human", S.C["human"], row("human (ref)")),
          ("Model-H", "mh", S.C["model"], None),
          ("GMM\ntarget", "gmm", S.C["gmm_t"], row("R_gmm")),
          ("Shuffle\ntarget", "shf", S.C["shuffle"], row("R_shuffle"))]
kkeys = ["k_human", "k_MH", "k_R_gmm", "k_R_shuffle"]
kvals = [PS.groupby("base")[k].mean().mean() for k in kkeys]
ksem = [sem_stim(PS, k) for k in kkeys]
skeys = ["sil_human", "sil_MH", "sil_R_gmm", "sil_R_shuffle"]
svals = [PS.groupby("base")[s].mean().mean() for s in skeys]
ssem = [sem_stim(PS, s) for s in skeys]
cols = [a[2] for a in agents]; names = [a[0] for a in agents]
axc1 = fig.add_subplot(gsc[0, 0])
axc1.bar(range(4), kvals, yerr=ksem, color=cols, width=0.72,
         error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor="#333333"))
axc1.axhline(kvals[0], color=S.C["human"], lw=0.7, ls=(0, (3, 2)))
# k: Model-H (3.30) below Human (3.87), t₇ = 3.94, p = .006, BF₁₀ = 10.3
# but far closer than shuffle (~2) or GMM (~6.7) — significance annotated, context in text
S.sig_bracket(axc1, 0, 1, max(kvals[0] + ksem[0], kvals[1] + ksem[1]) + 0.35,
              "p = .006", h=0.18, fs=5.0)
axc1.set_xticks(range(4)); axc1.set_xticklabels(names, fontsize=5.2, rotation=0)
axc1.set_ylabel("Mean cluster count"); axc1.set_ylim(0, 8.2)
axc1.set_yticks([0, 2, 4, 6, 8])
S.panel_label(axc1, "c", dx=-0.34)
axc2 = fig.add_subplot(gsc[0, 1])
axc2.bar(range(4), svals, yerr=ssem, color=cols, width=0.72,
         error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor="#333333"))
axc2.axhline(svals[0], color=S.C["human"], lw=0.7, ls=(0, (3, 2)))
# sil: Model-H (.377) matches Human (.369), t₇ = −0.46, p = .66, BF₀₁ = 2.7
S.sig_bracket(axc2, 0, 1, max(svals[0] + ssem[0], svals[1] + ssem[1]) + 0.03,
              "p = .66", h=0.012, fs=5.0)
axc2.set_xticks(range(4)); axc2.set_xticklabels(names, fontsize=5.2)
axc2.set_ylabel("Silhouette"); axc2.set_ylim(0, 0.52)
axc2.set_yticks([0, 0.1, 0.2, 0.3, 0.4])

# ============================================================ d: ceiling + decay
gsd = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.7,
                                       height_ratios=[1, 1])
axd1 = fig.add_subplot(gsd[0])
d1_cols = ["single_within", "ari_H", "consensus"]
d1_vals = [PT.groupby("base")[c].mean().mean() for c in d1_cols]
d1_sem = [sem_stim(PT, c) for c in d1_cols]
levels = [("Single person\n→ single person", d1_vals[0], "#8C8C8C"),
          ("Model-H", d1_vals[1], S.C["model"]),
          ("Consensus ceiling", d1_vals[2], S.C["human"])]
yy = np.arange(3)[::-1]
for i, (lab, v, c) in enumerate(levels):
    yi = yy[i]; e = d1_sem[i]
    axd1.barh(yi, v, color=c, height=0.6, xerr=e,
              error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#333333"))
    axd1.text(v + e + 0.006, yi, f"{v:.3f}", va="center", fontsize=5.3)
axd1.set_yticks(yy); axd1.set_yticklabels([l[0] for l in levels], fontsize=5.6)
axd1.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axd1.set_xlim(0, 0.72); axd1.set_xlabel("Agreement with humans (ARI)")
# manuscript: Model-H vs ceiling p = 4x10^-87, below on 8/8 stimuli
axd1.text(0.62, 2.0, "vs ceiling\np $= 4 \\times 10^{-87}$", fontsize=5.0,
          color=S.C["human"], ha="left", va="center")
S.panel_label(axd1, "d", dx=-0.30, dy=1.08)

axd2 = fig.add_subplot(gsd[1])
axd2.errorbar(COUP.abs_dk, COUP.mean_ari, yerr=COUP.sem_stim, color="#333333",
              marker="o", ms=3, lw=1.1, capsize=1.8, elinewidth=0.7)
axd2.set_xlabel("Difference in cluster count |Δk|")
axd2.set_ylabel("Mean ARI")
axd2.set_xticks(range(0, 6)); axd2.set_xlim(-0.3, 5.3)
axd2.set_ylim(0.18, 0.55)
axd2.set_yticks([0.2, 0.3, 0.4, 0.5])

S.save(fig, "fig4_model")
