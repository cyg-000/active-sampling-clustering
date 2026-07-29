"""Figure 1 | Paradigm and the reliability of imposed structure.

a  2x2 design matrix (visibility x response format -> Exp 1-4)
b  a provably-structureless stimulus (gap/BIC -> single cluster)
c  the same array partitioned by six participants (shared yet individual)
d  internal + inter-participant reliability (FM) across experiments,
   against Marupudi & Varma (2024) reference values.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
import nhbstyle as S

S.apply_style()
HERE = os.path.dirname(__file__)
DAT = os.path.join(HERE, "data_for_figs")
TAB = os.path.join(HERE, "..", "dataana", "outputs", "tables")

stim = pd.read_csv(os.path.join(DAT, "stim_coords.csv"))
part = pd.read_csv(os.path.join(DAT, "example_partitions.csv"))
# one row per (exp, sid): the paired data behind panel d's test (see export_fig1.R)
bysid = pd.read_csv(os.path.join(DAT, "f1_reliability_bysid.csv"))

CANVAS_W, CANVAS_H = 800.0, 500.0


def clean_array_ax(ax):
    # !! pad the limits: the real arrays reach y=16 and y=492 on a 500-tall canvas, so
    #    an exact 0..H box puts points on top of the frame.
    S.pad_lims(ax, CANVAS_W, CANVAS_H, frac=0.07)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color("#BBBBBB")


# ============================================================= figure
fig = plt.figure(figsize=(S.COL2, 108 * S.MM))
gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[1.0, 1.05],
                       hspace=0.42, wspace=0.55,
                       left=0.045, right=0.985, top=0.93, bottom=0.09)

# ---------------------------------------------------------- a: design matrix
axa = fig.add_subplot(gs[0, 0:7]); axa.axis("off")
axa.set_xlim(0, 10); axa.set_ylim(0, 10)
S.panel_label(axa, "a", dx=0.0, dy=1.0)

# grid geometry
x0, y0, cw, ch = 2.7, 1.1, 3.3, 3.6
cols = ["Lasso (draw loops)", "Anchor (place points)"]
rows = ["Masked\n(funnel)", "Full view"]
exp_at = {(0, 0): "exp1", (1, 0): "exp2", (0, 1): "exp3", (1, 1): "exp4"}
# NB rows drawn bottom-up: row idx0 = bottom = Full view
row_vis = ["full", "funnel"]

rng = np.random.default_rng(7)
for r in range(2):
    for c in range(2):
        cx, cy = x0 + c * cw, y0 + r * ch
        exp = {("full", 0): "exp1", ("funnel", 0): "exp2",
               ("full", 1): "exp3", ("funnel", 1): "exp4"}[(row_vis[r], c)]
        fmt = "lasso" if c == 0 else "anchor"
        edge = S.C[fmt]
        face = "white" if row_vis[r] == "full" else "#F0F0F0"
        rect = patches.FancyBboxPatch((cx, cy), cw * 0.9, ch * 0.82,
                                      boxstyle="round,pad=0.02,rounding_size=0.12",
                                      linewidth=1.1, edgecolor=edge,
                                      facecolor=face, zorder=1)
        axa.add_patch(rect)
        # mini task glyph: dots + (loop) or (voronoi-ish spokes).
        # spread_points, not raw uniform: uniform sampling puts pairs on top of each other.
        px, py = S.spread_points(rng, 9,
                                 cx + cw * 0.9 * 0.18, cx + cw * 0.9 * 0.82,
                                 cy + ch * 0.82 * 0.18, cy + ch * 0.82 * 0.78,
                                 min_d=0.44)
        S.scatter_pts(axa, px, py, s=5, color="#555555", lw=0.3)
        if fmt == "lasso":
            for grp in (px[:5], px[5:]):
                pass
            th = np.linspace(0, 2 * np.pi, 40)
            axa.plot(cx + cw * 0.30 + cw * 0.16 * np.cos(th),
                     cy + ch * 0.45 + ch * 0.18 * np.sin(th),
                     color=edge, lw=0.8, zorder=2)
            axa.plot(cx + cw * 0.60 + cw * 0.15 * np.cos(th),
                     cy + ch * 0.42 + ch * 0.16 * np.sin(th),
                     color=edge, lw=0.8, zorder=2)
        else:
            ax_anchor = np.array([cx + cw * 0.32, cx + cw * 0.60])
            ay_anchor = np.array([cy + ch * 0.5, cy + ch * 0.4])
            axa.scatter(ax_anchor, ay_anchor, s=16, marker="D",
                        color=edge, zorder=4)
            axa.plot([cx + cw * 0.46, cx + cw * 0.46],
                     [cy + ch * 0.15, cy + ch * 0.72],
                     color=edge, lw=0.7, ls=(0, (3, 2)), zorder=2)
        axa.text(cx + 0.12, cy + ch * 0.82 - 0.12, S.EXP_LABEL[exp],
                 ha="left", va="top", fontsize=6.5, fontweight="bold",
                 color=edge)

# column + row headers. The bold "Response format" sits ABOVE the two column names and
# is centred over the pair of columns, so it no longer overlaps "Lasso (draw loops)".
for c in range(2):
    axa.text(x0 + c * cw + cw * 0.45, y0 + 2 * ch + 0.18, cols[c],
             ha="center", va="bottom", fontsize=6.5)
axa.text(x0 + cw * 0.7, y0 + 2 * ch + 1.05, "Response format",
         ha="center", va="bottom", fontsize=7, fontweight="bold")
for r in range(2):
    axa.text(x0 - 0.25, y0 + r * ch + ch * 0.4, ["Full\nview", "Masked\n(funnel)"][r],
             ha="right", va="center", fontsize=6.5)
axa.text(0.55, y0 + ch, "Visibility", ha="center", va="center",
         rotation=90, fontsize=7, fontweight="bold")

# ---------------------------------------------------------- b: stimulus
axb = fig.add_subplot(gs[0, 7:12])
clean_array_ax(axb)
# closest neighbour pair is 16 px apart: a plain filled dot merges them, the white edge
# from scatter_pts keeps them readable as two points.
S.scatter_pts(axb, stim["x"], stim["y"], s=11, color="#444444", lw=0.4)
S.panel_label(axb, "b", dx=0.0, dy=1.04)

# ---------------------------------------------------------- c: partitions
gc = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, 0:8],
                                      hspace=0.42, wspace=0.14)
sids = list(pd.unique(part["sid"]))[:6]
for i, sid in enumerate(sids):
    ax = fig.add_subplot(gc[i // 3, i % 3])
    clean_array_ax(ax)
    p = part[part["sid"] == sid]
    for cid, g in p.groupby("cluster_id"):
        col = S.CLUSTER_CYCLE[(int(cid) - 1) % len(S.CLUSTER_CYCLE)]
        S.scatter_pts(ax, g["x"], g["y"], s=6, color=col, lw=0.4)
    k = int(p["n_clusters"].iloc[0])
    # k label OUTSIDE the box: inside the lower-left corner it landed on the data
    ax.text(0.0, 1.02, f"k = {k}", transform=ax.transAxes, fontsize=5.5,
            ha="left", va="bottom", color="#333333")
    if i == 0:
        S.panel_label(ax, "c", dx=-0.04, dy=1.10)

# ---------------------------------------------------------- d: reliability
axd = fig.add_subplot(gs[1, 8:12])
exps = ["exp1", "exp2", "exp3", "exp4"]


def mean_sem(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


internal, i_sem, inter, x_sem = [], [], [], []
for e in exps:
    g = bysid[bysid.exp == e]
    m, s = mean_sem(g.fm_internal); internal.append(m); i_sem.append(s)
    m, s = mean_sem(g.fm_inter); inter.append(m); x_sem.append(s)

x = np.arange(4); w = 0.38
ekw = dict(elinewidth=0.7, capsize=1.8, ecolor="#333333")
axd.bar(x - w / 2, internal, w, yerr=i_sem, color=S.C["human"],
        label="Internal (test–retest)", error_kw=ekw)
axd.bar(x + w / 2, inter, w, yerr=x_sem, color="#8C8C8C",
        label="Inter-participant", error_kw=ekw)
# paired t-test per experiment: all p < 2.7e-13, BF10 > 2.9e10
for i in range(4):
    top = max(internal[i] + i_sem[i], inter[i] + x_sem[i])
    t_p = stats.ttest_rel(bysid[bysid.exp == exps[i]].fm_internal,
                          bysid[bysid.exp == exps[i]].fm_inter).pvalue
    S.sig_bracket(axd, i - w / 2, i + w / 2, top + 0.185, S.pfmt(t_p),
                  h=0.022, fs=5.0, tpad=0.006)
# Marupudi & Varma (2024) references
axd.axhline(0.757, ls=(0, (4, 2)), lw=0.8, color=S.C["human"])
axd.axhline(0.63, ls=(0, (4, 2)), lw=0.8, color="#8C8C8C")
axd.text(3.55, 0.767, "M&V .76", fontsize=5, color=S.C["human"], va="bottom", ha="right")
axd.text(3.55, 0.64, "M&V .63", fontsize=5, color="#8C8C8C", va="bottom", ha="right")
axd.set_xticks(x); axd.set_xticklabels([S.EXP_LABEL[e] for e in exps], fontsize=6)
axd.set_ylim(0, 1.05); axd.set_ylabel("Fowlkes–Mallows agreement")
axd.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=5.5)
S.panel_label(axd, "d", dx=-0.16, dy=1.02)

S.save(fig, "fig1_reliability")
