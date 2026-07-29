"""Figure 2 | Two invariant slopes, reallocated by response format.

a  number of clusters vs set size (all slopes > 0; anchor shallower)
b  cluster size (numerosity) vs set size (all slopes > 0; anchor steeper)
c  pooled reallocation: change in each slope from format / visibility (beta +/- 95% CI)
d  grouping granularity is a stable individual trait (test-retest of median k)
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
DAT = os.path.join(HERE, "data_for_figs")
TAB = os.path.join(HERE, "..", "dataana", "outputs", "tables")

kc = pd.read_csv(os.path.join(DAT, "f2_nclusters_by_np.csv"))
nz = pd.read_csv(os.path.join(DAT, "f2_numerosity_by_np.csv"))
coef = pd.read_csv(os.path.join(DAT, "f2_pooled_coefs.csv"))
rt = pd.read_csv(os.path.join(DAT, "f2_retest_nclusters.csv"))

EXPS = ["exp1", "exp2", "exp3", "exp4"]
# format -> colour, visibility -> linestyle/marker
FCOL = {"exp1": S.C["lasso"], "exp2": S.C["lasso"],
        "exp3": S.C["anchor"], "exp4": S.C["anchor"]}
LS = {"exp1": "-", "exp2": (0, (4, 2)), "exp3": "-", "exp4": (0, (4, 2))}
MK = {"exp1": "o", "exp2": "s", "exp3": "o", "exp4": "s"}


def slope_panel(ax, df, ylab, letter):
    for e in EXPS:
        g = df[df.exp == e].sort_values("n_points")
        vis, fmt = S.EXP_META[e]
        ax.errorbar(g.n_points, g["mean"], yerr=g["se"], color=FCOL[e],
                    ls=LS[e], marker=MK[e], ms=3.2, lw=1.1, capsize=1.5,
                    elinewidth=0.6, mfc=(FCOL[e] if vis == "full" else "white"),
                    mec=FCOL[e], mew=0.9, zorder=3)
    ax.set_xlabel("Number of points in array")
    ax.set_ylabel(ylab)
    ax.set_xticks([10, 20, 30, 40])
    S.panel_label(ax, letter)


fig = plt.figure(figsize=(S.COL2, 118 * S.MM))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32,
                       left=0.115, right=0.985, top=0.9, bottom=0.09)

# ---------------------------------------------------------- a & b
axa = fig.add_subplot(gs[0, 0])
slope_panel(axa, kc, "Number of clusters", "a")

axb = fig.add_subplot(gs[0, 1])
slope_panel(axb, nz, "Cluster size (points per cluster)", "b")

# shared legend (format = colour, visibility = fill/line)
leg = [
    Line2D([0], [0], color=S.C["lasso"], marker="o", ls="-", ms=3.5,
           label="Lasso · full view (Exp 1)"),
    Line2D([0], [0], color=S.C["lasso"], marker="s", ls=(0, (4, 2)), ms=3.5,
           mfc="white", label="Lasso · masked (Exp 2)"),
    Line2D([0], [0], color=S.C["anchor"], marker="o", ls="-", ms=3.5,
           label="Anchor · full view (Exp 3)"),
    Line2D([0], [0], color=S.C["anchor"], marker="s", ls=(0, (4, 2)), ms=3.5,
           mfc="white", label="Anchor · masked (Exp 4)"),
]
axa.legend(handles=leg, loc="upper left", fontsize=5.6, ncol=1,
           borderaxespad=0.2)

# ---------------------------------------------------------- c: reallocation
axc = fig.add_subplot(gs[1, 0])
mods = [("n_points:formatanchor", "Anchor − Lasso\n(format)"),
        ("n_points:visibilityfunnel", "Masked − Full\n(visibility)")]
ypos = {"n_clusters": 0.14, "numerosity": -0.14}
mcol = {"n_clusters": "#444444", "numerosity": S.C["accent"]}
mmk = {"n_clusters": "o", "numerosity": "D"}
for j, (term, mlab) in enumerate(mods):
    for meas in ("n_clusters", "numerosity"):
        row = coef[(coef.model == meas) & (coef.term == term)].iloc[0]
        y = j + ypos[meas]
        ci = 1.96 * row.se
        axc.errorbar(row.beta, y, xerr=ci, color=mcol[meas], marker=mmk[meas],
                     ms=4.5, capsize=2, lw=0, elinewidth=1.0, zorder=3)
axc.axvline(0, color="#999999", lw=0.7, zorder=1)
axc.set_yticks([0, 1]); axc.set_yticklabels([m[1] for m in mods], fontsize=6)
axc.set_ylim(-0.6, 1.6); axc.invert_yaxis()
axc.set_xlabel("Change in per-point slope (β)")
axc.legend(handles=[
    Line2D([0], [0], color="#444444", marker="o", ls="", ms=4.5,
           label="Number-of-clusters slope"),
    Line2D([0], [0], color=S.C["accent"], marker="D", ls="", ms=4.5,
           label="Cluster-size slope")],
    loc="lower right", fontsize=5.6)
S.panel_label(axc, "c")

# ---------------------------------------------------------- d: trait stability
# !! log2 axes, not the old [0.5, 8.5] linear box. That box silently cut 9 of the 67
#    Exp-1 participants (median k up to 24) while the printed r was computed on all 67.
#    On log2 the identity line is still a straight diagonal and every participant shows.
axd = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(3)
rlab = {"exp1": S.C["lasso"], "exp3": S.C["anchor"]}
rname = {"exp1": "Lasso", "exp3": "Anchor"}
for e in ("exp1", "exp3"):
    g = rt[rt.exp == e]
    # jitter is multiplicative so it stays visually constant on a log axis
    jx = g.p1 * np.exp(rng.uniform(-0.055, 0.055, len(g)))
    jy = g.p2 * np.exp(rng.uniform(-0.055, 0.055, len(g)))
    r = np.corrcoef(g.p1, g.p2)[0, 1]
    axd.scatter(jx, jy, s=7, color=rlab[e], alpha=0.55, edgecolor="white", lw=0.3,
                label=f"{rname[e]}: r = {r:.2f}")
lim = [1.7, 27]
axd.plot(lim, lim, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=0)
axd.set_xscale("log", base=2); axd.set_yscale("log", base=2)
axd.set_xlim(lim); axd.set_ylim(lim); axd.set_aspect("equal")
for a in (axd.xaxis, axd.yaxis):
    a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(plt.NullFormatter())
axd.set_xticks([2, 4, 8, 16, 24]); axd.set_yticks([2, 4, 8, 16, 24])
axd.set_xlabel("Median clusters — presentation 1")
axd.set_ylabel("Presentation 2")
axd.legend(loc="lower right", fontsize=5.8, borderaxespad=0.3)
S.panel_label(axd, "d")

S.save(fig, "fig2_invariants")
