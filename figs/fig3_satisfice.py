"""Figure 3 | Humans satisfice rather than optimize.

a  bounded effort: response time scales strongly sublinearly with set size
   (full-view experiments; log-log exponent b << 1)
b  the boundedness is COGNITIVE, not motor: decomposing response time into
   drawing (motor) and deliberation (non-drawing) time, deliberation is itself
   strongly sublinear; the near-zero-drawing anchor format is the most sublinear
c  humans neither maximize separability (silhouette) nor minimize cluster
   count -- they sit below the algorithmic separability frontier
d  confidence tracks the cleanliness of one's OWN partition (silhouette),
   not agreement with the group consensus (the null result)
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
EXPL = os.path.join(HERE, "..", "dataana", "explore2", "outputs")

rt = pd.read_csv(os.path.join(DAT, "f3_rt_by_np.csv"))
bexp = pd.read_csv(os.path.join(EXPL, "e2_q1_scaling.csv"))
comp = pd.read_csv(os.path.join(EXPL, "e5_rt_components.csv"))
e1 = pd.read_csv(os.path.join(EXPL, "e1_trials.csv"))
e1m = pd.read_csv(os.path.join(EXPL, "e1_models.csv"))
# per-stimulus sil / k for panel c error bars (n = 8 stimuli)
PT = pd.read_csv(os.path.join(HERE, "..", "dataana", "ceiling", "outputs", "pertrial.csv"))
STIM_SIL = PT.groupby("base")[["sil_human", "sil_gmm", "sil_dbscan"]].mean()
STIM_K = PT.groupby("base")[["k_human", "k_gmm", "k_dbscan"]].mean()


def sem_stim(df, col):
    v = df[col].dropna().values
    return v.std(ddof=1) / np.sqrt(len(v))

fig = plt.figure(figsize=(S.COL2, 122 * S.MM))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.34,
                       left=0.075, right=0.985, top=0.9, bottom=0.085)

# ============================================================ a: RT scaling
axa = fig.add_subplot(gs[0, 0])
rcol = {"exp1": S.C["lasso"], "exp3": S.C["anchor"]}
rname = {"exp1": "Lasso (Exp 1)", "exp3": "Anchor (Exp 3)"}
for e in ("exp1", "exp3"):
    g = rt[rt.exp == e].sort_values("n_points")
    x = np.log10(g.n_points.values)
    y = np.log10(g.gm_ms.values)
    axa.errorbar(g.n_points, g.gm_ms / 1000,
                 yerr=[(g.gm_ms - g.lo) / 1000, (g.hi - g.gm_ms) / 1000],
                 fmt="o", ms=3.5, color=rcol[e], capsize=1.5, elinewidth=0.6,
                 zorder=3, label=rname[e])
    b = float(bexp.loc[bexp.exp == e, "b_mean"].iloc[0])
    lo = float(bexp.loc[bexp.exp == e, "ci_lo"].iloc[0])
    hi = float(bexp.loc[bexp.exp == e, "ci_hi"].iloc[0])
    xc, yc = x.mean(), y.mean()
    xx = np.array([np.log10(9), np.log10(44)])
    yy = yc + b * (xx - xc)
    axa.plot(10 ** xx, 10 ** yy / 1000, color=rcol[e], lw=1.1, zorder=2)
    ytxt = 18.0 if e == "exp1" else 6.0
    axa.text(41, ytxt, f"b = {b:.2f}\n[{lo:.2f}, {hi:.2f}]", fontsize=5.5,
             color=rcol[e], ha="right", va="center")
# slope = 1 (linear) reference
xr = np.array([10, 40.0])
axa.plot(xr, 6.5 * (xr / 10.0), color="#999999", lw=0.8, ls=(0, (3, 2)), zorder=1)
axa.text(20.5, 15.0, "slope = 1\n(linear)", fontsize=5.3, color="#888888",
         rotation=32, ha="center", va="bottom")
axa.set_xscale("log"); axa.set_yscale("log")
axa.set_xticks([10, 20, 40]); axa.set_xticklabels([10, 20, 40])
axa.set_yticks([4, 6, 8, 10, 14]); axa.set_yticklabels([4, 6, 8, 10, 14])
axa.xaxis.set_minor_formatter(plt.NullFormatter())
axa.yaxis.set_minor_formatter(plt.NullFormatter())
axa.set_xlabel("Number of points (log)")
axa.set_ylabel("Response time (s, log)")
axa.legend(loc="upper left", fontsize=5.6)
S.panel_label(axa, "a")

# ============================================================ b: motor vs deliberation
# response time = drawing (motor) + deliberation (non-drawing). If the sublinearity
# were a motor artefact, deliberation would scale ~linearly. It does not.
axb = fig.add_subplot(gs[0, 1])


def brow(exp, component):
    return comp[(comp.exp == exp) & (comp.component == component)].iloc[0]


# rows bottom->top; grouped: lasso (total/drawing/deliberation) then anchor total
rows = [
    ("exp3", "total", S.C["anchor"], "D", "Anchor · total\n(near-zero drawing)"),
    ("exp1", "delib", S.C["lasso"], "D", "Lasso · deliberation\n(non-drawing)"),
    ("exp1", "motor", S.C["lasso"], "s", "Lasso · drawing (motor)"),
    ("exp1", "total", S.C["lasso"], "o", "Lasso · total"),
]
for i, (e, cpt, col, mk, lab) in enumerate(rows):
    r = brow(e, cpt)
    mfc = col if cpt != "delib" else "white"
    axb.errorbar(r.b_mean, i, xerr=[[r.b_mean - r.ci_lo], [r.ci_hi - r.b_mean]],
                 color=col, marker=mk, ms=5, mfc=mfc, mec=col, mew=1.0,
                 capsize=2.5, lw=0, elinewidth=1.0, zorder=3)
axb.axvline(1.0, ymax=0.87, color=S.C["accent"], lw=1.0, ls=(0, (4, 2)), zorder=1)
axb.text(1.0, 3.62, "linear\n(b = 1)", color=S.C["accent"], fontsize=5.2,
         ha="center", va="bottom")
axb.axvline(0.0, color="#CCCCCC", lw=0.6, zorder=0)
axb.set_yticks(range(4))
axb.set_yticklabels([r[4] for r in rows], fontsize=5.4)
axb.set_ylim(-0.6, 4.0)
axb.set_xlim(-0.05, 1.18)
axb.set_xlabel("Log–log scaling exponent  b")
S.panel_label(axb, "b")

# ============================================================ c: satisficing
ref = {"human": "human (ref)", "gmm": "GMM (ref)", "dbscan": "DBSCAN (ref)"}
cmp = pd.read_csv(os.path.join(HERE, "..", "autodl", "arch", "runs", "comparison.csv"))
val = {a: cmp[cmp.model == ref[a]].iloc[0] for a in ref}
gsc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.55)
agents = ["human", "gmm", "dbscan"]
aname = {"human": "Human", "gmm": "GMM", "dbscan": "DBSCAN"}
acol = {"human": S.C["human"], "gmm": S.C["gmm"], "dbscan": S.C["dbscan"]}

axc1 = fig.add_subplot(gsc[0, 0])
sil_cols = ["sil_human", "sil_gmm", "sil_dbscan"]
sil = [STIM_SIL[c].mean() for c in sil_cols]
sil_sem = [sem_stim(STIM_SIL, c) for c in sil_cols]
axc1.bar(range(3), sil, yerr=sil_sem, color=[acol[a] for a in agents], width=0.7,
         error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor="#333333"))
axc1.set_xticks(range(3)); axc1.set_xticklabels([aname[a] for a in agents],
                                                rotation=35, ha="right", fontsize=5.6)
axc1.set_ylabel("Silhouette"); axc1.set_ylim(0, 0.62)
# manuscript: human vs DBSCAN Δ = −.139, z = 42, p < 10⁻¹⁶, dz = −.57, lower on 7/8 stimuli
S.sig_bracket(axc1, 0, 2, sil[2] + sil_sem[2] + 0.03,
              "p $< 10^{-16}$, dz $= -0.57$", h=0.016, fs=5.0)
S.panel_label(axc1, "c", dx=-0.28)

axc2 = fig.add_subplot(gsc[0, 1])
k_cols = ["k_human", "k_gmm", "k_dbscan"]
kk = [STIM_K[c].mean() for c in k_cols]
kk_sem = [sem_stim(STIM_K, c) for c in k_cols]
axc2.bar(range(3), kk, yerr=kk_sem, color=[acol[a] for a in agents], width=0.7,
         error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor="#333333"))
axc2.axhline(kk[0], color=S.C["human"], lw=0.7, ls=(0, (3, 2)))
# Human vs DBSCAN on k: not significant (t₇ = 0.61, p = .56), DBSCAN k close to human
S.sig_bracket(axc2, 0, 2, max(kk[0] + kk_sem[0], kk[2] + kk_sem[2]) + 0.45,
              "p = .56", h=0.20, fs=5.0)
axc2.set_xticks(range(3)); axc2.set_xticklabels([aname[a] for a in agents],
                                                rotation=35, ha="right", fontsize=5.6)
axc2.set_ylabel("Number of clusters"); axc2.set_ylim(0, 7.2)
axc2.set_yticks([0, 2, 4, 6])

# ============================================================ d: confidence null
# partial (mixed-model) betas: what predicts confidence AFTER controlling for the
# other terms. A raw binned trend would confound agreement with silhouette.
axd = fig.add_subplot(gs[1, 1])
conds = [("exp2", S.C["lasso"], "o", "Lasso"),
         ("exp4", S.C["anchor"], "s", "Anchor"),
         ("pooled", "#333333", "D", "Pooled")]
preds = [("scale(sil)", "Own\nsilhouette"), ("scale(agreement)", "Consensus\nagreement")]
yoff = {"exp2": 0.22, "exp4": 0.0, "pooled": -0.22}
for j, (term, plab) in enumerate(preds):
    for e, col, mk, _ in conds:
        r = e1m[(e1m.exp == e) & (e1m.term == term)].iloc[0]
        y = j + yoff[e]
        axd.errorbar(r.beta, y, xerr=1.96 * r.se, color=col, marker=mk,
                     ms=4, capsize=2, lw=0, elinewidth=1.0, zorder=3)
axd.axvline(0, color="#999999", lw=0.7, zorder=1)
axd.set_yticks([0, 1]); axd.set_yticklabels([p[1] for p in preds], fontsize=6.2)
axd.set_ylim(-0.55, 1.55); axd.invert_yaxis()
axd.set_xlim(-0.08, 0.50)
axd.set_xlabel("Standardized effect on confidence (β)")
axd.legend(handles=[Line2D([0], [0], color=c, marker=m, ls="", ms=4, label=l)
                    for _, c, m, l in conds], loc="lower right", fontsize=5.4,
           title="", ncol=1)
S.panel_label(axd, "d")

S.save(fig, "fig3_satisfice")
