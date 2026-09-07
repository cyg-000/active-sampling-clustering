"""Build the manuscript's six evidence-dense figures.

Figure 1 is the behavioral-design flowchart; Figure 2 is the point-information
oracle benchmark (next-dwell gain maps, oracle flow chart, and the behavioural
reliability result). Figures 3--6 carry the scaling, search/stopping, Model-H,
and control results. Inferential marks use the sampling units reported in the
manuscript tables.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "revision_analysis"
sys.path.insert(0, str(ROOT))
ORIG_DATA = ROOT / "figures" / "data"
RESULTS = ROOT / "revision_analysis" / "outputs"
RAW_RESULTS = RESULTS
OUT = HERE.parent / "outputs"
MODEL_H_SKETCH = HERE.parent / "assets" / "model_h_architecture_hi.png"

COL = {
    "human": "#111111",
    "model": "#0173B2",
    "quality": "#029E73",
    "comparator": "#DE8F05",
    "threshold": "#7B3FA0",
    "process": "#7A7A7A",
    "shuffle": "#B0B0B0",
    "lasso": "#0173B2",
    "anchor": "#DE8F05",
    "red": "#CC3311",
}
LABEL = {
    "exp1": "Lasso / full",
    "exp2": "Lasso / aperture",
    "exp3": "Anchor / full",
    "exp4": "Anchor / aperture",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def panel(ax, letter: str, x: float = -0.12, y: float = 1.07) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="right")


def summary_point(ax, x, mean, low, high, color, *, marker="o", ms=5.0,
                  zorder=4, fill="white") -> None:
    """Summerfield-style hollow summary marker with a thin 95% interval."""
    ax.errorbar(
        x, mean,
        yerr=[[mean - low], [high - mean]],
        fmt=marker, ms=ms, mfc=fill, mec="#111111", mew=0.8,
        color="#111111", ecolor="#111111", elinewidth=0.75,
        capsize=2.0, capthick=0.75, zorder=zorder,
    )


def mean_ci(values) -> tuple[float, float, float]:
    v = np.asarray(pd.Series(values).dropna(), float)
    m = float(np.mean(v))
    h = float(stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)) if len(v) > 1 else 0.0
    return m, m - h, m + h


def raincloud_v(ax, values, x, color, *, width=.32, seed=0, alpha=.20,
                marker="o", summary_offset=.075):
    """Vertical raincloud: right-half violin, left-side raw units, mean + 95% CI."""
    values = np.asarray(pd.Series(values).dropna(), float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    if len(np.unique(values)) > 1:
        violin = ax.violinplot(values, positions=[x], vert=True,
                               widths=width * 1.55, showmeans=False,
                               showmedians=False, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.16)
        path = body.get_paths()[0]
        path.vertices[:, 0] = np.maximum(path.vertices[:, 0], x)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-width * .44, -width * .08, len(values))
    ax.scatter(x + jitter, values, s=5.0, color=color, alpha=alpha,
               edgecolor="none", rasterized=True, zorder=2)
    m, lo, hi = mean_ci(values)
    ax.errorbar(x + summary_offset, m, yerr=[[m-lo], [hi-m]], fmt=marker,
                color=color, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=.8, ms=4.8, capsize=2.0,
                elinewidth=.9, zorder=4)
    return m, lo, hi


def raincloud_h(ax, values, y, color, *, width=.30, seed=0, alpha=.20,
                marker="o", summary_offset=.075):
    """Horizontal raincloud: upper-half violin, lower raw units, mean + 95% CI."""
    values = np.asarray(pd.Series(values).dropna(), float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    if len(np.unique(values)) > 1:
        violin = ax.violinplot(values, positions=[y], vert=False,
                               widths=width * 1.55, showmeans=False,
                               showmedians=False, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.16)
        path = body.get_paths()[0]
        path.vertices[:, 1] = np.maximum(path.vertices[:, 1], y)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-width * .44, -width * .08, len(values))
    ax.scatter(values, y + jitter, s=5.0, color=color, alpha=alpha,
               edgecolor="none", rasterized=True, zorder=2)
    m, lo, hi = mean_ci(values)
    ax.errorbar(m, y + summary_offset, xerr=[[m-lo], [hi-m]], fmt=marker,
                color=color, markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=.45, ms=4.8, capsize=2.0,
                elinewidth=.9, zorder=4)
    return m, lo, hi


def bootstrap_raincloud_h(ax, draws, observed, low, high, y, color, *,
                          width=.28, seed=0, marker="o"):
    """Raincloud for a bootstrap distribution with the reported estimate/CI.

    Bootstrap draws form the cloud only; the large marker and error bar retain
    the manuscript's original inferential estimate and confidence interval.
    """
    draws = np.asarray(pd.Series(draws).dropna(), float)
    draws = draws[np.isfinite(draws)]
    if len(np.unique(draws)) > 1:
        violin = ax.violinplot(draws, positions=[y], vert=False,
                               widths=width * 1.55, showmeans=False,
                               showmedians=False, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.16)
        path = body.get_paths()[0]
        path.vertices[:, 1] = np.maximum(path.vertices[:, 1], y)
    rng = np.random.default_rng(seed)
    n_show = min(170, len(draws))
    show = rng.choice(draws, n_show, replace=False)
    jitter = rng.uniform(-width * .44, -width * .08, n_show)
    ax.scatter(show, y + jitter, s=4.0, color=color, alpha=.13,
               edgecolor="none", rasterized=True, zorder=2)
    ax.errorbar(observed, y + .07,
                xerr=[[observed-low], [high-observed]], fmt=marker,
                color=color, markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=.45, ms=4.8, capsize=2.0,
                elinewidth=.9, zorder=4)


def fmt_p(p: float) -> str:
    """Unified significance symbols used throughout the main figures."""
    if p < .001:
        return "***"
    if p < .005:
        return "**"
    if p < .05:
        return "*"
    return "n.s."


def fmt_bf(value: float, inverse: bool = False) -> str:
    tag = r"BF$_{01}$" if inverse else r"BF$_{10}$"
    if value < 100:
        return f"{tag} = {value:.2f}"
    e = int(math.floor(math.log10(value)))
    m = value / 10 ** e
    return rf"{tag} = {m:.2g}$\times10^{{{e}}}$"


def bracket(ax, x1, x2, y, text, height=0.012, fontsize=5.0) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            color="#333333", lw=0.65, clip_on=False)
    ax.text((x1 + x2) / 2, y + height * 1.25, text, ha="center", va="bottom",
            fontsize=fontsize, color="#222222", clip_on=False)


def save(fig: plt.Figure, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    fig.savefig(png, facecolor="white")
    for ext in (".pdf", ".svg"):
        try:
            fig.savefig(OUT / f"{stem}{ext}", facecolor="white")
        except OSError as exc:  # e.g. the file is open in a viewer on Windows
            print(f"  warn: could not write {stem}{ext} ({exc})")
    plt.close(fig)
    return png


def build_figure1() -> Path:
    """A results-free visual overview of the four behavioral experiments."""
    tr = pd.read_csv(ORIG_DATA / "f1a_example_trial.csv")
    pts = tr[["x", "y"]].to_numpy(float)
    groups = tr.cluster_id.to_numpy(int)
    ptr = tr[["ptr_x", "ptr_y"]].iloc[0].to_numpy(float)
    radius = float(tr.reveal_radius.iloc[0])

    fig = plt.figure(figsize=(7.08, 3.55))
    gs = gridspec.GridSpec(
        1, 4, figure=fig, width_ratios=[0.92, 1.12, 1.12, 1.58],
        left=.035, right=.99, top=.92, bottom=.10, wspace=.31,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    blue, orange = COL["model"], COL["comparator"]
    teal, rose = "#2A9D8F", "#C96C7C"
    group_cols = np.array([blue, orange, teal])

    def card(ax, x, y, w, h, *, fc="white", ec="#B8B8B8", lw=.7, radius=.018):
        p = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(p)
        return p

    def map_points(box):
        x, y, w, h = box
        sx = w * .88 / 800.; sy = h * .82 / 500.; s = min(sx, sy)
        ox = x + (w - 800*s)/2; oy = y + (h - 500*s)/2
        X = ox + pts[:, 0]*s
        Y = oy + (500 - pts[:, 1])*s
        return X, Y, ox + ptr[0]*s, oy + (500-ptr[1])*s, s

    def dots(ax, box, *, color="#303030", alpha=1, size=7, grouped=False, aperture=False):
        X, Y, px, py, s = map_points(box)
        if aperture:
            # Small aperture circle centred on the array.  The panel axes are
            # not square, so measure the true display scale through the axes
            # transform and compensate so the aperture is a circle on screen;
            # visible points are assigned with the same screen-space geometry.
            R = 120.0
            ax0, ay0 = ax.transData.transform((0, 0))
            ax1, _ = ax.transData.transform((1, 0))
            _, ay1 = ax.transData.transform((0, 1))
            ratio_v = abs((ay1 - ay0) / (ax1 - ax0))   # |pixels per data-y / data-x|
            ax.add_patch(patches.Ellipse((px, py), 2*R*s, 2*R*s/ratio_v,
                                         facecolor="white", edgecolor="#777777",
                                         linewidth=.65, linestyle=(0, (3, 2)), zorder=1))
            d = np.hypot(pts[:, 0]-ptr[0], (ptr[1]-pts[:, 1])*ratio_v)
            visible = d <= R
            ax.scatter(X[~visible], Y[~visible], s=size, color="#D4D4D4",
                       edgecolor="white", linewidth=.25, zorder=2)
            ax.scatter(X[visible], Y[visible], s=size, color="#303030",
                       edgecolor="white", linewidth=.25, zorder=3)
            ax.plot(px, py, marker="o", ms=3.2, mfc="white", mec="#222222", mew=.7, zorder=4)
        elif grouped:
            for cid in (1, 2, 3):
                m = groups == cid
                ax.scatter(X[m], Y[m], s=size, color=group_cols[cid-1],
                           edgecolor="white", linewidth=.25, zorder=3)
        else:
            ax.scatter(X, Y, s=size, color=color, alpha=alpha,
                       edgecolor="white", linewidth=.25, zorder=3)
        return X, Y

    def flow_arrow(left_ax, right_ax):
        p0, p1 = left_ax.get_position(), right_ax.get_position()
        fig.add_artist(patches.FancyArrowPatch(
            (p0.x1 + .004, (p0.y0+p0.y1)/2),
            (p1.x0 - .004, (p1.y0+p1.y1)/2),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
            linewidth=.65, color="#6F6F6F",
        ))

    # a | fixed unstructured arrays.
    ax = axes[0]
    panel(ax, "a", x=.01, y=1.04)
    ax.text(.5, .99, "Stimulus set", ha="center", va="top", fontsize=8, fontweight="bold")
    box = (.10, .44, .80, .34)
    card(ax, *box, fc="#FAFAFA")
    dots(ax, box, size=8)
    ax.text(.5, .36, "10–40 points", ha="center", fontsize=7)

    # b | visibility manipulation.
    ax = axes[1]
    panel(ax, "b", x=.01, y=1.04)
    ax.text(.5, .99, "Visibility", ha="center", va="top", fontsize=8, fontweight="bold")
    for y, label, ap, fc in [(.57, "Full view", False, "#F3F8FB"),
                             (.14, "Local aperture", True, "#F6F6F6")]:
        ax.text(.5, y+.30, label, ha="center", va="bottom", fontsize=7,
                color=blue if not ap else "#555555")
        box = (.08, y, .84, .27)
        card(ax, *box, fc=fc, ec=blue if not ap else "#8A8A8A")
        dots(ax, box, size=5.5, aperture=ap)

    # c | response grammar.
    ax = axes[2]
    panel(ax, "c", x=.01, y=1.04)
    ax.text(.5, .99, "Grouping response", ha="center", va="top", fontsize=8, fontweight="bold")
    for y, label, kind, edge in [(.57, "Lasso: draw loops", "lasso", blue),
                                 (.14, "Anchor: place centres", "anchor", orange)]:
        ax.text(.5, y+.30, label, ha="center", va="bottom", fontsize=7, color=edge)
        box = (.08, y, .84, .27)
        card(ax, *box, fc="white", ec=edge)
        X, Y = dots(ax, box, size=5.5, grouped=True)
        if kind == "lasso":
            for cid in (1, 2, 3):
                m = groups == cid
                cx, cy = X[m].mean(), Y[m].mean()
                ww = max(X[m].max()-X[m].min(), .05) * 1.20
                hh = max(Y[m].max()-Y[m].min(), .04) * 1.25
                ax.add_patch(patches.Ellipse((cx, cy), ww, hh, fill=False,
                                             edgecolor=edge, linewidth=.75))
        else:
            for cid in (1, 2, 3):
                m = groups == cid
                ax.scatter([X[m].mean()], [Y[m].mean()], marker="D", s=17,
                           color=edge, edgecolor="white", linewidth=.3, zorder=5)

    # d | crossed design and cohort sizes.
    ax = axes[3]
    panel(ax, "d", x=.01, y=1.04)
    ax.text(.54, .99, "Four independent cohorts", ha="center", va="top",
            fontsize=8, fontweight="bold")
    ax.text(.57, .91, "Lasso", ha="center", fontsize=6.5, color=blue)
    ax.text(.86, .91, "Anchor", ha="center", fontsize=6.5, color=orange)
    ax.text(.10, .69, "Full", ha="left", va="center", fontsize=6.5)
    ax.text(.10, .35, "Aperture", ha="left", va="center", fontsize=6.5)
    cells = [
        (.42, .56, blue, "Exp. 1", "N = 67"),
        (.71, .56, orange, "Exp. 3", "N = 67"),
        (.42, .22, blue, "Exp. 2", "N = 71"),
        (.71, .22, orange, "Exp. 4", "N = 73"),
    ]
    for x, y, edge, exp, n in cells:
        card(ax, x, y, .25, .25, fc="white", ec=edge, lw=.9)
        ax.add_patch(patches.Rectangle((x, y+.205), .25, .045, facecolor=edge,
                                       edgecolor="none", alpha=.20))
        ax.text(x+.125, y+.218, exp, ha="center", va="center", fontsize=6.3, fontweight="bold")
        ax.text(x+.125, y+.135, n, ha="center", fontsize=6)

    for left, right in zip(axes[:-1], axes[1:]):
        flow_arrow(left, right)
    return save(fig, "figure1_design")


def build_figure2_reliability() -> Path:
    """Preserve the original Figure 1 results in a cleaner evidence figure."""
    stim = pd.read_csv(ORIG_DATA / "stim_coords.csv")
    partitions = pd.read_csv(ORIG_DATA / "example_partitions.csv")
    reliability = pd.read_csv(RESULTS / "behavioral_reliability.csv")
    contrasts = pd.read_csv(RESULTS / "behavioral_reliability_contrasts.csv")

    fig = plt.figure(figsize=(7.08, 3.15))
    outer = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[.82, 1.72, 2.35],
        left=.055, right=.992, top=.90, bottom=.22, wspace=.32,
    )

    def array_axes(ax):
        ax.set_xlim(-25, 825); ax.set_ylim(525, -25); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#B7B7B7"); spine.set_linewidth(.55)

    # a | the same structureless stimulus used for all illustrative partitions.
    ax = fig.add_subplot(outer[0, 0])
    array_axes(ax)
    ax.scatter(stim.x, stim.y, s=10, color="#333333", edgecolor="white", linewidth=.3)
    panel(ax, "a", x=-.12, y=1.12)
    ax.text(.5, 1.05, "Example random array", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7)

    # b | six human partitions of the same array.
    inner = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0, 1],
                                             hspace=.20, wspace=.16)
    sids = list(pd.unique(partitions.sid))[:6]
    for i, sid in enumerate(sids):
        ax = fig.add_subplot(inner[i//3, i%3])
        array_axes(ax)
        p = partitions.query("sid == @sid")
        for cid, g in p.groupby("cluster_id"):
            color = [COL["model"], COL["comparator"], "#2A9D8F", "#C96C7C", "#7B3FA0"][
                (int(cid)-1) % 5
            ]
            ax.scatter(g.x, g.y, s=8, color=color, edgecolor="white", linewidth=.25)
        ax.text(.03, .96, f"k = {int(p.n_clusters.iloc[0])}", transform=ax.transAxes,
                ha="left", va="top", fontsize=5.2, color="#333333")
        if i == 0:
            panel(ax, "b", x=-.12, y=1.12)
    fig.text(.405, .925, "Six participants, same array", ha="center", va="bottom", fontsize=7)

    # c | paired participant-level internal and inter-participant agreement.
    ax = fig.add_subplot(outer[0, 2])
    positions = [(0, .72), (1.65, 2.37), (3.30, 4.02), (4.95, 5.67)]
    for idx, (exp, (xi, xe)) in enumerate(zip(["exp1", "exp2", "exp3", "exp4"], positions)):
        g = reliability.query("experiment == @exp and metric == 'FM'").set_index("agreement")
        internal = g.loc["internal"]
        inter = g.loc["interparticipant"]
        ax.plot([xi, xe], [internal["mean"], inter["mean"]], color="#AFAFAF", lw=.65, zorder=1)
        for x, row, color in [(xi, internal, "#111111"), (xe, inter, "#777777")]:
            ax.errorbar(x, row["mean"],
                        yerr=[[row["mean"]-row.ci_low], [row.ci_high-row["mean"]]],
                        fmt="o", ms=5.0, mfc="white", mec=color, mew=.8,
                        color=color, capsize=2, elinewidth=.8, zorder=4)
        p = contrasts.query("experiment == @exp and metric == 'FM'").p_two_sided.iloc[0]
        top = max(internal.ci_high, inter.ci_high) + .012
        bracket(ax, xi, xe, top, fmt_p(p), height=.012, fontsize=4.6)
        ax.text((xi+xe)/2, .415, f"Exp. {idx+1}", ha="center", va="bottom",
                fontsize=5.8, color="#444444")
    ax.set_xlim(-.35, 6.02); ax.set_ylim(.55, .82)
    ax.set_xticks([p for pair in positions for p in pair],
                  [lab for _ in positions for lab in ("Within", "Across")],
                  rotation=32, ha="right", fontsize=5.4)
    ax.set_ylabel("Fowlkes–Mallows agreement")
    panel(ax, "c")
    return save(fig, "figure2_reliability")


def build_figure2() -> Path:
    # Final Figure 3 typography: match Figures 4–5 in every export format.
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.2,
        "axes.labelsize": 6.2,
        "axes.titlesize": 7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5,
        "savefig.dpi": 400,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    from revision_analysis.behavioral_core import _frame
    from revision_analysis.common import EXPERIMENTS, load_dataset, ols_slope

    kc = pd.read_csv(ORIG_DATA / "f2_nclusters_by_np.csv")
    nz = pd.read_csv(ORIG_DATA / "f2_numerosity_by_np.csv")
    rt = pd.read_csv(ORIG_DATA / "f2_retest_nclusters.csv")
    con = pd.read_csv(RESULTS / "behavioral_slope_contrasts.csv")
    rel = pd.read_csv(RESULTS / "behavioral_reliability.csv")
    behavior = _frame(load_dataset(ROOT / "data" / "processed_trials.npz"))
    participant_rows = []
    for exp in EXPERIMENTS:
        z = behavior.query("exp == @exp")
        for outcome in ("k", "mean_size"):
            values = z.groupby("sid").apply(
                lambda g: ols_slope(g.n_points, g[outcome]),
                include_groups=False).dropna()
            participant_rows.extend(
                {"experiment": exp, "outcome": outcome, "slope": value}
                for value in values.to_numpy(float))
    participant = pd.DataFrame(participant_rows)

    exps = ["exp1", "exp2", "exp3", "exp4"]
    colors = {"exp1": COL["lasso"], "exp2": COL["lasso"],
              "exp3": COL["anchor"], "exp4": COL["anchor"]}
    styles = {"exp1": "-", "exp2": (0, (4, 2)), "exp3": "-", "exp4": (0, (4, 2))}
    markers = {"exp1": "o", "exp2": "s", "exp3": "o", "exp4": "s"}

    fig = plt.figure(figsize=(7.08, 6.15))
    gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[1, 1.03],
                           hspace=0.50, wspace=0.88,
                           left=0.155, right=0.985, top=0.965, bottom=0.105)

    def slope_plot(ax, frame, ylabel, letter):
        for e in exps:
            g = frame.query("exp == @e").sort_values("n_points")
            filled = "white" if e in ("exp2", "exp4") else colors[e]
            ax.errorbar(g.n_points, g["mean"], yerr=g.se, color=colors[e],
                        ls=styles[e], marker=markers[e], ms=3.0, lw=0.9,
                        capsize=1.5, elinewidth=0.6, mfc=filled, mec=colors[e], mew=0.7)
        ax.set_xlabel("Number of points")
        ax.set_ylabel(ylabel)
        ax.set_xticks([10, 20, 30, 40])
        panel(ax, letter)

    # a | reproducible organization. Only audited aggregate CIs are retained
    # in the output table, so this panel does not invent a pseudo-raincloud.
    ax = fig.add_subplot(gs[0, 0:2])
    centers = np.arange(4)[::-1]
    for i, (e, y) in enumerate(zip(exps, centers)):
        a = rel.query("experiment == @e and metric == 'FM' and agreement == 'internal'").iloc[0]
        b = rel.query("experiment == @e and metric == 'FM' and agreement == 'interparticipant'").iloc[0]
        ax.plot([a["mean"], b["mean"]], [y + .13, y - .13],
                color="#8F8F8F", lw=.9, alpha=.75, zorder=0)
        for row, yy0, color in [(a, y+.13, "#222222"), (b, y-.13, "#8C8C8C")]:
            ax.errorbar(row["mean"], yy0,
                        xerr=[[row["mean"]-row.ci_low], [row.ci_high-row["mean"]]],
                        fmt="o", ms=4.7, mfc="white", mec=color, color=color,
                        mew=.8, capsize=1.8, elinewidth=.8, zorder=4)
    ax.set_yticks(centers, ["Exp. 1", "Exp. 2", "Exp. 3", "Exp. 4"])
    ax.set_xlim(.35, .91)
    ax.set_xlabel("Fowlkes–Mallows agreement")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color="#222222",
                              label="Within person"),
                       Line2D([0], [0], marker="o", ls="", color="#8C8C8C",
                              label="Across people")],
              loc="upper left", bbox_to_anchor=(0, 1.016),
              borderaxespad=0, fontsize=5.5)
    panel(ax, "a", x=-.22)

    ax = fig.add_subplot(gs[0, 2:4])
    slope_plot(ax, kc, "Number of clusters", "b")
    ax.legend(handles=[
        Line2D([0], [0], color=COL["lasso"], marker="o", label="L/F"),
        Line2D([0], [0], color=COL["lasso"], marker="s", mfc="white", ls=(0, (4, 2)), label="L/A"),
        Line2D([0], [0], color=COL["anchor"], marker="o", label="A/F"),
        Line2D([0], [0], color=COL["anchor"], marker="s", mfc="white", ls=(0, (4, 2)), label="A/A"),
    ], loc="upper left", ncol=1, fontsize=5.5, handlelength=1.8)

    ax = fig.add_subplot(gs[0, 4:6])
    slope_plot(ax, nz, "Mean cluster size", "c")

    ax = fig.add_subplot(gs[1, 0:3])
    order = [
        ("k", "anchor minus lasso under full view", "Count · A/F − L/F"),
        ("k", "anchor minus lasso under funnel view", "Count · A/A − L/A"),
        ("k", "funnel minus full within lasso", "Count · L/A − L/F"),
        ("k", "funnel minus full within anchor", "Count · A/A − A/F"),
        ("mean_size", "anchor minus lasso under full view", "Size · A/F − L/F"),
        ("mean_size", "anchor minus lasso under funnel view", "Size · A/A − L/A"),
        ("mean_size", "funnel minus full within lasso", "Size · L/A − L/F"),
        ("mean_size", "funnel minus full within anchor", "Size · A/A − A/F"),
    ]
    ypos = np.arange(len(order))[::-1]
    for row_i, (y, (outcome, contrast, _)) in enumerate(zip(ypos, order)):
        r = con.query("outcome == @outcome and contrast == @contrast").iloc[0]
        is_format = "anchor minus lasso" in contrast
        if is_format:
            c = COL["human"] if outcome == "k" else COL["red"]
        else:
            c = "#7F7F7F" if outcome == "k" else "#AAAAAA"
        marker = "o" if outcome == "k" else "D"
        group1, group2 = str(r.group1), str(r.group2)
        a = participant.query("experiment == @group1 and outcome == @outcome").slope.to_numpy(float)
        b = participant.query("experiment == @group2 and outcome == @outcome").slope.to_numpy(float)
        boot_rng = np.random.default_rng(300 + row_i)
        draws = (boot_rng.choice(a, size=(3000, len(a)), replace=True).mean(axis=1)
                 - boot_rng.choice(b, size=(3000, len(b)), replace=True).mean(axis=1))
        bootstrap_raincloud_h(ax, draws, r.mean_difference, r.ci_low, r.ci_high,
                              y, c, width=.30, seed=350+row_i, marker=marker)
        evidence = fmt_p(r.p_two_sided)
        ax.text(0.160, y, evidence, fontsize=5.5, va="center", ha="left")
    ax.axhline(3.5, color="#D7D7D7", lw=.7, zorder=0)
    ax.axvline(0, color="#999999", lw=0.7)
    ax.set_yticks(ypos, [x[2] for x in order], fontsize=5.5)
    ax.set_xlim(-0.105, 0.255)
    ax.set_xlabel("Difference in per-point slope (95% CI)")
    panel(ax, "d", x=-0.16)

    ax = fig.add_subplot(gs[1, 3:6])
    rng = np.random.default_rng(3)
    for e, name in [("exp1", "Lasso"), ("exp3", "Anchor")]:
        g = rt.query("exp == @e")
        jx = g.p1 * np.exp(rng.uniform(-0.055, 0.055, len(g)))
        jy = g.p2 * np.exp(rng.uniform(-0.055, 0.055, len(g)))
        r = np.corrcoef(g.p1, g.p2)[0, 1]
        ax.scatter(jx, jy, s=7, color=COL["lasso"] if e == "exp1" else COL["anchor"],
                   alpha=.40, edgecolor="white", lw=.25, label=name)
    lim = [1.7, 27]
    ax.plot(lim, lim, color="#999999", lw=.7, ls=(0, (3, 2)))
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xticks([2, 4, 8, 16, 24]); ax.set_yticks([2, 4, 8, 16, 24])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Median clusters — presentation 1")
    ax.set_ylabel("Presentation 2")
    ax.legend(loc="lower right", fontsize=5.5)
    panel(ax, "e")
    return save(fig, "figure3_scaling")


def build_figure3() -> Path:
    matched = pd.read_csv(RESULTS / "matched_k_ideal_all.csv")
    minf = pd.read_csv(RESULTS / "matched_k_ideal_all_inference.csv")
    search = pd.read_csv(RAW_RESULTS / "funnel_search_trials.csv")
    headline = pd.read_csv(RESULTS / "headline_gap_inference.csv")
    stop_pred = pd.read_csv(RAW_RESULTS / "stopping_oof_predictions.csv")
    stop_metrics = pd.read_csv(RESULTS / "stopping_model_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.70))
    fig.subplots_adjust(left=.075, right=.99, top=.91, bottom=.22, wspace=.42)
    rng = np.random.default_rng(17)

    # a | expose the base-stimulus distribution rather than hiding it in bars.
    ax = axes[0]
    base = matched.groupby(["exp", "base_uuid"], as_index=False).silhouette_gap.mean()
    x = np.arange(4)
    colors = [COL["lasso"], COL["lasso"], COL["anchor"], COL["anchor"]]
    for i, (e, color) in enumerate(zip(LABEL, colors)):
        values = base.query("exp == @e").silhouette_gap.to_numpy(float)
        raincloud_v(ax, values, i, color, width=.34, seed=100+i,
                    alpha=.20, summary_offset=.075)
    ax.axhline(0, color="#999999", lw=.55)
    ax.set_xticks(x, ["L/F", "L/A", "A/F", "A/A"])
    ax.set_ylabel("Best-found − human silhouette")
    overall = minf.query("metric == 'silhouette'").iloc[0]
    ax.text(.02, .98, fmt_p(overall.p_two_sided), transform=ax.transAxes,
            va="top", fontsize=5.6)
    panel(ax, "a")

    # b | paired stimulus-level search costs, with hollow means and 95% CIs.
    ax = axes[1]
    g = search.groupby(["exp", "base_uuid"], as_index=False)[
        ["human_actions_all", "oracle_actions_all"]
    ].mean()
    exps = ["exp2", "exp4"]
    pair_positions = [(0, 1), (3, 4)]
    max_hi = 0.0
    for e, (xh, xo) in zip(exps, pair_positions):
        d = g.query("exp == @e").sort_values("base_uuid")
        human = d.human_actions_all.to_numpy(float)
        oracle = d.oracle_actions_all.to_numpy(float)
        for yh, yo in zip(human, oracle):
            ax.plot([xh, xo], [yh, yo], color="#B8B8B8", lw=.35, alpha=.28, zorder=0)
        hm = raincloud_v(ax, human, xh, COL["human"], width=.31,
                         seed=120+xh, alpha=.18, summary_offset=.070)
        om = raincloud_v(ax, oracle, xo, COL["comparator"], width=.31,
                         seed=130+xo, alpha=.20, summary_offset=.070)
        max_hi = max(max_hi, hm[2], om[2])
        r = headline.query(
            "analysis == 'funnel_policy' and condition == @e and estimand.str.contains('actions')",
            engine="python",
        ).iloc[0]
        bracket(ax, xh, xo, max(hm[2], om[2]) + .28, fmt_p(r.p_two_sided),
                height=.10, fontsize=6)
    ax.set_xticks([0, 1, 3, 4], ["Human", "Oracle", "Human", "Oracle"])
    ax.text(.125, -.18, "Lasso / aperture", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.0)
    ax.text(.875, -.18, "Anchor / aperture", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.0)
    ax.set_ylabel("Actions to reveal all points")
    ax.set_xlim(-.55, 4.55)
    ax.set_ylim(0, max_hi + 1.05)
    panel(ax, "b")

    # c | paired OOF losses by base stimulus.
    ax = axes[2]
    eps = 1e-12
    for model in ["process", "linear_quality", "threshold"]:
        p = np.clip(stop_pred[model].to_numpy(float), eps, 1-eps)
        y = stop_pred.stop.to_numpy(float)
        stop_pred[model + "_loss"] = -(y*np.log(p) + (1-y)*np.log(1-p))
    base_loss = stop_pred.groupby("base_uuid")[[m+"_loss" for m in ["process", "linear_quality", "threshold"]]].mean()
    order = ["process", "linear_quality", "threshold"]
    point_colors = [COL["process"], COL["quality"], COL["threshold"]]
    for _, row in base_loss.iterrows():
        ax.plot(np.arange(3), [row[m + "_loss"] for m in order],
                color="#B8B8B8", lw=.35, alpha=.22, zorder=0)
    means = []
    intervals = []
    for i, (model, color) in enumerate(zip(order, point_colors)):
        values = base_loss[model + "_loss"].to_numpy(float)
        ci = raincloud_v(ax, values, i, color, width=.31, seed=150+i,
                         alpha=.16, summary_offset=.070)
        means.append(ci[0]); intervals.append(ci)
    means = np.asarray(means)
    half = np.asarray([v[2] - v[0] for v in intervals])
    ax.set_xticks(np.arange(3), ["Process", "Quality\nadded", "Threshold\nadded"])
    ax.set_ylabel("Stimulus-grouped OOF log loss")
    ax.set_ylim(base_loss.min().min() - .003, base_loss.max().max() + .010)
    r1 = headline.query("analysis == 'stopping_oof' and estimand.str.contains('linear_quality over process')", engine="python").iloc[0]
    r2 = headline.query("analysis == 'stopping_oof' and estimand.str.contains('threshold over linear_quality')", engine="python").iloc[0]
    y0 = base_loss.max().max() + .001
    bracket(ax, 0, 1, y0, fmt_p(r1.p_two_sided), height=.0007, fontsize=6)
    bracket(ax, 1, 2, y0+.004, fmt_p(r2.p_two_sided), height=.0007, fontsize=6)
    panel(ax, "c")

    return save(fig, "figure4_search_stopping")


def build_figure4() -> Path:
    oof = pd.read_csv(RESULTS / "neural_oof_summary.csv")
    inf = pd.read_csv(RESULTS / "neural_slope_inference.csv")
    conditions = oof.query("scope != 'pooled'").copy()

    fig = plt.figure(figsize=(7.08, 5.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.45, 1.2],
                           left=.07, right=.985, top=.96, bottom=.115,
                           wspace=.42, hspace=.42)

    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    original = plt.imread(MODEL_H_SKETCH)
    rgb = original[..., :3]
    nonwhite = np.any(rgb < .985, axis=2)
    yy, xx = np.where(nonwhite)
    pad = 18
    original = original[max(0, yy.min()-pad):min(original.shape[0], yy.max()+pad),
                        max(0, xx.min()-pad):min(original.shape[1], xx.max()+pad)]
    ax.imshow(original)
    ax.set_position([.06, .52, .84, .48])

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(4)
    ax.plot(x, conditions.ARI_vs_human, color=COL["model"], lw=.9,
            marker="o", ms=4.0, mfc="white", mec=COL["model"], mew=.8,
            label="ARI")
    ax.plot(x, conditions.FM_vs_human, color=COL["quality"], lw=.9,
            marker="o", ms=4.0, mfc="white", mec=COL["quality"], mew=.8,
            label="FM")
    ax.set_xticks(x, ["L/F", "L/A", "A/F", "A/A"])
    ax.set_ylim(.30, .72)
    ax.set_ylabel("Agreement with held-out human partition")
    ax.legend(loc="lower right", fontsize=6.0)
    panel(ax, "b")

    def slope_forest(ax, outcome, letter):
        d = inf.query("outcome == @outcome").set_index("experiment").loc[["exp1","exp2","exp3","exp4"]]
        yy = np.arange(4)[::-1]
        row_colors = [COL["lasso"], COL["lasso"], COL["anchor"], COL["anchor"]]
        if outcome == "k":
            ax.set_xlim(-.012, .255)
            p_x = .112
        else:
            ax.set_xlim(-.105, .185)
            p_x = .055
        for y, ((_, r), color) in zip(yy, zip(d.iterrows(), row_colors)):
            ax.errorbar(r.mean_difference, y,
                        xerr=[[r.mean_difference-r.ci_low], [r.ci_high-r.mean_difference]],
                        fmt="o", ms=4.5, capsize=2, color=color,
                        mfc="white", mec="#111111", mew=.75, elinewidth=.8)
            ev = fmt_p(r.p_two_sided)
            ax.text(p_x, y, ev, va="center", ha="left", fontsize=6)
        ax.axvline(0, color="#888888", lw=.7)
        ax.set_yticks(yy, ["L/F", "L/A", "A/F", "A/A"])
        ax.set_xlabel("Human − Model-H slope (95% CI)")
        panel(ax, letter)

    ax = fig.add_subplot(gs[1, 1]); slope_forest(ax, "k", "c")
    ax = fig.add_subplot(gs[1, 2]); slope_forest(ax, "mean_size", "d")
    return save(fig, "figure5_model_h")


def build_figure5() -> Path:
    perf = pd.read_csv(RESULTS / "neural_control_performance.csv")
    cinf = pd.read_csv(RESULTS / "neural_control_inference.csv")
    cond = pd.read_csv(RESULTS / "neural_control_condition_summary.csv")
    slopes = pd.read_csv(RESULTS / "behavioral_population_slopes.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.08, 4.65))
    fig.subplots_adjust(left=.09, right=.985, top=.96, bottom=.105,
                        wspace=.34, hspace=.44)

    # a | participant-equal estimates, without bar area.
    ax = axes[0, 0]
    fams = ["condition-aware Model-H", "label permutation"]
    xx = np.arange(2)
    offsets = {"ARI": -.10, "FM": .10}
    tops = []
    for j, metric in enumerate(["ARI", "FM"]):
        d = perf.query("family in @fams and metric == @metric").set_index("family").loc[fams]
        y = d["mean"].to_numpy()
        se = d.sd.to_numpy() / np.sqrt(d.n_participants.to_numpy())
        pos = xx + offsets[metric]
        color = COL["model"] if metric == "ARI" else COL["quality"]
        ax.plot(pos, y, color=color, lw=.8, zorder=2)
        ax.errorbar(pos, y, yerr=se, fmt="o", ms=4.8, mfc="white",
                    mec=color, mew=.9, color=color, capsize=2,
                    elinewidth=.75, label=metric, zorder=3)
        tops.append(y + se)
    r = cinf.query("contrast == 'human labels vs permuted labels' and metric == 'ARI'").iloc[0]
    bracket(ax, xx[0] + offsets["ARI"], xx[1] + offsets["ARI"],
            max(np.r_[tops[0], tops[1]]) + .018,
            fmt_p(r.p_two_sided), height=.010, fontsize=6)
    ax.set_xticks(xx, ["Model-H", "Permuted labels"])
    ax.set_ylim(.10, .71)
    ax.set_ylabel("Participant-equal agreement ± SE")
    ax.legend(loc="lower left", fontsize=6.0)
    panel(ax, "a")

    # b | architecture pairs are linked, making the +count manipulation clear.
    ax = axes[0, 1]
    fams = ["meanmax", "meanmax + count", "dpool", "dpool + count", "attn", "attn + count"]
    d = perf.query("family in @fams and metric == 'ARI'").set_index("family").loc[fams]
    y = d["mean"].to_numpy()
    se = d.sd.to_numpy() / np.sqrt(d.n_participants.to_numpy())
    x = np.array([0, 1, 2.7, 3.7, 5.4, 6.4])
    group_colors = [COL["process"], COL["process"], COL["model"], COL["model"],
                    COL["quality"], COL["quality"]]
    for a, b, color in [(0, 1, COL["process"]), (2, 3, COL["model"]),
                        (4, 5, COL["quality"])]:
        ax.plot(x[[a, b]], y[[a, b]], color=color, lw=.8, zorder=1)
    for xi, yi, sei, color in zip(x, y, se, group_colors):
        ax.errorbar(xi, yi, yerr=sei, fmt="o", ms=4.7, mfc="white",
                    mec=color, mew=.9, color=color, capsize=1.8,
                    elinewidth=.7, zorder=3)
    ax.set_xticks(x, ["Mean–max", "+ count", "D-pool", "+ count", "Attention", "+ count"],
                  rotation=22, ha="right")
    ax.set_ylim(.445, .515)
    ax.set_ylabel("Participant-equal ARI ± SE")
    r1 = cinf.query("contrast == 'architecture: dpool vs meanmax' and metric == 'ARI'").iloc[0]
    r2 = cinf.query("contrast == 'architecture: meanmax + count vs meanmax' and metric == 'ARI'").iloc[0]
    bracket(ax, x[0], x[2], .499, fmt_p(r1.p_two_sided), height=.0016, fontsize=6)
    bracket(ax, x[0], x[1], .507, fmt_p(r2.p_two_sided), height=.0016, fontsize=6)
    panel(ax, "b")

    doses = [0, .5, 1, 2, 4]
    capacity_fams = [f"capacity {v:g}" for v in doses]
    d = perf.query("family in @capacity_fams and metric == 'ARI'").set_index("family").loc[capacity_fams]
    y = d["mean"].to_numpy()
    se = d.sd.to_numpy() / np.sqrt(d.n_participants.to_numpy())

    # c | ARI dose response on its own axis.
    ax = axes[1, 0]
    ax.errorbar(doses, y, yerr=se, marker="o", ms=4.8, mfc="white",
                mec=COL["model"], mew=.9, lw=.9, capsize=2,
                color=COL["model"])
    ax.set_xlabel("Occupancy-penalty weight")
    ax.set_ylabel("Participant-equal ARI ± SE")
    ax.set_ylim(.32, .49)
    ax.text(.02, .96, r"Dose vs 0: all $P < 3.7\times10^{-59}$",
            transform=ax.transAxes, va="top", fontsize=6)
    panel(ax, "c")

    # d | count-slope response, separated from ARI to avoid a dual y-axis.
    ax = axes[1, 1]
    slope = []
    slope_se = []
    for fam in capacity_fams:
        z = cond.query("family == @fam")
        slope.append(z.slope_k_vs_n.mean())
        slope_se.append(np.sqrt(np.sum((z.slope_k_seed_sd.to_numpy()/np.sqrt(z.n_runs.to_numpy()))**2))/len(z))
    ax.errorbar(doses, slope, yerr=slope_se, marker="s", ms=4.6,
                mfc="white", mec=COL["comparator"], mew=.9,
                lw=.9, capsize=2, color=COL["comparator"])
    human = slopes.query("outcome == 'k'").slope.mean()
    ax.axhline(human, color="#777777", ls=(0, (3, 2)), lw=.65)
    ax.text(4.05, human, "human mean", ha="right", va="bottom",
            fontsize=8, color="#666666")
    ax.set_xlabel("Occupancy-penalty weight")
    ax.set_ylabel("Mean count slope ± seed SE")
    ax.set_ylim(0, .23)
    ax.text(.02, .96, r"Dose vs 0: all $P \leq 3.4\times10^{-7}$",
            transform=ax.transAxes, va="top", fontsize=6)
    panel(ax, "d")
    return save(fig,"figure6_controls")


CAPTIONS = [
    "Figure 1 | Behavioral experiment design. a, The fixed stimulus set comprised 56 random arrays containing 10–40 points and no planted groups. b, Visibility was manipulated between participants: the whole array remained visible or dots were discovered through a cursor-controlled local aperture after a 1-s dwell. c, Participants expressed each partition either by drawing closed lassos or by placing movable anchors that generated a Voronoi partition; responses could be revised until every dot was assigned. d, The crossed 2 × 2 design yielded four independent cohorts. Full-view cohorts completed 112 trials; aperture cohorts completed 56 trials and rated confidence after each trial. Trial order was randomized, and all conditions used mouse responses.",
    "Figure 2 | The point-information oracle and behavioural reliability. a, Next-dwell gain maps of the oracle on a real aperture trial (four optimal dwells); colour encodes how many still-hidden points each candidate dwell would reveal, the chosen centre is marked with a cross and its reveal radius, and the cumulative revealed count is shown. The oracle knows all point locations by construction, so the maps show next-dwell reveal gain rather than a Bayesian posterior. b, Flow chart of the greedy oracle. c, Within-person test–retest and across-participant Fowlkes–Mallows agreement in each experiment; hollow points and error bars show participant-level means and 95% confidence intervals, connecting lines link the two agreement estimates, and brackets report exact two-sided paired-participant P values. Visual organization follows Najemnik & Geisler (2005), Fig. 3. ARI robustness analyses appear in Supplementary Table 1.",
    "Figure 3 | Behavioral scaling depends on response format more than visibility. a–b, Cluster number and mean cluster size across array size; points are means ± s.e.m. c, Participant-slope contrasts shown as independent-participant bootstrap rainclouds; large markers and intervals retain the reported contrasts and 95% confidence intervals, and exact two-sided P values are shown. Bootstrap dots are resampling draws, not additional participants. d, Test–retest stability of individual grouping granularity. L/F, lasso/full; L/A, lasso/aperture; A/F, anchor/full; A/A, anchor/aperture.",
    "Figure 4 | Endpoint, search and stopping evidence for bounded construction. a, Base-stimulus rainclouds of the matched-K best-found minus human silhouette gap; half violins show the distributions, lower-side dots show the 56 base stimuli, and hollow points and intervals show means and 95% confidence intervals. b, Paired base-stimulus human and point-information-oracle actions required to reveal all points, shown with rainclouds and paired lines. c, Paired base-stimulus out-of-fold stopping-hazard log loss, shown with rainclouds; brackets report paired-model P values. All P values are two-sided.",
    "Figure 5 | Model-H transfers endpoint structure but does not recover the full behavioral profile. a, The original Model-H schematic: point-array input, shared MLP embedding, repeated set-interaction blocks, a reveal-order GRU for aperture conditions and per-point softmax assignment to slots. This pipeline predicts endpoint partitions; it is not a model of candidate generation or stopping. b, Stimulus-level distributions of mean group count for humans, Model-H and automatic k-means, shown as rainclouds with means and 95% confidence intervals. c–d, Participant distributions of human-minus-model differences in cluster-count and cluster-size slopes, shown as rainclouds with participant-based 95% confidence intervals and exact two-sided P values. L/F, lasso/full; L/A, lasso/aperture; A/F, anchor/full; A/A, anchor/aperture.",
    "Figure 6 | Training controls and bounded mechanism tests. a, Participant-equal agreement for Model-H and the label-permutation control on a common held-out split; hollow points and error bars show means ± participant s.e.m., and the bracket reports the paired ARI P value. b, The 3 × 2 architecture comparison; linked points show each base architecture and its count-input variant, error bars are participant s.e.m., and brackets report prespecified paired-contrast P values. c, Occupancy-penalty dose response for participant-equal ARI (s.e.m.). d, The corresponding mean condition-specific cluster-count slope (seed-based s.e.m.); the dashed line is the human mean. Exact two-sided P values are shown for the principal comparisons. Architecture and capacity tests are bounded sensitivity analyses, not substitutes for the primary fivefold out-of-fold Model-H estimate.",
]


def build_figure2_oracle() -> Path:
    """Figure 2: point-information oracle (gain maps + flow chart) + reliability.

    The figure itself lives in figures_v2/scripts/build_figure2_oracle_reliability.py
    (same colouring and layout language as Najemnik & Geisler 2005, Fig. 3, applied
    to this task's oracle).  This wrapper builds it and copies the outputs into the
    canonical figure set, so Fig 2 can be regenerated by this one script.
    """
    import shutil
    import sys as _sys
    src_dir = str(HERE.parent / "figures_v2" / "scripts")
    if src_dir not in _sys.path:
        _sys.path.insert(0, src_dir)
    import build_figure2_oracle_reliability as f2o
    png = f2o.build()                       # figures_v2/outputs/figure2_oracle_reliability.png
    for ext in (".png", ".pdf", ".svg"):
        src = Path(str(png)).with_suffix(ext)
        if src.exists():
            try:
                shutil.copy2(src, OUT / src.name)
            except OSError as exc:          # destination may be open in a viewer
                print(f"  warn: could not copy {src.name} ({exc})")
    return OUT / "figure2_oracle_reliability.png"


def main() -> None:
    apply_style()
    figures = [build_figure1(), build_figure2_oracle(), build_figure2(),
               build_figure3(), build_figure4(), build_figure5()]
    print("Figures:")
    for f in figures:
        print(f"  {f}")
    print(f"Publication figures written to {OUT}")


if __name__ == "__main__":
    main()
