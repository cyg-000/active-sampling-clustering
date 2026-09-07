"""Revised Figure 5: Model-H full diagnostic (a + b/c/d).

Panels
  a  Model-H architecture, 2.5x the previous size, alone on its own row
  b  mean group count by condition: human / Model-H / kmeans_auto   (was d)
  c  cluster-count scaling: human minus Model-H slope by condition   (was e)
  d  cluster-size scaling: human minus Model-H slope by condition    (was f)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "revision_analysis"))
import build_revised_five as brf  # noqa: E402
from revision_analysis.neural_oof_summary import _participant_slopes  # noqa: E402

# The shared figure module sets the house style during import. Reassert the
# export-critical settings afterwards so PNG, editable SVG, and PDF agree.
matplotlib.rcParams.update({
    "font.family": "Arial",
    "savefig.dpi": 400,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
ASSETS = HERE.parent / "assets"
ROOT = Path(__file__).resolve().parents[2] / "revision_analysis"
PKG = ROOT / "revision_analysis"
RESULTS = PKG / "outputs"
TMP = ROOT / "autodl" / "tmp" / "outputs"

COL = brf.COL
panel = brf.panel
fmt_p = brf.fmt_p
raincloud_v = brf.raincloud_v
raincloud_h = brf.raincloud_h
MODEL_H_SKETCH = ASSETS / "model_h_architecture_hi.png"
LABEL_SHORT = ["L/F", "L/A", "A/F", "A/A"]
CONDS = ["exp1", "exp2", "exp3", "exp4"]
AR = 1322.0 / 779.0  # arch image aspect


def save_local(fig, stem: str) -> Path:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}")
    return OUT / f"{stem}.png"


DV = np.array([0.34, 0.46])
A_BLUE, A_TAN, A_GREY, A_OUTB = "#DCE9F5", "#F5E6CC", "#ECECEC", "#CFE0F0"


def _shade(hexc, factor):
    h = hexc.lstrip("#")
    rgb = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return tuple(min(1, c * factor) for c in rgb)


def _slab(ax, x, y, w, h, depth, fc, z=1, lw=.8, ec="#5A5A5A"):
    d = DV * depth
    front = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    top = [(x, y+h), (x+w, y+h), (x+w+d[0], y+h+d[1]), (x+d[0], y+h+d[1])]
    side = [(x+w, y), (x+w, y+h), (x+w+d[0], y+h+d[1]), (x+w+d[0], y+d[1])]
    ax.add_patch(patches.Polygon(top, closed=True, fc=_shade(fc, 1.08), ec=ec, lw=lw, zorder=z))
    ax.add_patch(patches.Polygon(side, closed=True, fc=_shade(fc, .82), ec=ec, lw=lw, zorder=z))
    ax.add_patch(patches.Polygon(front, closed=True, fc=fc, ec=ec, lw=lw, zorder=z+.1))


def _arw(ax, p0, p1, color="#5A5A5A", lw=1.0, style="-|>", z=5, rad=0):
    ax.add_patch(patches.FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=7, lw=lw, color=color,
        zorder=z, connectionstyle=f"arc3,rad={rad}"))


def panel_arch(ax):
    """Native-vector Model-H architecture; no raster artwork is embedded."""
    yb, h = 2.2, 1.8
    _slab(ax, .4, yb, 1.15, h, .55, A_GREY)
    rng = np.random.default_rng(4)
    ax.scatter(.58 + .8*rng.random(9), yb+.22+(h-.44)*rng.random(9),
               s=8, color="#555555", zorder=3)
    ax.text(.975, yb-.3, "Point\narray", ha="center", va="top", fontsize=6.2)

    _slab(ax, 2.1, yb, 1.15, h, .55, A_BLUE)
    for i in range(5):
        yy = yb+.3+i*(h-.6)/4
        ax.plot([2.28, 3.06], [yy, yy], color="#7FA8CC", lw=1.3, zorder=3)
    ax.text(2.675, yb-.3, "Shared\nMLP embed", ha="center", va="top", fontsize=6.2)
    _arw(ax, (1.72, yb+h/2), (2.05, yb+h/2))

    bx = 4.05
    for k in range(3):
        off = DV*(2-k)*.9
        _slab(ax, bx+off[0], yb+off[1], 1.2, h, .55, A_BLUE, z=1+k)
    fcx = bx+.6
    for i in range(5):
        yy = yb+.3+i*(h-.6)/4
        ax.plot([bx+.16, bx+1.04], [yy, yy], color="#7FA8CC", lw=1.3, zorder=4)
    ax.text(fcx, yb-.3, "Set-interaction\nblocks  (× L)", ha="center", va="top", fontsize=6.2)
    agx, agy = fcx, yb+h+1.35
    ax.add_patch(patches.Polygon(
        [(agx-.5, agy), (agx, agy+.42), (agx+.5, agy), (agx, agy-.42)],
        closed=True, fc="#FBEAD1", ec="#B98A3E", lw=.9, zorder=6))
    for yy in (yb+.45, yb+h/2, yb+h-.45):
        _arw(ax, (bx+1.02, yy), (agx-.4, agy-.05), color="#D8BE8C",
             lw=.6, style="-", z=5, rad=-.10)
    _arw(ax, (agx+.15, agy-.4), (bx+1.05, yb+h-.15), color="#B98A3E",
         lw=.9, rad=-.35)
    ax.text(agx-1.2, agy+.5, "self + global pool → point", ha="center",
            va="bottom", fontsize=6.0, color="#8A6A2E")
    _arw(ax, (3.15, yb+h/2), (bx-.02, yb+h/2))

    gx, gyr = 6.55, yb+1.95
    _slab(ax, gx, gyr, 1.25, 1.35, .5, A_TAN, z=2)
    lx, ly = gx+.62, gyr+.68
    ax.add_patch(patches.FancyArrowPatch(
        (lx+.2, ly+.3), (lx-.2, ly+.3), connectionstyle="arc3,rad=1.6",
        arrowstyle="-|>", mutation_scale=7, lw=1.0, color="#B98A3E", zorder=6))
    ax.text(gx+.62, gyr-.28, "GRU over\nreveal order", ha="center", va="top", fontsize=6.2)
    ax.text(gx+.62, gyr+1.78, "masked (funnel)\nexperiments only", ha="center",
            va="bottom", fontsize=6.0, style="italic", color="#8A6A2E")
    _arw(ax, (fcx+.78, yb+h-.12), (gx+.08, gyr+.12), color="#B98A3E", lw=.9, rad=.20)
    _arw(ax, (gx+1.3, gyr+.12), (8.75, yb+h-.12), color="#B98A3E", lw=.9, rad=.20)
    _arw(ax, (fcx+.9, yb+h/2), (8.7, yb+h/2))

    ox = 8.8
    _slab(ax, ox, yb, 1.5, h, .55, A_OUTB, z=3)
    gx0, gy0 = ox+.18, yb+.26
    n_k, n_n = 4, 5
    cw, ch = (1.5-.36)/n_k, (h-.52)/n_n
    assigned = [0, 0, 1, 2, 1]
    slot_colors = ["#1677B7", "#DF9200", "#009E73", "#CC79A7"]
    for i in range(n_n):
        for j in range(n_k):
            color = slot_colors[j] if assigned[i] == j else "white"
            ax.add_patch(patches.Rectangle(
                (gx0+j*cw, gy0+i*ch), cw, ch, facecolor=color,
                edgecolor="#B9C7D6", lw=.4, zorder=4))
    ax.text(ox+.75, yb-.3, "Per-point softmax\n→ K slots", ha="center", va="top", fontsize=6.2)
    _arw(ax, (10.35, yb+h/2), (10.95, yb+h/2))
    ax.text(11.02, yb+h/2, "soft partition\neach point →\none soft slot",
            ha="left", va="center", fontsize=6.2)
    # Crop to the actual architecture so the vector artwork uses the full top row.
    ax.set_xlim(.35, 12.75); ax.set_ylim(1.55, 6.45)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_b(ax):
    human = pd.read_csv(ASSETS / "kn_human.csv")
    model_h = pd.read_csv(ASSETS / "kn_modelh.csv")
    kauto = pd.read_csv(ASSETS / "kn_kauto.csv")
    sources = [(human, "#111111", "Human"),
               (model_h, COL["model"], "Model-H"),
               (kauto, COL["comparator"], "Auto-K")]
    centers = np.arange(4)[::-1]
    offsets = [.22, 0, -.22]
    for ci, cond in enumerate(CONDS):
        for mi, ((frame, color, _), off) in enumerate(zip(sources, offsets)):
            values = frame.loc[frame.condition == cond, "k"].to_numpy(float)
            raincloud_h(ax, values, centers[ci] + off, color, width=.15,
                        seed=210 + ci * 10 + mi, alpha=.12,
                        summary_offset=.035)
    ax.axvline(3.0, color="#BBBBBB", lw=.6, ls=(0, (3, 2)))
    ax.set_yticks(centers, LABEL_SHORT)
    ax.tick_params(axis="y", labelsize=5.5, pad=1)
    ax.set_xlabel("Mean group count", fontsize=6.2)
    ax.set_xlim(1.4, 8.2)
    ax.set_ylim(-.55, 3.55)


def panel_agreement(ax):
    mh = pd.read_csv(TMP / "modelh_perstim_ari.csv")
    lk = pd.read_csv(TMP / "lookers_fast_perstim.csv")
    model_h = mh.ari.to_numpy(float)
    auto_k = lk.kmeans_auto_ari.to_numpy(float)
    given_k = lk.kmeans_ari.to_numpy(float)
    rows = [(model_h, 3, COL["model"], "Model-H"),
            (auto_k, 2, COL["comparator"], "Auto-K k-means"),
            (given_k, 1, "#7B3FA0", "Given-K k-means")]
    for i, (values, y, color, _) in enumerate(rows):
        raincloud_h(ax, values, y, color, width=.30, seed=320+i,
                    alpha=.14, summary_offset=.065)
    ctrl = pd.read_csv(RESULTS / "neural_control_performance.csv")
    r = ctrl.query("family == 'label permutation' and metric == 'ARI'").iloc[0]
    h = 1.96 * r.sd / np.sqrt(r.n_participants)
    ax.axhline(.5, color="#BBBBBB", lw=.65, ls=(0, (3, 2)))
    ax.errorbar(r["mean"], 0, xerr=h, fmt="o", ms=4.5, color="#999999",
                mfc="white", mec="#777777", capsize=2, elinewidth=.8)
    ax.set_yticks([3, 2, 1, 0], ["Model-H", "Auto-K", "Given-K", "Permutation"])
    ax.set_xlim(.05, .72)
    ax.set_xlabel("Held-out ARI", fontsize=6.2)
    ax.tick_params(axis="y", labelsize=5.5, pad=1)


def panel_count_boundary(ax, participant):
    endpoints = pd.read_csv(RESULTS / "stats_endpoint_slopes.csv")
    rows = [("Human", "human", COL["human"]), ("Model-H", "Model-H", COL["model"]),
            ("Auto-K", "kmeans_auto", COL["comparator"]),
            ("Given-K", "kmeans_given_K", "#7B3FA0"),
            ("Greedy", "greedy", "#C44E52")]
    yy = np.arange(len(rows))[::-1]
    offsets = [.24, .08, -.08, -.24]
    marks = [("o", True), ("o", False), ("s", True), ("s", False)]
    for y, (_, key, color) in zip(yy, rows):
        for cond, off, (marker, filled) in zip(CONDS, offsets, marks):
            if key == "human":
                vals = participant.query("experiment == @cond and outcome == 'k'").slope.to_numpy(float)
                m, lo, hi = brf.mean_ci(vals)
                ax.errorbar(m, y+off, xerr=[[m-lo], [hi-m]], fmt=marker, ms=4.0,
                            color=color, mfc=color if filled else "white", mec=color,
                            mew=.8, elinewidth=.75, capsize=1.5)
            else:
                d = endpoints.query("condition == @cond and model == @key")
                if len(d):
                    v = float(d.iloc[0].slope_k_vs_n)
                    ax.plot(v, y+off, marker=marker, ms=4.0, color=color,
                            mfc=color if filled else "white", mec=color, mew=.8, ls="none")
    ax.axvline(0, color="#999999", lw=.65)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=5.5, pad=1)
    ax.set_xlim(-.025, .125)
    ax.set_xlabel("Cluster-count slope (K per point)", fontsize=6.2)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", color="#555555", mfc="#555555", label="L/F"),
        Line2D([0], [0], marker="o", ls="", color="#555555", mfc="white", label="L/A"),
        Line2D([0], [0], marker="s", ls="", color="#555555", mfc="#555555", label="A/F"),
        Line2D([0], [0], marker="s", ls="", color="#555555", mfc="white", label="A/A")],
        loc="upper center", bbox_to_anchor=(.5, 1.18), fontsize=5.5, ncol=2,
        columnspacing=.30, handletextpad=.12)


def slope_forest(ax, outcome, letter, participant):
    inf = pd.read_csv(RESULTS / "neural_slope_inference.csv")
    d = inf.query("outcome == @outcome").set_index("experiment").loc[CONDS]
    yy = np.arange(4)[::-1]
    row_colors = [COL["lasso"], COL["lasso"], COL["anchor"], COL["anchor"]]
    if outcome == "k":
        ax.set_xlim(-.012, .255)
        p_x = .112
    else:
        ax.set_xlim(-.105, .185)
        p_x = .055
    for y, ((_, r), color) in zip(yy, zip(d.iterrows(), row_colors)):
        exp = r.name
        values = (participant.query("experiment == @exp and outcome == @outcome").slope
                  .to_numpy(float) - float(r.model_slope))
        raincloud_h(ax, values, y, color, width=.31,
                    seed=260 + int(exp[-1]) + (0 if outcome == "k" else 20),
                    alpha=.17, summary_offset=.070)
        ev = fmt_p(r.p_two_sided)
        ax.text(.94, y, ev, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=5.5)
    ax.axvline(0, color="#888888", lw=.7)
    ax.set_yticks(yy, LABEL_SHORT)
    ax.tick_params(axis="y", labelsize=6.2, pad=1)
    ax.set_xlabel("Human − Model-H slope", fontsize=6.2)
    panel(ax, letter)


def main() -> Path:
    fig = plt.figure(figsize=(7.08, 6.50))
    participant = _participant_slopes(ROOT / "data" / "processed_trials.npz")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.04, 1.0],
                          left=.108, right=.988, top=.960, bottom=.095,
                          wspace=.40, hspace=.38)

    ax = fig.add_subplot(gs[0, :]); panel_arch(ax); panel(ax, "a", x=-.022)
    ax = fig.add_subplot(gs[1, 0]); panel_agreement(ax); panel(ax, "b", x=-.032)
    ax = fig.add_subplot(gs[1, 1]); panel_b(ax); panel(ax, "c", x=-.028)
    ax = fig.add_subplot(gs[1, 2]); panel_count_boundary(ax, participant); panel(ax, "d", x=-.030)
    ax = fig.add_subplot(gs[1, 3]); slope_forest(ax, "mean_size", "e", participant)

    return save_local(fig, "figure5_model_h_revised")


if __name__ == "__main__":
    p = main()
    print(f"-> {p}")
