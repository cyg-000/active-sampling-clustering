"""Shared NHB figure style: palette, rcParams, layout helpers.

All five figures import from here so the visual system is identical:
one fixed, colourblind-safe entity colour per actor, thin marks,
sans-serif, panel labels in bold lowercase, vector PDF + 300-dpi PNG.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# ---------------------------------------------------------------- units
MM = 1.0 / 25.4          # millimetres -> inches (NHB specifies mm)
COL1 = 88 * MM           # single column width
COL15 = 120 * MM         # 1.5 column
COL2 = 180 * MM          # double column width -- the hard maximum NHB allows

# ---------------------------------------------------------------- type sizes
# NHB: "Use a 5-7 pt sans serif font for standard text labelling."  Every label in
# these figures is drawn through one of these names so the floor cannot drift; TINY
# is 5.0, the smallest the guidelines permit, not 4.3.
FS = {
    "tiny":   5.0,       # in-panel annotations, P values, bracket labels
    "small":  5.5,       # secondary labels, legends in dense panels
    "base":   6.0,       # tick labels, standard legends
    "label":  7.0,       # axis labels, panel headers -- top of the allowed range
    "panel":  8.0,       # bold panel letters (a, b, c) -- Nature house size
}

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- palette
# Colourblind-safe (Wong 2011) entity colours. FIXED across every figure:
# an actor keeps its colour whether it appears in fig 1 or fig 5.
C = {
    "human":     "#111111",   # humans = near-black (the reference)
    "model":     "#0173B2",   # Model-H = blue
    "gmm":       "#DE8F05",   # GMM = orange
    "dbscan":    "#029E73",   # DBSCAN = teal-green
    "ceiling":   "#7A7A7A",   # consensus ceiling = grey reference
    "shuffle":   "#B0B0B0",   # shuffled-label lesion = light grey
    "gmm_t":     "#F0C05A",   # GMM-target lesion = muted orange
    "dbscan_t":  "#7FC8B3",   # DBSCAN-target lesion = muted teal
    "accent":    "#CC3311",   # sparing highlight (human target line)
    # 2x2 design: response format = hue, visibility = fill
    "lasso":     "#0173B2",   # loop/lasso format
    "anchor":    "#DE8F05",   # anchor/Voronoi format
}
# ordered categorical ramp for many-cluster fills in example partitions
CLUSTER_CYCLE = ["#0173B2", "#DE8F05", "#029E73", "#CC3311", "#7B3FA0",
                 "#E377C2", "#8C6D31", "#17A2B8", "#B0B0B0", "#111111"]

# ---------------------------------------------------------------- fonts
# Prefer Arial/Helvetica (NHB house). Fall back gracefully.
_pref = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_have = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _pref if f in _have), "DejaVu Sans")


def apply_style():
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": FS["label"],
        "axes.titlesize": FS["label"],
        "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["base"],
        "ytick.labelsize": FS["base"],
        "legend.fontsize": FS["base"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # exponents in P values go through mathtext ($10^{-12}$): Arial has no
        # U+207B superscript minus, so render them as maths in the same face.
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.default": "regular",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,      # editable text in the PDF
        "ps.fonttype": 42,
    })


def panel_label(ax, letter, dx=-0.02, dy=1.02, **kw):
    """Bold lowercase panel tag in axis-fraction coords."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS["panel"], fontweight="bold", va="bottom", ha="right",
            family=FONT, **kw)


def fig_label(fig, letter, x, y):
    """Panel tag placed in figure-fraction coords (for drawn schematics)."""
    fig.text(x, y, letter, fontsize=FS["panel"], fontweight="bold",
             va="bottom", ha="left", family=FONT)


# ---------------------------------------------------------------- statistics
def pfmt(p, prefix="P "):
    """Exact P value, NHB house style (exact P preferred over asterisks).

    Very small values become a power of ten rather than "P < .001", because that floor
    hides real differences between e.g. 1e-11 and 1e-157. The exponent is mathtext, so
    it needs the custom mathtext fontset configured in apply_style().
    """
    if p is None or not np.isfinite(p):
        return ""
    if p == 0.0:
        return f"{prefix}$< 1 \\times 10^{{-300}}$"      # underflowed double
    if p >= 0.10:
        return f"{prefix}= {p:.2f}"
    if p >= 0.001:
        return f"{prefix}= {p:.3f}".replace("0.", ".")
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    return f"{prefix}$= {m:.0f} \\times 10^{{{e}}}$"


def sig_bracket(ax, x1, x2, y, text, h=0.02, lw=0.6, fs=FS["tiny"],
                color="#333333", tpad=0.004, va="bottom"):
    """Vertical-axis comparison bracket spanning x1..x2 at height y (data coords)."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=color,
            solid_joinstyle="miter", clip_on=False, zorder=6)
    ax.text((x1 + x2) / 2, y + h + tpad, text, ha="center", va=va,
            fontsize=fs, color=color, clip_on=False, zorder=6)


def sig_bracket_h(ax, y1, y2, x, text, w=0.012, lw=0.6, fs=FS["tiny"],
                  color="#333333", tpad=0.006):
    """Horizontal-axis (barh / forest) bracket spanning y1..y2 at x (data coords)."""
    ax.plot([x, x + w, x + w, x], [y1, y1, y2, y2], lw=lw, color=color,
            solid_joinstyle="miter", clip_on=False, zorder=6)
    ax.text(x + w + tpad, (y1 + y2) / 2, text, ha="left", va="center",
            fontsize=fs, color=color, clip_on=False, zorder=6)


# ---------------------------------------------------------------- marks
def scatter_pts(ax, x, y, s=9, color="#444444", lw=0.35, zorder=3, **kw):
    """Standard stimulus-dot mark.

    The real arrays have neighbour pairs only ~16 px apart on an 800x500 canvas,
    so a filled dot alone renders as one blob: the white edge is what separates them.
    """
    return ax.scatter(x, y, s=s, color=color, edgecolor="white", linewidth=lw,
                      zorder=zorder, **kw)


def pad_lims(ax, w, h, frac=0.08, y_down=True):
    """Expand an array panel by `frac` of its extent so no datum sits on the frame."""
    px, py = w * frac, h * frac
    ax.set_xlim(-px, w + px)
    ax.set_ylim(h + py, -py) if y_down else ax.set_ylim(-py, h + py)


def spread_points(rng, n, x0, x1, y0, y1, min_d, tries=400):
    """Rejection-sample n points in the box keeping pairs >= min_d apart.

    Used for the schematic glyph dots in fig 1a / fig 4a, where raw uniform
    sampling produces touching pairs.
    """
    pts = []
    for _ in range(n * tries):
        if len(pts) == n:
            break
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if all(np.hypot(*(p - q)) >= min_d for q in pts):
            pts.append(p)
    while len(pts) < n:                       # box too tight: relax rather than hang
        pts.append(np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]))
    a = np.array(pts)
    return a[:, 0], a[:, 1]


def save(fig, name, max_mm=180.0):
    """Write both PDF (vector) and PNG (300 dpi) into outputs/.

    !! Saved on the FULL canvas, not with bbox_inches="tight". Tight bbox grows the
    output past the declared figsize to wrap stray labels, which silently pushed two
    of these figures to 182 and 185 mm -- over NHB's 180 mm hard maximum. Keeping the
    canvas means the width is exactly what figsize says; anything that would have
    spilled has to be brought inside the margins instead.
    """
    w_mm = fig.get_size_inches()[0] * 25.4
    if w_mm > max_mm + 0.01:
        raise ValueError(f"{name}: figure is {w_mm:.1f} mm wide, over the "
                         f"{max_mm:.0f} mm maximum")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {name}.pdf / .png   ({w_mm:.0f} x "
          f"{fig.get_size_inches()[1] * 25.4:.0f} mm, 300 dpi)")


def _check_clipping(fig, name, tol_pt=3.0):
    """Warn if any text now falls outside the canvas (the cost of dropping tight bbox)."""
    fig.canvas.draw()
    W, H = (d * fig.dpi for d in fig.get_size_inches())
    for t in fig.findobj(plt.Text):
        if not t.get_text() or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(fig.canvas.get_renderer())
        except Exception:
            continue
        if bb.x0 < -tol_pt or bb.y0 < -tol_pt or bb.x1 > W + tol_pt or bb.y1 > H + tol_pt:
            # ascii-safe: the console here is cp936 and chokes on the minus sign / Greek
            snippet = t.get_text()[:40].encode("ascii", "replace").decode()
            print(f"  !! {name}: text outside canvas -> {snippet!r}")


EXP_LABEL = {
    "exp1": "Exp 1", "exp2": "Exp 2", "exp3": "Exp 3", "exp4": "Exp 4",
}
# which format / visibility each experiment is (for the 2x2 code)
EXP_META = {
    "exp1": ("full",   "lasso"),
    "exp2": ("funnel", "lasso"),
    "exp3": ("full",   "anchor"),
    "exp4": ("funnel", "anchor"),
}
