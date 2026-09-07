"""Build Figure 2: one behavior, two explanatory levels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
DATA = ROOT / "revision_analysis" / "data" / "processed_trials.npz"
OUT = PACKAGE / "outputs"
ORIGINAL = PACKAGE / "assets" / "imagefig2_original.jpg"
TRIAL_INDEX = 11502  # real Exp. 3, N=30 human partition

BLUE = "#2F6FB3"
BLUE_LIGHT = "#EDF4FA"
GREEN = "#27883B"
GREEN_LIGHT = "#EDF7EF"
INK = "#222222"
MUTED = "#666666"
GROUPS = ["#4C86C6", "#E6A13A", "#4AA879"]


def clean(ax):
    ax.set_xlim(-40, 840); ax.set_ylim(540, -40); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def point_array(ax, pts, labels=None, alpha=1.0, size=24):
    if labels is None:
        ax.scatter(pts[:, 0], pts[:, 1], s=size, color="#555555",
                   edgecolor="white", linewidth=.45, alpha=alpha)
    else:
        for lab, color in zip(np.unique(labels), GROUPS):
            p = pts[labels == lab]
            ax.scatter(p[:, 0], p[:, 1], s=size, color=color,
                       edgecolor="white", linewidth=.5, alpha=alpha)
    clean(ax)


def abstract_state(ax, organized=False):
    """Abstract node-state icon matching the visual grammar of the source."""
    draw_abstract(ax, (0, 0, 1, 1), stage=3 if organized else 0, size=80)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); ax.axis("off")


def draw_abstract(ax, box, stage=0, size=42):
    """Draw one reusable state; stage 0 and 3 are shared across panels."""
    x0, y0, w, h = box
    pos = np.array([[.20,.66],[.34,.82],[.43,.60],
                    [.20,.30],[.36,.20],[.46,.40],
                    [.65,.72],[.80,.55],[.70,.28]])
    pos[:, 0] = x0 + w * pos[:, 0]; pos[:, 1] = y0 + h * pos[:, 1]
    edges = [(0,1),(1,2),(2,5),(5,4),(4,3),(3,0),(2,6),(6,7),(7,8),(8,5)]
    for i, j in edges:
        ax.plot([pos[i,0],pos[j,0]], [pos[i,1],pos[j,1]], color="#888888",
                lw=.55, ls=(0,(3,2)), zorder=1)
    if stage == 3:
        for cx, cy, color in [(.32,.68,GROUPS[0]),(.33,.30,GROUPS[1]),(.72,.52,GROUPS[2])]:
            ax.add_patch(patches.Ellipse((x0+w*cx, y0+h*cy), w*.42, h*.42,
                         facecolor=color, edgecolor="none", alpha=.10, zorder=0))
    colors = ["#A5A5A5"] * 9
    if stage >= 1:
        colors[:3] = [GROUPS[0]] * 3
    if stage >= 2:
        colors[3:6] = [GROUPS[1]] * 3
    if stage >= 3:
        colors[6:9] = [GROUPS[2]] * 3
    ax.scatter(pos[:,0], pos[:,1], s=size, color=colors, edgecolor=INK,
               linewidth=.45, zorder=3)


def state_partition(ax, pts, centroids, k):
    active = centroids[:k]
    assignment = np.argmin(((pts[:, None, :] - active[None, :, :]) ** 2).sum(2), axis=1)
    for j in range(k):
        p = pts[assignment == j]
        ax.scatter(p[:, 0], p[:, 1], s=17, color=GROUPS[j], alpha=.72,
                   edgecolor="white", linewidth=.35)
    ax.scatter(active[:, 0], active[:, 1], s=58, marker="*", color=GROUPS[:k],
               edgecolor=INK, linewidth=.55, zorder=5)
    clean(ax)


def arrow_between(fig, ax1, ax2, color, y_shift=0):
    a, b = ax1.get_position(), ax2.get_position()
    y = (a.y0 + a.y1) / 2 + y_shift
    fig.add_artist(patches.FancyArrowPatch(
        (a.x1 + .006, y), (b.x0 - .006, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12, lw=1.25, color=color))


def main() -> Path:
    original = plt.imread(ORIGINAL)

    fig = plt.figure(figsize=(7.08, 4.55), facecolor="white")
    # Common empirical target
    top_box = patches.FancyBboxPatch((.285, .635), .43, .205,
                                     boxstyle="round,pad=.012,rounding_size=.018",
                                     transform=fig.transFigure, facecolor="#FAFAFA",
                                     edgecolor="#A0A0A0", linewidth=.8, zorder=-10)
    fig.add_artist(top_box)
    fig.text(.50, .815, "Same human construction", ha="center", va="center",
             fontsize=9, fontweight="semibold", color=INK)
    ax_raw = fig.add_axes([.345, .665, .12, .12]); abstract_state(ax_raw, organized=False)
    ax_final = fig.add_axes([.535, .665, .12, .12]); abstract_state(ax_final, organized=True)
    arrow_between(fig, ax_raw, ax_final, INK)

    # Branch arrows from one behavior to two explanatory levels
    fig.add_artist(patches.FancyArrowPatch((.46, .635), (.255, .470),
                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12,
                   lw=1.15, color=BLUE, connectionstyle="arc3,rad=.10"))
    fig.add_artist(patches.FancyArrowPatch((.54, .635), (.745, .470),
                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12,
                   lw=1.15, color=GREEN, connectionstyle="arc3,rad=-.10"))

    # Explanatory-level containers
    left = patches.FancyBboxPatch((.045, .140), .43, .320,
                                  boxstyle="round,pad=.012,rounding_size=.018",
                                  transform=fig.transFigure, facecolor=BLUE_LIGHT,
                                  edgecolor=BLUE, linewidth=1.0, zorder=-10)
    right = patches.FancyBboxPatch((.525, .140), .43, .320,
                                   boxstyle="round,pad=.012,rounding_size=.018",
                                   transform=fig.transFigure, facecolor=GREEN_LIGHT,
                                   edgecolor=GREEN, linewidth=1.0, zorder=-10)
    fig.add_artist(left); fig.add_artist(right)

    fig.text(.105, .425, "Endpoint account", ha="left", va="center",
             fontsize=10, fontweight="semibold", color=BLUE)
    fig.text(.585, .425, "Stopping account", ha="left", va="center",
             fontsize=10, fontweight="semibold", color=GREEN)

    # Reuse the source figure's abstract static-mapping illustration.
    # Pixel crops intentionally exclude the old headings and lower statistics.
    left_crop = original[142:315, 30:602]
    lmap = fig.add_axes([.072, .165, .380, .200])
    lmap.imshow(left_crop, interpolation="lanczos"); lmap.axis("off")
    # Preserve the source figure's endpoint mapping: the visual vocabulary is
    # already recognizable as input -> representation -> final assignment.

    # Preserve the source figure's time-unfolding sequence. Its successive
    # frames communicate process more directly than abstract recolouring.
    right_crop = original[158:345, 650:1240]
    rmap = fig.add_axes([.528, .165, .400, .220])
    rmap.imshow(right_crop, interpolation="lanczos"); rmap.axis("off")

    stem = OUT / "figure2"
    fig.savefig(stem.with_suffix(".jpg"), dpi=450, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".jpg")


if __name__ == "__main__":
    print(main())
