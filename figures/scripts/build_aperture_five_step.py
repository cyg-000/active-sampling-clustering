"""Build a five-step sequential-aperture illustration for future Figure 1e.

The stimulus is a real 30-point array from the formal 800 x 500 experiment.
The aperture radius is deliberately reduced from the old 240 px schematic to
170 px so information unfolds across five distinct dwells.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = ROOT / "revision_analysis" / "data" / "processed_trials.npz"
OUT = HERE.parent / "assets"
WIDTH, HEIGHT = 800.0, 500.0
PAD_X, PAD_Y = 80.0, 50.0
RADIUS = 170.0
TRIAL_INDEX = 7522
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "reveal_gain", ["#78666D", "#906872", "#B46F76", "#D98570", "#F1B777", "#FFF0B8"]
)


def disk_centers(points: np.ndarray) -> np.ndarray:
    centers = [point.copy() for point in points]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a, b = points[i], points[j]
            vector = b - a
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-12 or distance > 2 * RADIUS:
                continue
            midpoint = (a + b) / 2
            height = np.sqrt(max(RADIUS**2 - (distance / 2) ** 2, 0.0))
            normal = np.asarray([-vector[1], vector[0]]) / distance
            centers.extend([midpoint + height * normal, midpoint - height * normal])
    return np.asarray(centers, float)


def five_step_plan(points: np.ndarray):
    centers = disk_centers(points)
    covers = np.sum((centers[:, None, :] - points[None, :, :]) ** 2, axis=2) <= RADIUS**2
    revealed = np.zeros(len(points), dtype=bool)
    chosen, before, after, gains = [], [], [], []
    while not np.all(revealed):
        candidate_gains = np.sum(covers & ~revealed[None, :], axis=1)
        best = int(np.argmax(candidate_gains))
        if candidate_gains[best] <= 0:
            break
        before.append(revealed.copy())
        chosen.append(centers[best].copy())
        gains.append(int(candidate_gains[best]))
        revealed = revealed | covers[best]
        after.append(revealed.copy())
    if len(chosen) != 5:
        raise RuntimeError(f"Expected five dwells, obtained {len(chosen)}")
    return np.asarray(chosen), before, after, gains


def gain_field(points: np.ndarray, revealed: np.ndarray, grid=(210, 132)):
    # Extend the rendered field beyond the formal 800 x 500 stimulus canvas so
    # edge points and partially clipped apertures retain visual breathing room.
    gx = np.linspace(-PAD_X, WIDTH + PAD_X, grid[0])
    gy = np.linspace(-PAD_Y, HEIGHT + PAD_Y, grid[1])
    xx, yy = np.meshgrid(gx, gy)
    remaining = points[~revealed]
    distance = np.sqrt((xx[:, :, None] - remaining[None, None, :, 0]) ** 2
                       + (yy[:, :, None] - remaining[None, None, :, 1]) ** 2)
    # Smooth rendering of the exact fixed-radius reveal count. The field is in
    # units of newly revealed points, not posterior probability.
    gain = np.sum(1.0 / (1.0 + np.exp((distance - RADIUS) / 10.0)), axis=2)
    return xx, yy, gain


def draw_step(ax, points, chosen, before, after, gains, step, vmax):
    xx, yy, gain = gain_field(points, before[step])
    image = ax.pcolormesh(xx, yy, gain, cmap=GAIN_CMAP, vmin=0, vmax=vmax,
                         shading="auto", rasterized=True)
    hidden = ~after[step]
    revealed = after[step]
    # Hidden points remain only as a faint spatial reference. Once revealed,
    # points become opaque white and stay white in all subsequent states.
    if hidden.any():
        ax.scatter(points[hidden, 0], points[hidden, 1], s=7,
                   color="#D8C8CD", alpha=.28, edgecolor="none", zorder=4)
    if revealed.any():
        ax.scatter(points[revealed, 0], points[revealed, 1], s=14,
                   color="white", alpha=.98, edgecolor="#FFFFFF",
                   linewidth=.25, zorder=6)
    aperture = patches.Circle(chosen[step], RADIUS, facecolor="none",
                              edgecolor="white", linewidth=.75,
                              linestyle=(0, (3, 2)), alpha=.75, zorder=5)
    ax.add_patch(aperture)
    ax.set_xlim(-PAD_X, WIDTH + PAD_X)
    ax.set_ylim(HEIGHT + PAD_Y, -PAD_Y)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333"); spine.set_linewidth(.6)
    return image


def main() -> Path:
    data = np.load(DATASET, allow_pickle=True)
    points = np.asarray(list(data["pts"])[TRIAL_INDEX], float)
    chosen, before, after, gains = five_step_plan(points)
    fig = plt.figure(figsize=(8.9, 4.35))
    gs = fig.add_gridspec(2, 3, left=.035, right=.965, top=.965, bottom=.035,
                          wspace=.54, hspace=.39)
    # Clockwise sequence: top-left -> top-centre -> top-right ->
    # bottom-right -> bottom-centre. The lower-left cell remains open.
    slots = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1)]
    axes = [fig.add_subplot(gs[row, col]) for row, col in slots]
    vmax = float(gains[0])
    for step, ax in enumerate(axes):
        image = draw_step(ax, points, chosen, before, after, gains, step, vmax)
    # Figure-level arrows provide the only sequencing cue; all explanatory
    # text is intentionally omitted for integration into the final Fig. 1.
    for source_ax, target_ax in zip(axes[:-1], axes[1:]):
        source = source_ax.get_position()
        target = target_ax.get_position()
        arrow_len = source.width / 3
        if abs(source.y0 - target.y0) < .02:
            # Horizontal arrows are centred in the enlarged inter-panel gap.
            gap_mid = (source.x1 + target.x0) / 2
            direction = 1 if target.x0 > source.x0 else -1
            start = (gap_mid - direction * arrow_len / 2, (source.y0 + source.y1) / 2)
            end = (gap_mid + direction * arrow_len / 2, (source.y0 + source.y1) / 2)
        else:
            # The turn from panel 3 to panel 4 is vertical, preserving the
            # clockwise reading order.
            gap_mid = (target.y1 + source.y0) / 2
            start = ((source.x0 + source.x1) / 2, gap_mid + arrow_len / 2)
            end = ((source.x0 + source.x1) / 2, gap_mid - arrow_len / 2)
        axes[0].annotate("", xy=end, xytext=start,
                         xycoords=fig.transFigure, textcoords=fig.transFigure,
                         arrowprops=dict(arrowstyle="-|>", lw=1.15,
                                         color="#333333", mutation_scale=11))
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "figure1e_aperture_five_step"
    fig.savefig(stem.with_suffix(".png"), dpi=500, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"stimulus index={TRIAL_INDEX}, n={len(points)}, radius={RADIUS:g}, gains={gains}")
    return stem.with_suffix(".png")


if __name__ == "__main__":
    print(main())
