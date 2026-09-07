"""Assemble final Figure 1: task design and sequential information access."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "outputs"
ASSETS = PACKAGE / "assets"
INK = "#242424"
MUTED = "#666666"
BLUE = "#1677B7"
BLUE_LIGHT = "#EEF6FB"
ORANGE = "#DF9200"
ORANGE_LIGHT = "#FFF6E6"
LASSO = "#7467A8"
GRAY = "#8B8B8B"


def trimmed(path: Path, pad: int = 8):
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        l, t, r, b = box
        box = (max(0, l-pad), max(0, t-pad), min(im.width, r+pad), min(im.height, b+pad))
        im = im.crop(box)
    return np.asarray(im)


def reveal_image(path: Path):
    """Compensate only for contrast lost when the large source is reduced."""
    im = Image.fromarray(trimmed(path, pad=12))
    im = ImageEnhance.Contrast(im).enhance(1.16)
    im = ImageEnhance.Color(im).enhance(1.06)
    im = ImageEnhance.Sharpness(im).enhance(1.18)
    return np.asarray(im)


def image_ax(fig, rect, path, border="#C8C8C8", lw=.65):
    ax = fig.add_axes(rect)
    ax.imshow(trimmed(path), interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(border); s.set_linewidth(lw)
    return ax


def panel_label(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=9.5, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(x + .025, y, title, fontsize=8.2, fontweight="semibold", color=INK,
             ha="left", va="top")


def cohort_card(ax, x, y, w, h, exp, n, color, fill):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=.008,rounding_size=.025",
        facecolor="white", edgecolor=color, linewidth=1.15))
    ax.add_patch(patches.Rectangle((x, y+h*.73), w, h*.27,
                                   facecolor=fill, edgecolor="none"))
    ax.text(x+w/2, y+h*.865, f"Exp. {exp}", ha="center", va="center",
            fontsize=7.2, fontweight="semibold", color=INK)
    ax.text(x+w/2, y+h*.40, f"N = {n}", ha="center", va="center",
            fontsize=7.0, color=INK)


def main():
    a = ASSETS / "imgfig1a.png"
    lasso = ASSETS / "imgfig1b1.png"
    aperture = ASSETS / "imgfig1b2.png"
    anchor = ASSETS / "imgfig1c1.png"
    reveal = ASSETS / "figure1e_aperture_five_step.png"

    fig = plt.figure(figsize=(7.08, 6.05), facecolor="white")

    # a | common random input
    fig.text(.035, .955, "a", fontsize=9.5, fontweight="bold", color=INK,
             ha="left", va="top")
    image_ax(fig, [.055, .585, .225, .295], a)

    # b | visibility manipulation
    fig.text(.345, .955, "b", fontsize=9.5, fontweight="bold", color=INK,
             ha="left", va="top")
    image_ax(fig, [.365, .755, .245, .145], a, border=BLUE)
    image_ax(fig, [.365, .505, .245, .145], aperture, border=GRAY)
    fig.text(.488, .908, "Full view", ha="center", va="bottom", fontsize=7.1, color=BLUE)
    fig.text(.488, .658, "Local aperture", ha="center", va="bottom", fontsize=7.1, color=MUTED)

    # c | response grammar
    fig.text(.655, .955, "c", fontsize=9.5, fontweight="bold", color=INK,
             ha="left", va="top")
    image_ax(fig, [.675, .755, .27, .145], lasso, border=LASSO)
    image_ax(fig, [.675, .505, .27, .145], anchor, border=ORANGE)
    fig.text(.810, .908, "Lasso", ha="center", va="bottom", fontsize=7.1, color=LASSO)
    fig.text(.810, .658, "Anchor", ha="center", va="bottom", fontsize=7.1, color=ORANGE)

    # d | sequential information access; retain the approved warm map.
    fig.text(.035, .487, "d", fontsize=9.5, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(.060, .487, "Aperture process", fontsize=9.0, fontweight="normal",
             color=INK, ha="left", va="top")
    eax = fig.add_axes([.045, .008, .92, .435])
    eax.imshow(reveal_image(reveal), interpolation="lanczos")
    eax.set_xticks([]); eax.set_yticks([])
    for s in eax.spines.values(): s.set_visible(False)

    stem = OUT / "figure1"
    fig.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(stem.with_suffix(".jpg"), dpi=450, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


if __name__ == "__main__":
    print(main())
