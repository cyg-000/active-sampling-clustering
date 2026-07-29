"""DRAFT sketch for Fig 4a: isometric 'layered' schematic of the
permutation-invariant assignment network (route 1 — faithful, not CNN blocks).

Standalone: does NOT touch fig4_model.py. Renders outputs/fig4a_sketch.png so we
can iterate on the look before folding it into the real figure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
import nhbstyle as S

S.apply_style()
OUT = os.path.join(os.path.dirname(__file__), "outputs")

# ---- isometric depth vector (pushes a face "back" into the page) ----
DV = np.array([0.34, 0.46])          # one unit of depth -> screen offset


def _shade(hexc, f):
    """Multiply an #rrggbb colour by factor f (<1 darker, >1 lighter-clamped)."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(min(1, (c / 255) * f) for c in (r, g, b))


def slab(ax, x, y, w, h, depth, fc, z=1, lw=0.8, ec="#5A5A5A"):
    """Draw one isometric cuboid; front face at (x,y) size w x h, extruded `depth`."""
    d = DV * depth
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + d[0], y + h + d[1]), (x + d[0], y + h + d[1])]
    side = [(x + w, y), (x + w, y + h), (x + w + d[0], y + h + d[1]), (x + w + d[0], y + d[1])]
    ax.add_patch(Polygon(top, closed=True, fc=_shade(fc, 1.08), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon(side, closed=True, fc=_shade(fc, 0.82), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon(front, closed=True, fc=fc, ec=ec, lw=lw, zorder=z + 0.1))
    return x + w / 2, y + h / 2                      # front-face centre


def arrow(ax, p0, p1, color="#5A5A5A", lw=1.0, style="-|>", z=5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8,
                                 lw=lw, color=color, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


BLUE = "#DCE9F5"; TAN = "#F5E6CC"; GREY = "#ECECEC"; OUTB = "#CFE0F0"

fig, ax = plt.subplots(figsize=(112 * S.MM, 66 * S.MM))
ax.set_xlim(0, 13.2); ax.set_ylim(0, 7.4); ax.axis("off"); ax.set_aspect("equal")

yb, h = 2.2, 1.8                                     # baseline y, slab height

# ---------------------------------------------------------- 1. point array
cx0, _ = slab(ax, 0.4, yb, 1.15, h, 0.55, GREY)
rng = np.random.default_rng(4)
px = 0.4 + 0.18 + 0.8 * rng.random(9)
py = yb + 0.22 + (h - 0.44) * rng.random(9)
ax.scatter(px, py, s=7, color="#555", zorder=3)
ax.text(0.4 + 0.575, yb - 0.32, "Point\narray", ha="center", va="top", fontsize=5.6)

# ---------------------------------------------------------- 2. shared MLP embed
mx, _ = slab(ax, 2.1, yb, 1.15, h, 0.55, BLUE)
for i in range(5):                                  # N per-point vectors as rows
    yy = yb + 0.3 + i * (h - 0.6) / 4
    ax.plot([2.28, 3.06], [yy, yy], color="#7FA8CC", lw=1.4, zorder=3)
ax.text(2.1 + 0.575, yb - 0.32, "Shared\nMLP embed", ha="center", va="top", fontsize=5.6)
arrow(ax, (1.55 + 0.34 * 0.55, yb + h / 2 + 0.46 * 0.55 / 2), (2.05, yb + h / 2))

# ---------------------------------------------------------- 3. set-interaction (L stacked)
bx = 4.05
for k in range(3):                                  # three slabs receding in depth = L blocks
    off = DV * (2 - k) * 0.9
    slab(ax, bx + off[0], yb + off[1], 1.2, h, 0.55, BLUE, z=1 + k)
fcx = bx + 0.6
for i in range(5):
    yy = yb + 0.3 + i * (h - 0.6) / 4
    ax.plot([bx + 0.16, bx + 1.04], [yy, yy], color="#7FA8CC", lw=1.4, zorder=4)
ax.text(bx + 0.6, yb - 0.32, "Set-interaction\nblocks  (× L)", ha="center", va="top",
        fontsize=5.6)

# aggregation motif: global mean/max pooled from all points, broadcast back
agx, agy = bx + 0.6, yb + h + 1.35
ax.add_patch(Polygon([(agx - 0.5, agy), (agx, agy + 0.42), (agx + 0.5, agy),
                      (agx, agy - 0.42)], closed=True, fc="#FBEAD1", ec="#B98A3E",
                     lw=0.9, zorder=6))
ax.text(agx, agy, "mean\nmax", ha="center", va="center", fontsize=5.0, color="#7A5A1E")
for yy in (yb + 0.45, yb + h / 2, yb + h - 0.45):        # 3 clean gather lines
    arrow(ax, (bx + 1.02, yy), (agx - 0.4, agy - 0.05), color="#D8BE8C", lw=0.6,
          style="-", z=5, rad=-0.10)
arrow(ax, (agx + 0.15, agy - 0.4), (bx + 1.05, yb + h - 0.15), color="#B98A3E",
      lw=0.9, style="-|>", z=6, rad=-0.35)
ax.text(agx - 0.3, agy + 0.5, "self + global pool → point", ha="center", va="bottom",
        fontsize=5.0, color="#8A6A2E")
arrow(ax, (3.15, yb + h / 2), (bx - 0.02, yb + h / 2))

# ---------------------------------------------------------- 4. GRU (optional branch)
gx, gyr = 6.55, yb + 1.95                              # lowered, closer to main axis
slab(ax, gx, gyr, 1.25, 1.35, 0.5, TAN, z=2)
# little recurrence loop
lx, ly = gx + 0.62, gyr + 0.68
ax.add_patch(FancyArrowPatch((lx + 0.18, ly + 0.28), (lx - 0.18, ly + 0.28),
             connectionstyle="arc3,rad=1.6", arrowstyle="-|>", mutation_scale=6,
             lw=0.9, color="#B98A3E", zorder=6))
ax.text(gx + 0.62, gyr - 0.28, "GRU over\nreveal order", ha="center", va="top",
        fontsize=5.4)
ax.text(gx + 0.62, gyr + 1.75, "masked (funnel)\nexperiments only", ha="center",
        va="bottom", fontsize=5.0, style="italic", color="#8A6A2E")
arrow(ax, (fcx + 0.78, yb + h - 0.12), (gx + 0.08, gyr + 0.12), color="#B98A3E",
      lw=0.9, style="-|>", rad=0.20)
arrow(ax, (gx + 1.3, gyr + 0.12), (8.75, yb + h - 0.12), color="#B98A3E", lw=0.9,
      style="-|>", rad=0.20)
arrow(ax, (fcx + 0.9, yb + h / 2), (8.7, yb + h / 2))   # main path bypasses GRU

# ---------------------------------------------------------- 5. per-point softmax -> slots
ox = 8.8
slab(ax, ox, yb, 1.5, h, 0.55, OUTB, z=3)
# N x K assignment grid on the front face, a few slots populated (emergent clusters)
nK, nN = 4, 5
gx0, gy0 = ox + 0.18, yb + 0.26
cw, ch = (1.5 - 0.36) / nK, (h - 0.52) / nN
asg = [0, 0, 1, 2, 1]                                # which slot each point lands in
for i in range(nN):
    for j in range(nK):
        on = (asg[i] == j)
        col = S.CLUSTER_CYCLE[j] if on else "white"
        ax.add_patch(Polygon([(gx0 + j * cw, gy0 + i * ch),
                              (gx0 + (j + 1) * cw, gy0 + i * ch),
                              (gx0 + (j + 1) * cw, gy0 + (i + 1) * ch),
                              (gx0 + j * cw, gy0 + (i + 1) * ch)], closed=True,
                             fc=col, ec="#B9C7D6", lw=0.4, zorder=4))
ax.text(ox + 0.75, yb - 0.32, "Per-point softmax\n→ K slots", ha="center", va="top",
        fontsize=5.6)
arrow(ax, (10.35, yb + h / 2), (10.95, yb + h / 2))
ax.text(11.02, yb + h / 2, "soft partition\neach point →\none soft slot", ha="left",
        va="center", fontsize=5.0)

S.panel_label(ax, "a", dx=0.0, dy=1.0)

# full canvas, no tight bbox: same policy as nhbstyle.save, so the declared figure
# width is the width that ships
fig.savefig(os.path.join(OUT, "fig4a_sketch.png"), dpi=300)
plt.close(fig)
print("saved outputs/fig4a_sketch.png "
      f"({fig.get_size_inches()[0] * 25.4:.0f} mm wide, 300 dpi)")
