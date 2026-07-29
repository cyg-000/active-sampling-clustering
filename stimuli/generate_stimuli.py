"""
Minimal stimulus generator — pure Python, no C++ dependency.
Replicates the Clark-Evans rejection sampling from Marupudi & Varma (2024).

Generation rule:
1. Sample N points uniformly in [margin, W-margin] x [margin, H-margin]
2. Reject any point within 12px of an existing point
3. Compute vacuumed Z-score (Clark-Evans nearest-neighbor statistic,
   measured within the convex hull / bounding box of the points)
4. Keep only stimuli whose std_vz falls in target ranges:
   - clustered:  std_vz in (-2.05, -1.95)
   - disperse:   std_vz in ( 0.95,  1.05)
"""
import math, random, json, os, uuid
from dataclasses import dataclass, field
from typing import Optional

W, H = 800, 500
MARGIN = 7
MIN_DIST = 12

# --- Clark-Evans Z-score statistics ---
def mean_nearest_neighbor_distance(points):
    n = len(points)
    if n < 2: return 0.0
    total = 0.0
    for i in range(n):
        min_d = float('inf')
        xi, yi = points[i]
        for j in range(n):
            if i == j: continue
            xj, yj = points[j]
            d = (xi-xj)**2 + (yi-yj)**2
            if d < min_d: min_d = d
        total += math.sqrt(min_d)
    return total / n

def expected_nnd(n, area, perimeter):
    if n == 0: return 0
    return 0.5 * math.sqrt(area/n) + (0.0514 + 0.041/math.sqrt(n)) * (perimeter/n)

def var_nnd(area, n):
    if n == 0: return 0
    return 0.070 * area/(n**2) + 0.037 * math.sqrt(area/(n**5))

def z_score(points, width, height):
    """Clark-Evans Z: negative=clustered, positive=dispersed."""
    n = len(points)
    area = width * height
    perimeter = 2 * (width + height)
    obs = mean_nearest_neighbor_distance(points)
    exp_val = expected_nnd(n, area, perimeter)
    var_val = var_nnd(area, n)
    if var_val == 0: return 0.0
    return (obs - exp_val) / math.sqrt(var_val)

def vacuumed_z_score(points):
    """Z-score computed within the bounding box of the points (remove whitespace)."""
    if not points: return 0.0
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    w = max(xs) - min(xs); h = max(ys) - min(ys)
    if w < 1 or h < 1: return 0.0
    return z_score(points, w, h)

# --- STD_VZ_TABLE: mean & sd of vacuumed_z per point count ---
STD_VZ_TABLE = {
    10: (1.25, 1.03), 15: (1.03, 1.00), 20: (0.924, 0.992),
    25: (0.873, 0.979), 30: (0.864, 0.971), 35: (0.867, 0.962),
    40: (0.880, 0.952), 45: (0.907, 0.948), 50: (0.945, 0.940),
    55: (0.979, 0.935), 60: (1.03, 0.927), 65: (1.08, 0.917),
    70: (1.14, 0.915), 75: (1.20, 0.915), 80: (1.26, 0.909),
    85: (1.32, 0.910), 90: (1.39, 0.901), 95: (1.46, 0.893),
    100: (1.55, 0.897),
}

def std_vz_score(n, vz):
    mean, sd = STD_VZ_TABLE[n]
    return (vz - mean) / sd

# --- Point & Stimulus classes ---
@dataclass
class Stimulus:
    number_of_points: int
    points: list  # [(x,y), ...]
    base_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    unique_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    z_score_val: float = 0.0
    vacuumed_z_score_val: float = 0.0
    std_vaccumed_z_score_val: float = 0.0
    flipped: bool = False
    group: Optional[str] = None
    practice_stimulus: bool = False

    @classmethod
    def generate(cls, n_points):
        """Rejection sampling: uniform random with min-distance constraint."""
        points = []
        while len(points) < n_points:
            x = random.uniform(MARGIN, W - MARGIN)
            y = random.uniform(MARGIN, H - MARGIN)
            too_close = any(
                (x-px)**2 + (y-py)**2 < MIN_DIST**2 for px, py in points
            )
            if not too_close:
                points.append((x, y))

        vz = vacuumed_z_score(points)
        z = z_score(points, W, H)
        svz = std_vz_score(n_points, vz)

        return cls(
            number_of_points=n_points, points=points,
            z_score_val=z, vacuumed_z_score_val=vz,
            std_vaccumed_z_score_val=svz
        )

    def to_dict(self):
        return {
            "base_uuid": self.base_uuid,
            "unique_uuid": self.unique_uuid,
            "number_of_points": self.number_of_points,
            "z_score": self.z_score_val,
            "vacuumed_z_score": self.vacuumed_z_score_val,
            "std_vaccumed_z_score": self.std_vaccumed_z_score_val,
            "flipped": self.flipped,
            "group": self.group,
            "points": [{"x": p[0], "y": p[1]} for p in self.points],
            "practice_stimulus": self.practice_stimulus
        }


# --- Generate ---
def generate_dataset(n_per_group=3, groups=None):
    if groups is None:
        groups = [
            ("disperse", 0.95, 1.05),
            ("clustered", -2.05, -1.95),
            ("very_clustered", -3.55, -3.45),
        ]

    all_stimuli = []
    for n_pts in (10, 15, 20, 25, 30, 35, 40):
        print(f"  n={n_pts}...")
        for name, lo, hi in groups:
            count = 0
            while count < n_per_group:
                stim = Stimulus.generate(n_pts)
                if lo < stim.std_vaccumed_z_score_val < hi:
                    stim.group = name
                    all_stimuli.append(stim)
                    count += 1
    return all_stimuli


if __name__ == "__main__":
    random.seed(42)
    print("Generating stimuli (pure Python)...")
    stimuli = generate_dataset(n_per_group=3)

    out = {}
    for s in stimuli:
        out[s.unique_uuid] = s.to_dict()

    os.makedirs("output", exist_ok=True)
    with open("output/stimuli.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {len(stimuli)} stimuli → output/stimuli.json")
    for name in ("disperse", "clustered", "very_clustered"):
        n = sum(1 for s in stimuli if s.group == name)
        print(f"  {name}: {n}")
