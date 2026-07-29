# Stimulus Generation Rules (Marupudi & Varma 2024)

## Core Principle

Stimuli are NOT purely random. They are generated via **rejection sampling** to have specific statistical spatial structure.

## Generation Steps

1. **Uniform random sampling**: N points sampled from U(margin, W-margin) × U(margin, H-margin) within an 800×500 canvas
2. **Minimum distance constraint**: Each new point must be ≥12px from all existing points (rejection within generation)
3. **Clark-Evans Z-score**: Measures spatial clustering — negative = clustered, positive = dispersed
4. **Vacuumed Z-score**: Same as Z but computed on the bounding box of the points (removes canvas edge effects)
5. **Standardized Z-score**: (vz - mean_for_N) / sd_for_N — from empirical distribution of 100K samples per N
6. **Target filtering**: Keep only stimuli with std_vz in target ranges

## Target Ranges

| Group | std_vz range | Meaning |
|-------|-------------|---------|
| clustered | (-2.05, -1.95) | Points more clustered than random |
| disperse | (0.95, 1.05) | Points more dispersed than random |
| very_clustered | (-3.55, -3.45) | Extremely clustered (not used in Exp1-3) |

## Implication for RNN

The RNN does NOT need ground-truth labels to learn clustering structure. The point coordinates themselves carry the statistical signal (through local density, nearest-neighbor distances, etc.). The rejection sampling ensures that "clustered" and "disperse" stimuli have systematically different spatial statistics that an RNN can learn to detect from raw (x,y) sequences.
