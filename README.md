# Neural network models & figures — Human Visual Clustering of Random Point Arrays

This repository contains the **neural network training code, training data, and
figure-generation scripts** for the manuscript *"..."* (Nature Human Behaviour).
Behavioural analysis (R), raw participant data, and experiment presentation code
are deposited on OSF.

## What this repo produces

- **Model training** — permutation-invariant assignment network (PyTorch + CUDA)
- **Lesion study** — Model-H (trained on human partitions) vs Model-R (GMM / DBSCAN / shuffle targets)
- **Architecture ablation** — 3×2 ablation crossing pairwise-interaction mechanism × point-count provision
- **Hidden-layer probe** — at which layer does cluster structure emerge?
- **Capacity dose–response** — per-group occupancy penalty (pre-registered confirmatory analysis)
- **All five main figures** — PDF + 300-dpi PNG, NHB-compliant formatting

## Repository structure

```
├── README.md
├── ENVIRONMENT.md           # Software versions & package list
├── RESULTS_TO_CODE.md       # Every result → script → command → output
│
├── autodl/                  # Neural network training (Linux + CUDA)
│   ├── tmp/                 # Lesion study: Model-H vs Model-R (core)
│   ├── arch/                # Architecture ablation + hidden-layer probe
│   ├── capacity/            # Capacity dose–response (coarse lambda sweep)
│   ├── capacity2/           # Low-lambda fill sweep
│   └── capacity3_dense/     # Dense re-sweep + frontier analysis
│
├── figs/                    # Main-figure generation (matplotlib)
│   ├── build_all.py         # One-shot: rebuild all 5 figures
│   ├── nhbstyle.py          # NHB visual style (colourblind-safe, 180mm, 300dpi)
│   ├── fig1–5_*.py          # Individual figure scripts
│   ├── prep_*.py            # Data-export scripts (import from analysis modules)
│   ├── export_*.R           # R data exports (run once after analysis pipeline)
│   ├── data_for_figs/       # Pre-computed CSV (figures build without R/OSF data)
│   └── outputs/             # Built PDF + PNG figures
│
├── dataana/
│   └── ceiling/             # Human consensus ceiling + satisficing analysis
│
├── benchmark/               # Ideal-observer degeneracy proofs (Methods)
│
└── stimuli/                 # Provably-random stimulus generator
```

## Quick start

### Reproduce figures (no GPU, no raw data needed)

```bash
cd figs
python build_all.py
```

All figure data is pre-computed in `figs/data_for_figs/`. The scripts read
`autodl/*/runs/comparison.csv` and `dataana/ceiling/outputs/*.csv` directly.

### Train models from pre-built dataset (requires CUDA GPU)

Each `autodl/` subdirectory contains a self-contained `dataset.npz` (1.4 MB,
all five copies are identical). From any of them:

```bash
cd autodl/tmp
bash run.sh                # 3 seeds × Model-H + 4 lesion + ablations

cd autodl/arch
bash run.sh                # 18-model architecture ablation

cd autodl/capacity
bash run_capacity.sh       # Capacity dose–response (coarse)

cd autodl/capacity2
bash run_capacity2.sh      # Low-lambda fill

cd autodl/capacity3_dense
bash run_dense.sh          # Dense re-sweep (50 networks)
```

Set `JOBS=N` to control GPU concurrency (default 4). See each directory's
`run*.sh` for details.

### Rebuild dataset.npz from raw JSON (requires OSF data)

```bash
cd autodl/tmp
python build_dataset.py    # Reads dataexp1-4/ → dataset.npz
```

## Key design notes

- **ARI is the primary metric**, not FM. FM rewards merging (all-one-cluster
  FM = 0.575 vs human–human ceiling 0.683, while ARI = 0.000). See
  `autodl/tmp/metrics.py`.
- **Split by stimulus (base_uuid), not by trial.** The same stimulus was seen
  by hundreds of participants; random trial splitting leaks responses.
- **K is emergent**, not a hyperparameter. Softmax over K_MAX=10 slots; the
  pairwise co-assignment BCE loss determines how many are actually used.
- **Pairwise BCE must NOT use pos_weight** (`PAIR_BALANCE=False`). Weighting
  the positive class rewards merging and collapses the model to K=2.
- **Checkpoint selection uses ARI, not FM** — FM-selected checkpoints freeze
  at the degenerate K=2 state.

## Data availability

- Human behavioural data: **OSF** (restricted access)
- Behavioural analysis (R): **OSF**
- Experiment presentation code: **OSF**
- Model checkpoints (`.pt` files): available on request (~20 MB each)
- This repository: training code, `dataset.npz`, result tables, figures

## Environment

See `ENVIRONMENT.md` for exact software versions. Summary:
- Python 3.14, PyTorch ≥ 2.x with CUDA
- R 4.5 (for `figs/export_*.R` data-export scripts only)
- Linux with NVIDIA GPU (training); CPU-only (evaluation & figures)
