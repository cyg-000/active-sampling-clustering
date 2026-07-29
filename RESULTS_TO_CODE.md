# Result → Code → Command → Output

Every main-text result mapped to the script that produces it.
Paths relative to repository root. **R** = R 4.5; **Py** = Python 3.14;
**GPU** = requires CUDA.

---

## R2 — Lesion study: the grouping prior is learned (`autodl/tmp/`)

Build data (on machine with raw JSON): `python autodl/tmp/build_dataset.py` → `dataset.npz`.
Train (GPU): `bash autodl/tmp/run.sh` (Model-H seeds + Model-R gmm/dbscan/shuffle).
Evaluate: `python autodl/tmp/evaluate.py` → `autodl/tmp/runs/comparison.csv`.

| Result | Code | Output | Where |
|--------|------|--------|-------|
| Model-H ARI 0.44 vs Model-R 0.15–0.31 | `autodl/tmp/{model,losses,metrics,train,evaluate,data,gpu_data,config}.py` | `runs/comparison.csv` | MT core |
| Metric choice (ARI primary; FM rewards merging) | `autodl/tmp/metrics.py` | — | MT Methods |
| Loss design (pairwise co-assignment; balance OFF) | `autodl/tmp/losses.py`, `config.py` | — | MT Methods |

---

## R3 — Satisficing: humans do not maximise separability

| Result | Code | Output | Where |
|--------|------|--------|-------|
| Human sil 0.37 < GMM 0.40 < DBSCAN 0.51 | `dataana/ceiling/ceiling.py` (section C) | `gmm_mismatch.csv` | MT core |
| Methods basis: ideal observer degenerates to K=1 | `benchmark/s8_benchmark.py`, `s9_generative_null.py` | stdout | MT Methods + Supp |

Run: `python dataana/ceiling/ceiling.py`

---

## R4 — Human consensus ceiling (`dataana/ceiling/`)

| Result | Code | Output | Where |
|--------|------|--------|-------|
| Single-human ARI 0.39 / consensus 0.51 / Model-H = 85% | `dataana/ceiling/ceiling.py` (section A) | `ceiling_summary.csv`, `human_human_ari.csv` | MT core |
| ARI–|Δk| coupling (human & model) | `dataana/ceiling/ceiling.py` (section B) | `coupling.csv` | MT |

---

## R5 — Capacity-limited account of the cluster-count invariant

| Result | Code | Output | Where |
|--------|------|--------|-------|
| Architecture ablation: slope_k not learned | `autodl/arch/{model2,train,evaluate,config}.py` | `autodl/arch/runs/comparison.csv` | MT mechanism |
| Hidden-layer probe | `autodl/arch/probe.py` | `autodl/arch/runs/probe.json` | MT/Supp |
| Capacity dose–response (per-group cap = 4) | `autodl/capacity/losses.py` (`reg_capacity`), `run_capacity.sh` | `autodl/capacity/runs/comparison.csv` | MT mechanism |
| Low-λ fill (confirms Case B) | `autodl/capacity2/run_capacity2.sh` | `outputs_merged_curve.csv` | MT/Supp |
| Dense re-sweep + frontier analysis | `autodl/capacity3_dense/run_dense.sh`, `analyze_frontier.py` | `outputs/frontier_summary.csv` | MT |

Train/eval (GPU): `bash autodl/arch/run.sh`; `bash autodl/capacity/run_capacity.sh`;
`bash autodl/capacity2/run_capacity2.sh`; `bash autodl/capacity3_dense/run_dense.sh`.

---

## Stimuli (Methods)

| Item | Code | Output |
|------|------|--------|
| Provably-random stimulus generator (Clark-Evans rejection sampling) | `stimuli/generate_stimuli.py` | stimulus JSON |

---

## Figures

| Fig | Message | Script(s) | Data sources |
|-----|---------|-----------|-------------|
| 1 | Paradigm + reliable, shared structure | `figs/fig1_reliability.py` | `export_examples.R`, `export_fig1.R` |
| 2 | Two invariant slopes, reallocated by format | `figs/fig2_invariants.py` | `export_fig2.R` |
| 3 | Satisficing: bounded RT, sub-optimal separability | `figs/fig3_satisfice.py` | `export_fig3.R`, `e5_rt_decomp.R`, `prep_fig3_silhouette.py` |
| 4 | Neural model reverse-engineers the learned prior | `figs/fig4_model.py` | `prep_fig4_stats.py`, `autodl/tmp/runs/comparison.csv` |
| 5 | Cluster-count slope = graded capacity | `figs/fig5_capacity.py` | `prep_fig5_probe.py`, `autodl/arch/runs/comparison.csv`, `autodl/capacity3_dense/outputs/frontier_summary.csv` |

Build all: `python figs/build_all.py`

---

## Derived artefacts for reproduction (without raw JSON)

- `autodl/*/dataset.npz` — all model training reads this (5 identical copies, 1.4 MB each)
- `autodl/*/runs/comparison.csv` — model result tables (small)
- `autodl/arch/runs/probe.json` — probe result
- `dataana/ceiling/outputs/*.csv` — ceiling + satisficing summary
- `figs/data_for_figs/*.csv` — pre-computed figure data
