# Software environment

## Python (model training + figures)

- **Python 3.14**
- **PyTorch** ≥ 2.x with **CUDA** (training); CPU-only for evaluation & figures
- Packages: `numpy`, `scipy`, `scikit-learn`, `torch`, `matplotlib`, `pandas`

Training was done on a rented RTX 5090. Evaluation and all figure generation
run on CPU (`torch` 2.12.0+cpu verified locally).

`matplotlib` CJK figures use `font.sans-serif = ["Microsoft YaHei", ...]` on
the development machine. Figures built from `data_for_figs/` (CSV only) do not
require CJK fonts.

## R (figure data exports only)

- **R 4.5** (developed on 4.5.0)
- CRAN packages: `data.table`, `jsonlite`, `lme4`, `lmerTest`, `cluster`,
  `ggplot2`, `dplyr`, `tidyr`, `purrr`, `car`, `moments`, `knitr`, `rmarkdown`,
  `patchwork`

R is only needed to re-export `figs/data_for_figs/` from the behavioural
analysis pipeline (OSF). The pre-computed CSVs are included, so R is not
required for figure generation or model training.

R is at `C:\Program Files\R\R-4.5.0\bin\Rscript.exe` on the dev machine
(not on PATH). On Linux, install R 4.5 from CRAN and ensure `Rscript` is
on PATH.

## Linux / GPU host

- NVIDIA GPU with CUDA 12.x
- `nvidia-smi` available for GPU utilisation logging
- `bash` (the `run*.sh` scripts use bash, not zsh)
- Standard Unix tools: `awk`, `wc`, `tee`

## Reproduction notes

- R analyses read `dataana/outputs/data_cache.rds` (built by the OSF
  behavioural pipeline). With that cache present, all R export scripts
  run in minutes on CPU.
- Model scripts read a prebuilt `dataset.npz`. Each `autodl/` subdirectory
  contains an identical copy (MD5 `bb36499fda12c6e4ea6e674cda2d2611`).
- Randomness: data splits are by `base_uuid` with a fixed seed
  (`SPLIT_SEED = 42`, `TEST_FRAC = 0.15`), so the held-out set is identical
  across all analyses.
- `PYTHONIOENCODING=utf-8` may be needed on some systems.
