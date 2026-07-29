# figs_nhb — Nature Human Behaviour main figures

Five main figures for `manucluster_cyg1.docx`, one conceptual message each, covering
all nine Results. Pure matplotlib (incl. schematics); vector **PDF** + 300-dpi **PNG**
in `outputs/`.

## Build
```
# 1) data exports (only needed once, or after the analysis pipelines change)
Rscript figs_nhb/export_examples.R      # F1 b/c: real stimulus + partitions
Rscript figs_nhb/export_fig1.R          # F1d: per-participant internal vs inter-participant FM
Rscript figs_nhb/export_fig2.R          # F2: slopes-by-set-size + pooled reallocation model + retest
Rscript figs_nhb/export_fig3.R          # F3a: RT-by-set-size
python  figs_nhb/prep_fig3_silhouette.py# (diagnostic only; NOT used in F3c — see note)
python  figs_nhb/prep_fig4_stats.py     # F4c/F4d: per-trial model k+silhouette, |Δk| coupling SEM (~3 min, CPU)
python  figs_nhb/prep_fig5_probe.py     # F5b: per-trial layer probe, 3 seeds + coord baseline (~5 min, CPU)
Rscript figs_nhb/../dataana/explore2/e5_rt_decomp.R  # F3b: RT motor/deliberation decomposition
# 2) render
python  figs_nhb/build_all.py
```
R is at `C:\Program Files\R\R-4.5.0\bin\Rscript.exe` (not on PATH).
The two `prep_*.py` steps import the real analysis modules (`dataana/ceiling/ceiling.py`,
`autodl/tmp/model.py`, `autodl/arch/probe.py`) rather than re-implementing anything, so the
recomputed values reproduce `comparison.csv` / `probe.json` exactly.

## Visual system — `nhbstyle.py`
Fixed colourblind-safe entity palette, same colour per actor across every figure:
human = near-black, Model-H = blue, GMM = orange, DBSCAN = teal; response format
code lasso = blue / anchor = orange, visibility = fill (full solid / masked hollow).

## Figure ↔ Result ↔ data source
| Fig | Message | Results | Data |
|-----|---------|---------|------|
| 1 | paradigm + reliable, shared structure | R1 | `dataana/.../rq1_reliability_desc`, `rq2_agreement_desc`; real partitions via `export_examples.R` |
| 2 | two invariant slopes, reallocated by format | R2, R3 | `export_fig2.R` (per-set-size means; **re-fit** pooled interaction reproduces manuscript β to 3 dp; retest r=.90/.86) |
| 3 | satisficing: bounded (cognitive, not motor) RT, sub-optimal separability, confidence null | R4, R5, R8 | a: `f3_rt_by_np` + `e2_q1_scaling`; **b (motor/deliberation decomposition): `e5_rt_components`**; c: `autodl/arch/runs/comparison.csv`; d: `e1_models` |
| 4 | neural model reverse-engineers the learned prior; consensus ceiling | R6, R7 | `autodl/tmp/runs/comparison.csv` (lesion); `dataana/ceiling/outputs/ceiling_summary`+`coupling` |
| 5 | cluster-count slope = graded capacity, not architecture (**1×3: a arch-ablation, b layer-probe, c λ dose-response**; the old c≈4/working-memory schematic panel d was dropped — the "graded not fixed" claim rests on c's monotone trade-off, not on a working-memory number) | R9 | `autodl/arch/runs/comparison.csv` + `probe.json`; `autodl/capacity2/outputs_merged_curve.csv` |

## NHB submission formatting (AIP requirements)

Legends for all five figures are in **`FIGURE_LEGENDS.md`**, written to the AIP spec: brief
title, panels described in sequence, every centre value and error bar defined and its
derivation stated, plus n, test and exact P; keys are described verbally ("open circles",
"red dashed line") rather than by symbol.

Enforced automatically by `nhbstyle.py`, so a later edit cannot quietly break them:

| Requirement | How it is met |
|---|---|
| **≤ 180 mm wide** | `save()` refuses to write a figure wider than 180 mm. It also saves on the **full canvas** — `bbox_inches="tight"` used to grow the output past `figsize` to wrap stray labels, which had silently pushed F2 to 182 mm and F5 to 185 mm. All five are now exactly 180.0 mm. |
| **≥ 300 dpi** | `savefig.dpi = 300` for PNG; the PDF is vector. |
| **5–7 pt sans-serif labelling** | every size comes from the `FS` dict (tiny 5.0 / small 5.5 / base 6.0 / label 7.0); panel letters are 8 pt bold. 20 labels were below the floor, down to 4.1 pt — the worst were in F4a, whose schematic multiplied every size by 0.82. |
| **Labels not flattened onto images** | `pdf.fonttype = 42`; the PDFs contain live Arial text, verified extractable. |
| **Error bars where appropriate** | every panel with a point estimate has one; all defined in the legends. |

Because the canvas is now fixed, anything that would previously have spilled outside has to
fit inside the margins instead. `save()` therefore runs a clipping check after each render
and prints a warning naming any text that falls outside the canvas.

## Statistics shown on the figures

Exact **P** values throughout (NHB prefers exact P to asterisks); `nhbstyle.pfmt` formats
them and renders small exponents as mathtext, because Arial has no U+207B superscript minus.

**Unit of analysis.** Everything that compares model / algorithm summary statistics is paired
at the **stimulus** level (n = 8), never at the trial level. The held-out set is 3295 trials
over only **8 stimuli**, and GMM / DBSCAN are deterministic — they emit exactly one silhouette
and one k per stimulus — so a trial-level test replicates the same constant ~400 times and
inflates n accordingly. Concretely, human-vs-GMM silhouette is P = 4 × 10⁻¹¹ at the trial level
and **P = .69** over the 8 stimuli. F3c and F4c therefore plot the eight per-stimulus values as
dots on top of the bars, so the real n is visible. The per-stimulus results reproduce
`dataana/ceiling/outputs/pertrial_stats.txt` exactly (H > GMM target 8/8, P = .0078;
H > DBSCAN target 7/8, P = .039).

- **F1d** paired Wilcoxon within participant (n = 67/71/67/74) on `f1_reliability_bysid.csv`;
  bars carry 95% CIs on the same participant-level n.
- **F2 a/b** per-experiment LMM slope + P from `rq5_{nclusters,numerosity}_models.csv`;
  **c** Wald z on β/se. **d** is on **log₂ axes** — the old `[0.5, 8.5]` box silently hid 9 of
  the 67 Exp-1 participants (median k up to 24) while quoting r computed on all 67.
- **F3a** exponent vs b = 1 from `e2_q1_scaling.csv:p_vs_1`; **c** per-stimulus (see above);
  **d** mixed-model P from `e1_models.csv`.
- **F4 b/c/d-top** per-stimulus paired Wilcoxon; **d-bottom** error bars are stimulus-clustered,
  not pair-level (the ~677k human–human pairs share raters and stimuli).
- **F5b** points are the mean over 3 seeds ± s.e.m.; the baseline test is per-stimulus
  (every block of every family beats the raw-coordinate baseline 8/8, P = .0078). No band is
  drawn around the baseline: its across-stimulus spread is irrelevant to a within-stimulus
  paired test and would misrepresent the comparison as far more uncertain than it is.

**Caveat, stated not hidden:** Model-H is three seeds (H_s0/s1/s2), but the lesioned targets
`R_gmm` / `R_dbscan` / `R_shuffle` are a **single training run each**. Their comparisons are
valid per stimulus and carry no across-seed generalisation.

