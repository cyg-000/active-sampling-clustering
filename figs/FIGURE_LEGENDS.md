# Figure legends — main figures 1–5

Written to the Nature Human Behaviour AIP figure-legend spec: a brief title, then each
panel described in sequence; centre values and every error bar defined and said how they
were computed; sample size, statistical test and exact P given; keys described in words
("open circles"), never by symbol; methodological detail left to Methods.

Word counts are given per legend so they can be checked against the article-type limit.
Greek characters (β, Δ, λ) are set in Symbol font in the Word manuscript; in the supplied
PDFs they are live Unicode text in the embedded Arial subset (`pdf.fonttype = 42`), so a
copy editor can restyle them without redrawing.

---

## Fig. 1 | A structureless array is partitioned reliably and in shared ways.

**a**, Design. Two response formats (lasso, draw closed loops around groups; anchor, place
group-defining points) crossed with two visibility conditions (full view; masked, points
revealed through a moving funnel aperture), giving Experiments 1–4. Blue outlines mark
lasso experiments, orange outlines anchor experiments; grey fill marks the masked
conditions. **b**, One of the eight stimulus arrays. Points are a hard-core Poisson process;
the gap statistic and BIC both select a single cluster, so no ground-truth partition
exists. **c**, The same array as partitioned by six different participants; filled circles
are coloured by the cluster the participant assigned them to, and *k* above each panel is
that participant's cluster count. **d**, Fowlkes–Mallows agreement. Black bars, internal
agreement (a participant's two presentations of the same array); grey bars,
inter-participant agreement (that participant against every other). Bars are means over
participants and error bars are 95% confidence intervals of that mean (1.96 × s.e.m.);
*n* = 67, 71, 67 and 74 participants for Experiments 1–4. Brackets give two-sided
Wilcoxon signed-rank tests paired within participant: P = 5 × 10⁻¹², 3 × 10⁻⁹, 9 × 10⁻¹²
and 8 × 10⁻¹¹. Black and grey dashed lines mark the internal (0.76) and inter-participant
(0.63) values reported by Marupudi and Varma for the same paradigm.

*(≈210 words)*

---

## Fig. 2 | Two invariant slopes, reallocated by response format.

**a**, Number of clusters against array set size, and **b**, cluster size (points per
cluster) against set size. Symbols are condition means at each set size and error bars are
s.e.m. across trials; filled circles with solid lines are full-view experiments, open
squares with dashed lines are masked experiments, blue is the lasso format and orange the
anchor format. Text beside each curve gives the per-point slope β from a linear
mixed-effects model with participant and stimulus as crossed random effects, and its exact
two-sided P; *n* = 7,498, 3,976, 7,504 and 4,148 trials for Experiments 1–4. Every slope is
positive in every cell. **c**, Change in each per-point slope attributable to response
format (anchor minus lasso) and to visibility (masked minus full), from a single pooled
model. Filled circles are the number-of-clusters slope, red diamonds the cluster-size
slope; error bars are 95% confidence intervals and P values are two-sided Wald tests.
Format moves the two slopes in opposite directions (β = −0.046, P = 4 × 10⁻¹⁰⁹ and
β = +0.101, P < 1 × 10⁻³⁰⁰); visibility does not (β = −0.013, P = 6 × 10⁻⁹ and β = −0.004,
P = 0.082). **d**, Test–retest of each participant's median cluster count across two
presentations of the same array, on log₂ axes; blue circles, lasso (Experiment 1,
*n* = 67 participants); orange circles, anchor (Experiment 3, *n* = 67). Points are jittered
multiplicatively to separate ties, *r* is the Pearson correlation on the unjittered values
and the grey dashed line is the identity.

*(≈235 words)*

---

## Fig. 3 | Humans satisfice rather than optimize.

**a**, Response time against set size on log–log axes for the two full-view experiments.
Filled circles are geometric means and error bars are s.e.m. of the log-transformed times;
lines are the fitted power laws. Text gives the scaling exponent *b* with its 95%
confidence interval and the exact two-sided P against *b* = 1 (bootstrap over participants;
*n* = 67 participants and 7,498 trials for lasso, 67 and 7,504 for anchor). The grey dashed
line shows the linear reference. **b**, Scaling exponents after decomposing response time
into drawing (motor) and deliberation (non-drawing) components. Filled circle, lasso total;
filled square, lasso drawing; open diamond, lasso deliberation; filled diamond, anchor
total. Error bars are 95% confidence intervals; the red dashed line marks *b* = 1.
Deliberation alone remains strongly sublinear, so the boundedness is not motor.
**c**, Silhouette (left) and number of clusters (right) for human partitions and for GMM and
DBSCAN fitted to the same arrays. Bars are means over the eight held-out stimuli, open
circles are the individual stimuli and error bars are 95% confidence intervals of the mean
across stimuli. Brackets give two-sided Wilcoxon signed-rank tests paired **by stimulus**,
with the number of complete pairs in parentheses: silhouette, P = 0.69 (*n* = 6) versus GMM
and P = 0.16 (*n* = 7) versus DBSCAN; cluster count, P = 0.11 and P = 0.25 (*n* = 8). The
stimulus is the unit of analysis because both algorithms are deterministic and return one
value per stimulus. **d**, Standardized mixed-model effects on trial-wise confidence. Blue
circles, lasso (Experiment 2); orange squares, anchor (Experiment 4); black diamonds,
pooled. Error bars are 95% confidence intervals and P values are two-sided. Own-partition
silhouette predicts confidence (pooled β = 0.26, P = 1 × 10⁻⁷¹) whereas agreement with the
group consensus does not (pooled β = −0.01, P = 0.33).

*(≈285 words)*

---

## Fig. 4 | A neural model reverse-engineers the learned human prior.

**a**, Model schematic. A point array is embedded by a shared multilayer perceptron, passed
through *L* permutation-equivariant set-interaction blocks that combine each point with a
global mean/max pool, and read out by a per-point softmax over *K* slots, so the cluster
count is emergent rather than specified. A gated recurrent unit over the reveal order is
added for the masked experiments only. **b**, Agreement with held-out human partitions
(adjusted Rand index) for the model trained on human partitions (Model-H, blue) and for the
same architecture trained on DBSCAN, GMM and shuffled targets. Bars are means over the
eight held-out stimuli, error bars are 95% confidence intervals of that mean, and P values
are two-sided Wilcoxon signed-rank tests paired by stimulus against Model-H (*n* = 8
stimuli, 3,295 trials). Orange and green dashed lines mark each algorithm's own agreement
with humans. **c**, Mean cluster count (left) and silhouette (right). Bars are means over the
eight stimuli, open circles the individual stimuli, error bars 95% confidence intervals,
and P values two-sided stimulus-paired Wilcoxon tests against Human; the black dashed line
marks the human value. Model-H reproduces the human silhouette (P = 0.64), a statistic that
never enters its loss, whereas the GMM- and shuffle-target models do not (P = 0.016 and
P = 0.008). **d**, Top, the same agreement measure for one person predicting another, for
Model-H, and for the leave-one-out consensus of all other participants; bars, error bars and
tests as in **b**. Bottom, mean agreement between two human partitions as a function of the
difference in their cluster counts. All 677,123 within-stimulus participant pairs were
computed; the 629,205 pairs differing by five clusters or fewer (92.9%) are plotted.
Circles are means and error bars are s.e.m. across the eight stimuli, which is the
clustered unit, not across pairs.

*(≈280 words)*

---

## Fig. 5 | The cluster-count slope reflects a graded capacity limit, not architecture.

**a**, Recovered cluster-count slope against agreement with humans for six architecture
variants, three training seeds each. Circles darken with increasing interaction richness,
from a plain per-point baseline to full attention; the red dashed line marks the human
slope. Richer architectures raise agreement but leave the recovered slope near zero.
**b**, Linear-probe agreement with human partitions at each hidden block and at the output,
for three architectures. Circles joined by lines are hidden blocks and squares are the
output layer; points are means over three training seeds and error bars are s.e.m. across
those seeds. The grey dashed line is the raw-coordinate baseline, obtained by clustering the
stimulus coordinates alone at the human cluster count. Every block of every architecture
exceeds that baseline on 8 of 8 stimuli, two-sided stimulus-paired Wilcoxon P = 0.008.
**c**, Two-slope trade-off from a dose–response sweep of the per-group penalty weight λ (10
values, 5 seeds each). Circles are per-λ means coloured by λ, error bars are s.e.m. across
seeds on both axes, and the grey line and shaded region show the achievable frontier.
Raising the cluster-count slope forces the cluster-size slope down; the red star, the human
value on both slopes, lies outside that frontier (gap = 0.085, 95% confidence interval
0.082–0.088, BF₁₀ ≈ 1.9 × 10⁴ against a pre-registered margin of 0.05), so no single fixed
capacity reproduces both.

*(≈215 words)*

---

## Cross-checks against the AIP figure requirements

| Requirement | Status |
|---|---|
| Cited in sequence as Fig. 1, Fig. 2 … in the main text | to verify in `manucluster_cyg1.docx` — not controlled by this directory |
| ≥ 300 dpi | 300 dpi PNG; PDF is vector |
| ≤ 180 mm wide | all five are exactly 180.0 mm (enforced by `nhbstyle.save`) |
| 5–7 pt sans-serif labelling | Arial; every label 5.0–7.0 pt via `nhbstyle.FS`; panel letters 8 pt bold |
| Symbol font for Greek | live Unicode Arial text in the PDF, restylable without redrawing |
| Scale bars not magnification | not applicable — no micrographs |
| Error bars included | every panel with a point estimate carries one, defined above |
| Labels not flattened onto images | all text is live vector text (`pdf.fonttype = 42`) |
| Legend: centre value, error bars, n, test, P | given for every panel above |
| Legend: verbal cues, not symbols | keys described as "open circles", "red dashed line", etc. |
| Legend: no methodological detail | model fitting and stimulus generation left to Methods |
