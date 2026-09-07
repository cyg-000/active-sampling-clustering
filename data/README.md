# Public analysis data

`processed_trials.npz` is the audited analysis dataset used by both the CPU
analyses and Model-H. It contains 23,066 rows, 278 arbitrary release participant
IDs, and 56 generated base stimuli. Loading requires
`numpy.load(path, allow_pickle=True)` because point arrays have variable length.

## Arrays

| Key | Meaning |
| --- | --- |
| `pts` | N × 2 point coordinates in the 800 × 500 stimulus canvas |
| `human` | human cluster label for each point; `-1` denotes unassigned |
| `reveal` | first reveal-event index for each point; `-1` in full view |
| `hover` | aperture cursor locations in temporal order |
| `gmm`, `dbscan` | deterministic comparison/lesion targets |
| `exp` | experiment (`exp1`–`exp4`) |
| `sid` | arbitrary release participant ID, unique within experiment |
| `base_uuid` | ID of one of the 56 generated base arrays |
| `group` | original stimulus-size group |
| `response_type` | `lasso` or `voronoi`/anchor response |
| `visibility` | `full` or `funnel`/aperture view |
| `n_points` | number of points in the array |
| `presentation` | repeat presentation index within participant and stimulus |
| `flipped` | whether the stored array used the mirrored presentation |

The larger CSVs in `revision_analysis/outputs/` are de-identified intermediate
tables. They retain only variables needed to refit the published stopping and RT
models or recompute stimulus-/participant-level inference. `person_hash` is a
fresh arbitrary release code—not a hash of a public identifier—and exists only
to reproduce the exclusion of people who participated in both anchor cohorts.

Raw browser event logs are withheld because they contained direct identifiers
and are not required for the public reconstruction. The maintainer-only
`tools/prepare_public_release.py` documents the transformation from the audited
private build to these public files.
