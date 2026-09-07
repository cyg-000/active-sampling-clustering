# Active sampling and perceptual clustering

Minimal public analysis package for the manuscript on human construction of
perceptual groups from random point arrays.

## What is included

- `data/processed_trials.npz`: de-identified trial-level responses from four
  experiments (23,066 trials; 278 public participant IDs; 56 base arrays).
- `analysis/`: behavioral reliability and scaling analyses, anchor-state
  stopping models, participant/stimulus-held-out generative tests, and hazard
  component ablations.
- `analysis/four_experiment_granularity_states.py`: constructs a common
  terminal-granularity state representation for all four experiments from the
  public trial file.
- `figures/`: final main-figure source scripts and source artwork.

Raw browser event logs are not distributed because they contain source
identifiers. Consequently, the public package includes the de-identified
anchor-state table used by the stopping analyses, but not the raw-event parser.
The lasso contour-level diagnostic is also excluded because its source event
stream cannot be safely released. No manuscript, reviewer correspondence,
model checkpoint, or private maintenance script is included.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Core analyses

Run commands from the repository root:

```bash
python -m analysis.behavioral_core
python -m analysis.condition_targets
python analysis/four_experiment_granularity_states.py
python -m analysis.stopping_hazard
python -m analysis.stats_quality_proxies
python -m analysis.process_oos --repeats 20 --folds 5
python -m analysis.process_ablation
```

The longer repeated-CV commands are deterministic given their documented
seeds. Existing aggregate outputs are included in `analysis/outputs/` so the
reported results can be inspected without rerunning them.

## Scope of the process model

The anchor stopping walk is a minimal generative-sufficiency test. Current
organization quality improves prediction of when participants stop, whereas
the generated cluster-count slope is carried mainly by the numerosity
dependence of the empirical stopping propensity. The model is not claimed to
reproduce absolute cluster-count calibration or a complete candidate-generation
mechanism.

## Data dictionary

See `data/README.md`. Participant identifiers in the public file are study-local
pseudonyms and cannot be linked back to recruitment-platform identities.

## License

Code and data are provided for peer review and non-commercial research use.
Please contact the authors before redistribution or use beyond replication of
the associated manuscript.
