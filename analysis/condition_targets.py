"""Build four condition-specific human targets on full and model-test sets.

The estimand is explicit: average participants within each base stimulus, give
each base stimulus equal weight, then fit an OLS slope across stimuli.  CIs use
a crossed participant x stimulus bootstrap.  This is the target that model
predictions should reproduce; it is intentionally distinct from the population
mixed-effects estimand reported in the behavioural Results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .common import (
    CONDITION_META,
    DEFAULT_DATASET,
    EXPERIMENTS,
    condition_mask,
    load_dataset,
    participant_stimulus_bootstrap,
    stimulus_equal_summary,
    stimulus_split,
)


HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--out", default=str(HERE / "outputs" / "condition_targets.csv"))
    args = ap.parse_args()

    d = load_dataset(args.dataset)
    splits = stimulus_split(d)
    scopes = {"all": list(range(len(d["exp"]))), "model_test": splits["test"]}
    rows = []
    for scope, scope_idx in scopes.items():
        for offset, exp in enumerate(EXPERIMENTS):
            idx = condition_mask(d, exp, scope_idx)
            point = stimulus_equal_summary(d, idx)
            ci = participant_stimulus_bootstrap(
                d, idx, n_boot=args.bootstrap, seed=args.seed + offset + 100 * (scope == "model_test")
            )
            fmt, vis = CONDITION_META[exp]
            rows.append({
                "scope": scope,
                "experiment": exp,
                "response_format": fmt,
                "visibility": vis,
                "estimator": "stimulus-equal OLS after participant averaging",
                **point,
                "slope_k_ci_lo": ci["slope_k_vs_n"][0],
                "slope_k_ci_hi": ci["slope_k_vs_n"][1],
                "slope_size_ci_lo": ci["slope_size_vs_n"][0],
                "slope_size_ci_hi": ci["slope_size_vs_n"][1],
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("Condition-specific human targets")
    for r in rows:
        print(
            f"{r['scope']:10s} {r['experiment']} {r['response_format']:7s}/{r['visibility']:6s} "
            f"n={r['n_trials']:5d}, stimuli={r['n_stimuli']:2d}, "
            f"slope_k={r['slope_k_vs_n']:.4f} "
            f"[{r['slope_k_ci_lo']:.4f}, {r['slope_k_ci_hi']:.4f}], "
            f"slope_size={r['slope_size_vs_n']:.4f} "
            f"[{r['slope_size_ci_lo']:.4f}, {r['slope_size_ci_hi']:.4f}]"
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
