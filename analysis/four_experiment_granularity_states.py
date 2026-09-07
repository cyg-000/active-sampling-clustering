"""Build the granularity-state table used by Layer 1.

For each trial we observe the terminal cluster count K_final and the array
size N (and stimulus group). We expand the trial into a sequence of
granularity states:

  (K=2, stop=0)         if K_final >= 3
  (K=3, stop=0)         if K_final >= 4
  ...
  (K=K_final-1, stop=0)
  (K=K_final, stop=1)

K=2 is the legal start state; we never emit a K=2 stop observation
(except in the trivial case K_final=2, which is rare).

Each row carries:
  - visibility  (full / aperture)
  - format      (lasso / anchor)
  - experiment  (exp1..exp4)
  - sid         (de-identified participant)
  - base_uuid   (the stimulus)
  - n_points    (N)
  - K           (current granularity state, integer >= 2)
  - stop        (1 iff K == K_final else 0)
  - group       (clustered / disperse)

Output: outputs/granularity_hazard.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed_trials.npz"
OUT = HERE / "outputs"


def load_trials() -> pd.DataFrame:
    d = np.load(DATA, allow_pickle=True)
    # K_final = number of unique cluster ids (>= 2 by construction of the task)
    K_final = np.array([len(np.unique(h)) for h in d["human"]], dtype=int)
    df = pd.DataFrame({
        "sid": d["sid"],
        "base_uuid": d["base_uuid"],
        "exp": d["exp"],
        "n_points": d["n_points"].astype(int),
        "K": K_final,
        "visibility": d["visibility"],
        "format": d["response_type"],
        "group": d["group"],
    })
    return df


def expand_to_states(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (trial, K) with K in [2, K_final]. stop=1 iff K==K_final."""
    # vectorised: for each trial, emit K_final-2 states (K-1) and one terminal state K_final.
    K_final = df["K"].to_numpy()
    base_K = np.repeat(K_final - 2, K_final - 1).clip(min=0)
    rows_K = []
    rows_stop = []
    for kf in K_final:
        if kf < 3:
            # degenerate: only emit a single (K=2, stop=1) row
            rows_K.append(np.array([2], dtype=int))
            rows_stop.append(np.array([1], dtype=int))
        else:
            # emit K_final rows: K=2..K_final-1 stop=0, K=K_final stop=1
            arr_K = np.arange(2, kf + 1, dtype=int)
            arr_stop = np.zeros_like(arr_K)
            arr_stop[-1] = 1
            rows_K.append(arr_K)
            rows_stop.append(arr_stop)
    K_arr = np.concatenate(rows_K)
    S_arr = np.concatenate(rows_stop)
    rep = np.repeat(df.index.to_numpy(), [len(a) for a in rows_K])
    out = df.iloc[rep].reset_index(drop=True).copy()
    out["K_state"] = K_arr
    out["stop"] = S_arr.astype(int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "granularity_hazard.csv"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_trials()
    print(f"loaded {len(df)} trials")
    print(df.groupby(["exp", "visibility", "format"]).agg(
        n=("K", "size"), K_min=("K", "min"), K_max=("K", "max"),
        K_mean=("K", "mean"), K_median=("K", "median")))

    states = expand_to_states(df)
    print(f"expanded to {len(states)} granularity states")
    print("stop rate per experiment (should approach 1 / K_mean):")
    summary = states.groupby("exp").agg(
        n_states=("stop", "size"),
        stop_rate=("stop", "mean"),
        K_max=("K_state", "max"),
    )
    print(summary)
    states.to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()