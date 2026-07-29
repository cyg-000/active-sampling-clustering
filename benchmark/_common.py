"""
Exploratory modules (not in paper). Python rather than R
because delta computation chain is in paperfigs/, cached as pickle.
"""
import os
import pickle
import sys

import numpy as np

ROOT = "E:/deepseekapitest"
PF = os.path.join(ROOT, "paperfigs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

CANVAS = np.array([800.0, 500.0])


def load():
    with open(os.path.join(PF, "outputs", "error_vectors.pkl"), "rb") as f:
        d = pickle.load(f)
    evs, ideal = d["evs"], d["ideal"]
    for e in evs:
        e["delta"] = np.asarray(e["delta"], float)
        e["task"] = "lasso" if e["response_type"] == "lasso" else "anchor"
    assert {e["task"] for e in evs} == {"lasso", "anchor"}, "unexpected task keys"
    return evs, ideal


def by_stim_task(evs, agg="mean"):
    """Exploratory modules (not in paper). Python rather than R
because delta computation chain is in paperfigs/, cached as pickle."""
    from collections import defaultdict
    d = defaultdict(list)
    for e in evs:
        d[(e["base_uuid"], e["task"])].append(e["delta"])
    return {k: np.asarray(v, float) for k, v in d.items()}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def hdr(s):
    print("\n" + "=" * 76)
    print(s)
    print("=" * 76)


def sub(s):
    print(f"\n── {s} " + "─" * max(0, 70 - len(s)))
