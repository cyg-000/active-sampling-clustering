"""Matched-K operational ideal-observer benchmarks (CPU only).

The observer is given the same point coordinates and the human participant's
cluster count K.  It then searches for the best partition under two explicit
global objectives: mean silhouette and normalized within-cluster compactness.
This removes the cluster-count advantage that an unconstrained optimizer would
otherwise have.  Because exact partition search is combinatorial, results are
labelled *best found*, not a proof of the mathematical global optimum.
"""
from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")

try:
    from sklearn.cluster import AgglomerativeClustering, KMeans
except ImportError:  # pragma: no cover - retained for minimal CPU environments
    AgglomerativeClustering = KMeans = None

from .common import DEFAULT_DATASET, EXPERIMENTS, load_dataset, stimulus_split


OUT_DIR = Path(__file__).resolve().parent / "outputs"


def pairwise_distance(x: np.ndarray) -> np.ndarray:
    z = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(z * z, axis=2))


def silhouette_score(distance: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, int)
    groups = np.unique(labels)
    if len(groups) < 2 or len(labels) < 3:
        return np.nan
    values = np.zeros(len(labels), float)
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not np.any(same):
            values[i] = 0.0
            continue
        a = float(np.mean(distance[i, same]))
        b = min(float(np.mean(distance[i, labels == g])) for g in groups if g != labels[i])
        values[i] = (b - a) / max(a, b, 1e-12)
    return float(np.mean(values))


def compactness_score(x: np.ndarray, labels: np.ndarray) -> float:
    """One minus within-cluster SSE / total SSE (larger is better)."""
    total = float(np.sum((x - np.mean(x, axis=0)) ** 2))
    if total <= 1e-12:
        return np.nan
    within = 0.0
    for g in np.unique(labels):
        p = x[labels == g]
        within += float(np.sum((p - np.mean(p, axis=0)) ** 2))
    return 1.0 - within / total


def _kmeans_once(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    centers = [x[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min(np.sum((x[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2), axis=1)
        if float(np.sum(d2)) <= 1e-12:
            remaining = [i for i in range(n) if not any(np.allclose(x[i], c) for c in centers)]
            centers.append(x[remaining[0] if remaining else int(rng.integers(n))])
        else:
            centers.append(x[int(rng.choice(n, p=d2 / np.sum(d2)))])
    centers = np.asarray(centers, float)
    labels = np.zeros(n, int)
    for _ in range(100):
        new = np.argmin(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        # Repair empty clusters using points farthest from their assigned centre.
        for g in range(k):
            if not np.any(new == g):
                loss = np.sum((x - centers[new]) ** 2, axis=1)
                donor = int(np.argmax(loss))
                new[donor] = g
        updated = np.asarray([np.mean(x[new == g], axis=0) for g in range(k)])
        if np.array_equal(new, labels) and np.allclose(updated, centers):
            labels = new
            break
        labels, centers = new, updated
    return labels


def _ordered_partition(order: np.ndarray, k: int) -> np.ndarray:
    labels = np.empty(len(order), int)
    for g, block in enumerate(np.array_split(order, k)):
        labels[block] = g
    return labels


def candidate_partitions(x: np.ndarray, k: int, seed: int, n_init: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if KMeans is not None:
        candidates = [KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                             random_state=seed).fit_predict(x)]
        for linkage in ("ward", "complete", "average", "single"):
            candidates.append(AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(x))
    else:
        candidates = [_kmeans_once(x, k, rng) for _ in range(n_init)]
    centered = x - np.mean(x, axis=0)
    angle = np.arctan2(centered[:, 1], centered[:, 0])
    radius = np.sqrt(np.sum(centered * centered, axis=1))
    for order in (np.argsort(x[:, 0]), np.argsort(x[:, 1]), np.argsort(angle), np.argsort(radius)):
        candidates.append(_ordered_partition(order, k))
    return candidates


def greedy_refine(
    x: np.ndarray,
    labels: np.ndarray,
    objective: str,
    distance: np.ndarray,
    max_passes: int = 15,
) -> tuple[np.ndarray, float]:
    labels = labels.copy()
    score_fn = (lambda z: silhouette_score(distance, z)) if objective == "silhouette" else (lambda z: compactness_score(x, z))
    score = score_fn(labels)
    k = len(np.unique(labels))
    for _ in range(max_passes):
        changed = False
        for i in range(len(labels)):
            old = int(labels[i])
            if np.sum(labels == old) <= 1:
                continue
            best_group, best_score = old, score
            for g in range(k):
                if g == old:
                    continue
                labels[i] = g
                candidate = score_fn(labels)
                if candidate > best_score + 1e-10:
                    best_group, best_score = g, candidate
            labels[i] = best_group
            if best_group != old:
                score, changed = best_score, True
        if not changed:
            break
    return labels, float(score)


def best_found(
    x: np.ndarray,
    k: int,
    seed: int,
    n_init: int,
    refine_top: int,
    refine_passes: int,
) -> dict[str, float]:
    distance = pairwise_distance(x)
    candidates = candidate_partitions(x, k, seed, n_init)
    out = {}
    for objective in ("silhouette", "compactness"):
        best = -np.inf
        # Refine every deterministic candidate and the strongest K-means starts.
        ranked = sorted(
            candidates,
            key=lambda z: silhouette_score(distance, z) if objective == "silhouette" else compactness_score(x, z),
            reverse=True,
        )[:min(refine_top, len(candidates))]
        for labels in ranked:
            _, score = greedy_refine(x, labels, objective, distance,
                                     max_passes=refine_passes)
            best = max(best, score)
        out[objective] = float(best)
    return out


def run(
    dataset: Path,
    scope: str,
    n_init: int,
    max_trials: int | None,
    seed: int,
    refine_top: int,
    refine_passes: int,
) -> tuple[Path, Path]:
    d = load_dataset(dataset)
    idx = np.arange(len(d["exp"])) if scope == "all" else stimulus_split(d)["test"]
    if max_trials is not None:
        idx = idx[:max_trials]
    cache: dict[tuple[bytes, int], dict[str, float]] = {}
    rows = []
    for count, j in enumerate(idx, 1):
        labels = np.asarray(d["human"][j], int)
        valid = labels >= 0
        x, human = np.asarray(d["pts"][j], float)[valid], labels[valid]
        groups = np.unique(human)
        k = len(groups)
        if k < 2 or k >= len(x):
            continue
        # Relabel only for stable bookkeeping; objective values are label invariant.
        human = np.searchsorted(groups, human)
        key = (np.round(x, 8).tobytes(), k)
        if key not in cache:
            cache[key] = best_found(
                x, k, seed + len(cache), n_init, refine_top, refine_passes
            )
        distance = pairwise_distance(x)
        human_sil = silhouette_score(distance, human)
        human_comp = compactness_score(x, human)
        ideal = cache[key]
        rows.append({
            "row_index": int(j), "exp": str(d["exp"][j]), "sid": str(d["sid"][j]),
            "base_uuid": str(d["base_uuid"][j]), "n_points": int(len(x)), "human_k": int(k),
            "human_silhouette": human_sil, "ideal_silhouette": ideal["silhouette"],
            "silhouette_gap": ideal["silhouette"] - human_sil,
            "human_compactness": human_comp, "ideal_compactness": ideal["compactness"],
            "compactness_gap": ideal["compactness"] - human_comp,
        })
        if count % 1000 == 0:
            print(f"processed {count}/{len(idx)} trials; {len(cache)} unique geometry-K problems", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_path = OUT_DIR / f"matched_k_ideal_{scope}.csv"
    with trial_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for exp in (*EXPERIMENTS, "pooled"):
        rr = rows if exp == "pooled" else [r for r in rows if r["exp"] == exp]
        if not rr:
            continue
        sg = np.asarray([r["silhouette_gap"] for r in rr], float)
        cg = np.asarray([r["compactness_gap"] for r in rr], float)
        summary.append({
            "condition": exp, "n_trials": len(rr),
            "silhouette_gap_mean": float(np.mean(sg)), "silhouette_gap_median": float(np.median(sg)),
            "within_0.02_silhouette_optimum": float(np.mean(sg <= 0.02)),
            "compactness_gap_mean": float(np.mean(cg)), "compactness_gap_median": float(np.median(cg)),
            "within_0.02_compactness_optimum": float(np.mean(cg <= 0.02)),
        })
    summary_path = OUT_DIR / f"matched_k_ideal_{scope}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(f"wrote {trial_path}\nwrote {summary_path}")
    return trial_path, summary_path


def clean_existing(dataset: Path, scope: str) -> tuple[Path, Path]:
    """Drop sessions absent from the audited dataset and rebuild the summary."""
    d = load_dataset(dataset)
    allowed = set(np.asarray(d["sid"]).astype(str))
    trial_path = OUT_DIR / f"matched_k_ideal_{scope}.csv"
    with trial_path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["sid"] in allowed]
    numeric = ["row_index", "n_points", "human_k", "human_silhouette", "ideal_silhouette",
               "silhouette_gap", "human_compactness", "ideal_compactness", "compactness_gap"]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    with trial_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for exp in (*EXPERIMENTS, "pooled"):
        rr = rows if exp == "pooled" else [r for r in rows if r["exp"] == exp]
        if not rr:
            continue
        sg = np.asarray([r["silhouette_gap"] for r in rr]); cg = np.asarray([r["compactness_gap"] for r in rr])
        summary.append({"condition": exp, "n_trials": len(rr),
                        "silhouette_gap_mean": np.mean(sg), "silhouette_gap_median": np.median(sg),
                        "within_0.02_silhouette_optimum": np.mean(sg <= .02),
                        "compactness_gap_mean": np.mean(cg), "compactness_gap_median": np.median(cg),
                        "within_0.02_compactness_optimum": np.mean(cg <= .02)})
    summary_path = OUT_DIR / f"matched_k_ideal_{scope}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(f"retained {len(rows)} existing ideal-observer trials under audited dataset")
    print(f"rewrote {trial_path}\nrewrote {summary_path}")
    return trial_path, summary_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--scope", choices=("all", "model_test"), default="all")
    ap.add_argument("--n-init", type=int, default=8,
                    help="K-means++ restarts; use 24 for the slower sensitivity run")
    ap.add_argument("--refine-top", type=int, default=1,
                    help="number of strongest candidates greedily refined")
    ap.add_argument("--refine-passes", type=int, default=0,
                    help="point-move refinement passes per selected candidate")
    ap.add_argument("--max-trials", type=int)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--clean-existing", action="store_true",
                    help="filter existing trial output using audited dataset; do not re-optimize")
    args = ap.parse_args()
    if args.clean_existing:
        clean_existing(args.dataset, args.scope)
    else:
        run(args.dataset, args.scope, args.n_init, args.max_trials, args.seed,
            args.refine_top, args.refine_passes)


if __name__ == "__main__":
    main()
