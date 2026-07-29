"""
Build training dataset from raw dataexp1-4 JSON -> dataset.npz
Run LOCALLY (machine with raw JSON). Upload dataset.npz to GPU server.
Output: one record per trial, variable-length: points (N,2), human/GMM/DBSCAN
labels, and metadata. GMM/DBSCAN are for lesion targets only.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("CLUSTER_ROOT", "E:/deepseekapitest")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.npz")

# (exp, glob, trial_type, response_type, visibility)
SETS = [
    ("exp1", "dataexp1/data/*.json", "clustering", "lasso", "full"),
    ("exp2", "dataexp2/*.json", "funnel-clustering", "lasso", "funnel"),
    ("exp3", "dataexp3/data/*.json", "voronoi-clustering", "voronoi", "full"),
    ("exp4", "dataexp4/*.json", "funnel-voronoi-clustering", "voronoi", "funnel"),
]


def partition_lasso(cd, n):
    """Build training dataset from raw dataexp1-4 JSON -> dataset.npz
Run LOCALLY (machine with raw JSON). Upload dataset.npz to GPU server.
Output: one record per trial, variable-length: points (N,2), human/GMM/DBSCAN
labels, and metadata. GMM/DBSCAN are for lesion targets only."""
    hum = np.full(n, -1, np.int16)
    for ci, c in enumerate(cd.get("clusters") or []):
        pm = c.get("pointMembership") or []
        if len(pm) != n:
            continue
        for i, m in enumerate(pm):
            member = m.get("member") if isinstance(m, dict) else m
            if member is True and hum[i] < 0:
                hum[i] = ci
    return hum


def partition_voronoi(cd, pts):
    """Build training dataset from raw dataexp1-4 JSON -> dataset.npz
Run LOCALLY (machine with raw JSON). Upload dataset.npz to GPU server.
Output: one record per trial, variable-length: points (N,2), human/GMM/DBSCAN
labels, and metadata. GMM/DBSCAN are for lesion targets only."""
    n = len(pts)
    hum = np.full(n, -1, np.int16)
    pa = cd.get("pointAssignments")
    if pa:
        loc = {(round(p[0], 3), round(p[1], 3)): i for i, p in enumerate(pts)}
        for a in pa:
            p = a.get("point") or {}
            i = loc.get((round(float(p.get("x", -1)), 3), round(float(p.get("y", -1)), 3)))
            if i is not None:
                hum[i] = int(a.get("anchorIdx", -1))
        if (hum >= 0).sum() >= max(4, 0.5 * n):
            return hum
    anc = cd.get("anchors") or []
    if len(anc) < 2:
        return np.full(n, -1, np.int16)
    A = np.asarray([[a["x"], a["y"]] for a in anc], float)
    d = np.hypot(pts[:, None, 0] - A[None, :, 0], pts[:, None, 1] - A[None, :, 1])
    return d.argmin(1).astype(np.int16)


def algo_partitions(pts):
    """
    Build training dataset from raw dataexp1-4 JSON -> dataset.npz
    Run LOCALLY (machine with raw JSON). Upload dataset.npz to GPU server.
    Output: one record per trial, variable-length: points (N,2), human/GMM/DBSCAN
    labels, and metadata. GMM/DBSCAN are for lesion targets only.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.mixture import GaussianMixture
    n = len(pts)
    best, bic = None, np.inf
    for k in range(1, min(9, n)):
        try:
            g = GaussianMixture(k, covariance_type="full", random_state=0,
                                reg_covar=1e-3).fit(pts)
            b = g.bic(pts)
            if b < bic:
                bic, best = b, g.predict(pts).astype(np.int16)
        except Exception:
            continue
    if best is None:
        best = np.zeros(n, np.int16)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    eps = float(np.median(d.min(1)) * 2.0)
    db = DBSCAN(eps=eps, min_samples=2).fit_predict(pts).astype(np.int16)
    return best, db


def main():
    recs = []
    for exp, pat, tt, rtype, vis in SETS:
        files = sorted(glob.glob(os.path.join(ROOT, pat)))
        seen = {}
        n_ok = n_skip = 0
        for f in files:
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            sid = f"{exp}/{os.path.basename(f)}"
            for e in d.get("experimentData", []):
                if e.get("trial_type") != tt:
                    continue
                raw = e.get("clusteringData")
                if not isinstance(raw, str):
                    continue
                try:
                    cd = json.loads(raw)
                except Exception:
                    continue
                st = cd.get("stimulus") or {}
                uid = st.get("base_uuid")
                if st.get("practice_stimulus") or not st.get("group") or not uid \
                        or uid == "__detection__":
                    continue
                pl = st.get("points") or []
                n = len(pl)
                if n < 4:
                    continue
                pts = np.asarray([[p["x"], p["y"]] for p in pl], np.float32)

                hum = partition_lasso(cd, n) if rtype == "lasso" \
                    else partition_voronoi(cd, pts)
                ok = hum >= 0
                if ok.sum() < 4 or len(np.unique(hum[ok])) < 2:
                    n_skip += 1
                    continue
                lab = np.full(n, -1, np.int16)
                for k, v in enumerate(np.unique(hum[ok])):
                    lab[hum == v] = k

                rev = np.full(n, -1, np.int16)
                hov = np.zeros((0, 2), np.float32)
                if vis == "funnel":
                    he = cd.get("explorationHoverEvents") or []
                    hov = np.asarray([[h["x"], h["y"]] for h in he], np.float32) \
                        if he else np.zeros((0, 2), np.float32)
                    for t, ev in enumerate(he):
                        for i in ev.get("newlyRevealedIndices", []) or []:
                            if 0 <= i < n and rev[i] < 0:
                                rev[i] = t

                gmm, dbs = algo_partitions(pts.astype(float))
                key = (sid, uid)
                seen[key] = seen.get(key, 0) + 1
                recs.append(dict(
                    pts=pts, human=lab, reveal=rev, hover=hov, gmm=gmm, dbscan=dbs,
                    exp=exp, sid=sid, base_uuid=uid, n_points=n,
                    group=st.get("group", ""), flipped=bool(st.get("flipped")),
                    response_type=rtype, visibility=vis, presentation=seen[key]))
                n_ok += 1
        print(f"  {exp:5s} {rtype:8s}/{vis:6s}: {n_ok:5d} trials (skipped {n_skip})", flush=True)

    print(f"\nTotal {len(recs)} trials")
    keys = ["pts", "human", "reveal", "hover", "gmm", "dbscan"]
    save = {k: np.array([r[k] for r in recs], dtype=object) for k in keys}
    for k in ["exp", "sid", "base_uuid", "group", "response_type", "visibility"]:
        save[k] = np.array([r[k] for r in recs])
    for k in ["n_points", "presentation"]:
        save[k] = np.array([r[k] for r in recs], np.int32)
    save["flipped"] = np.array([r["flipped"] for r in recs], bool)
    np.savez_compressed(OUT, **save)
    print(f"-> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")

    import collections
    print("\nHuman cluster-count distribution:", dict(sorted(collections.Counter(
        [len(np.unique(r['human'][r['human'] >= 0])) for r in recs]).items())))
    print("Point-count distribution:", dict(sorted(collections.Counter(
        [r['n_points'] for r in recs]).items())))
    print("Orphan-point rate: %.3f" % np.mean([np.mean(r['human'] < 0) for r in recs]))


if __name__ == "__main__":
    sys.exit(main())
