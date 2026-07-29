"""Panel b prep: per-trial silhouette + cluster count for human / GMM / DBSCAN,
computed on the same arrays from the model dataset. Saves a tidy CSV and prints
aggregate means to compare against the manuscript (human .369/k3.87,
GMM .402/k5.36, DBSCAN .507/k3.35)."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(__file__)
z = np.load(os.path.join(HERE, "..", "autodl", "arch", "dataset.npz"),
            allow_pickle=True)
pts_all = z["pts"]
lab = {"human": z["human"], "gmm": z["gmm"], "dbscan": z["dbscan"]}

rows = []
for i in range(len(pts_all)):
    P = np.asarray(pts_all[i], dtype=float)
    for ag in ("human", "gmm", "dbscan"):
        L = np.asarray(lab[ag][i]).ravel()
        k = len(np.unique(L))
        sil = np.nan
        if 2 <= k <= len(L) - 1:
            try:
                sil = silhouette_score(P, L)
            except Exception:
                sil = np.nan
        rows.append((i, ag, k, sil))

df = pd.DataFrame(rows, columns=["trial", "agent", "k", "sil"])
df.to_csv(os.path.join(HERE, "data_for_figs", "f3_silhouette_k.csv"), index=False)

print("agent      k_mean   sil_mean   sil_median   n_valid_sil")
for ag in ("human", "gmm", "dbscan"):
    g = df[df.agent == ag]
    print(f"{ag:10s} {g.k.mean():6.3f}   {g.sil.mean():7.3f}   "
          f"{g.sil.median():8.3f}   {g.sil.notna().sum()}")
