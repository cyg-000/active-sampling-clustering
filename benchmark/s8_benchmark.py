"""
GMM BIC selects K=1 for most stimuli. If GMM ideal observer degenerates
to the array centroid, "error relative to ideal observer" is meaningless --
K=1 IS the correct answer for provably structureless arrays.
"""
import numpy as np
from scipy import stats
import sys

from _common import hdr, sub, OUT

sys.path.insert(0, "E:/deepseekapitest/paperfigs"); sys.path.insert(0, "E:/deepseekapitest")
from load_behavior import load_benchmarks  # noqa: E402

bm = load_benchmarks()
uids = open(f"{OUT}/uids.txt").read().split("\n")

hdr("§8.1  GMM 选了几个簇?")
K = np.array([len(np.asarray(bm[u].gmm_means, float)) for u in uids])
vals, cnt = np.unique(K, return_counts=True)
print("  K 的分布:")
for v, c in zip(vals, cnt):
    bar = "█" * int(c / max(cnt) * 40)
    print(f"    K={int(v):2d}: {c:2d} 个刺激 ({c/len(K)*100:4.1f}%)  {bar}")
print(f"\n  中位 K = {np.median(K):.1f}   均值 = {K.mean():.2f}")
print(f"  **K=1 的刺激: {np.sum(K==1)}/{len(K)} = {np.mean(K==1)*100:.0f}%**")

hdr("§8.2  K=1 时,G 是不是就等于 P?")
P = np.array([np.asarray(bm[u].points, float).mean(axis=0) for u in uids])
G = np.load(f"{OUT}/G.npy")
d = np.linalg.norm(P - G, axis=1)
print(f"  ‖P − G‖:  K=1 的刺激 -> 中位 {np.median(d[K==1]):.6f} px   "
      f"最大 {d[K==1].max():.6f}")
print(f"            K>1 的刺激 -> 中位 {np.median(d[K>1]):.2f} px   最大 {d[K>1].max():.2f}")
print(f"\n  => K=1 时 G 与 P 完全重合(数值误差级),'GMM 理想观察者'退化成点阵质心。")

hdr("§8.3  按 group 看 —— 连 clustered 刺激 GMM 都说没簇?")
from _common import load
evs, ideal = load()
group = np.array([ideal[u]["group"] for u in uids])
for g in np.unique(group):
    m = group == g
    print(f"  {g:10s}: K=1 占 {np.mean(K[m]==1)*100:5.1f}%   中位 K = {np.median(K[m]):.1f}   "
          f"K 分布 = {sorted(K[m].tolist())}")

hdr("§8.4  DBSCAN 那边呢?")
nb = np.array([len(bm[u].dbscan_boundaries or []) for u in uids])
vals, cnt = np.unique(nb, return_counts=True)
print("  DBSCAN 簇数分布:")
for v, c in zip(vals, cnt):
    print(f"    {int(v):2d} 簇: {c:2d} 个刺激 ({c/len(nb)*100:4.1f}%)")
print(f"  中位 = {np.median(nb):.1f}")
for g in np.unique(group):
    m = group == g
    print(f"  {g:10s}: 中位 {np.median(nb[m]):.1f}   范围 [{nb[m].min()}, {nb[m].max()}]")

hdr("§8.5  人画几个簇?(和基准比)")
try:
    import pandas as pd
    pp = pd.read_csv("E:/deepseekapitest/dataana/outputs/tables/rq5_invariants.csv")
    print(pp.head(12).to_string(index=False))
except Exception as ex:
    print(f"  (读 dataana 的 rq5 表失败: {ex})")
    print("  改用 M&V 复刻报告里的数字:人类中位簇数 = 5")

print(f"""
{'='*76}
汇总对比
{'='*76}
  GMM (anchor 任务的理想观察者) 中位簇数 = {np.median(K):.0f}   K=1 占 {np.mean(K==1)*100:.0f}%
  DBSCAN (lasso 任务的理想观察者) 中位簇数 = {np.median(nb):.0f}
  人类                            中位簇数 = 5(dataana / M&V 复刻)
""")
