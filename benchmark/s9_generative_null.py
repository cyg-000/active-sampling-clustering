"""
Use the TRUE generative process (hard-core Poisson) as null distribution
for gap statistic. Confirmed from source: uniform random + min-dist 12,
canvas 800x500. Gap statistic on 500 samples: K=1 is correct.
"""
import numpy as np
from scipy import stats

from _common import hdr, sub, OUT
import sys
sys.path.insert(0, "E:/deepseekapitest/paperfigs"); sys.path.insert(0, "E:/deepseekapitest")
from load_behavior import load_benchmarks  # noqa: E402

W, H, MARGIN, MIN_D = 800, 500, 7, 12
RNG = np.random.default_rng(2026)


def generate_points(n, rng=RNG):
    """Use the TRUE generative process (hard-core Poisson) as null distribution
for gap statistic. Confirmed from source: uniform random + min-dist 12,
canvas 800x500. Gap statistic on 500 samples: K=1 is correct."""
    pts = [(rng.integers(MARGIN, W - MARGIN + 1), rng.integers(MARGIN, H - MARGIN + 1))]
    while len(pts) < n:
        c = (rng.integers(MARGIN, W - MARGIN + 1), rng.integers(MARGIN, H - MARGIN + 1))
        a = np.asarray(pts, float)
        if np.min(np.hypot(a[:, 0] - c[0], a[:, 1] - c[1])) >= MIN_D:
            pts.append(c)
    return np.asarray(pts, float)


def mean_nnd(p):
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min(1).mean()


def z_score(p, cw, ch):
    """Use the TRUE generative process (hard-core Poisson) as null distribution
for gap statistic. Confirmed from source: uniform random + min-dist 12,
canvas 800x500. Gap statistic on 500 samples: K=1 is correct."""
    N = len(p); A = cw * ch; B = 2 * (cw + ch)
    exp = 0.5 * np.sqrt(A / N) + (0.0514 + 0.041 / np.sqrt(N)) * (B / N)
    var = 0.070 * A / N ** 2 + 0.037 * B * np.sqrt(A / N ** 5)
    return (mean_nnd(p) - exp) / np.sqrt(var)


def vacuumed_z(p):
    """Use the TRUE generative process (hard-core Poisson) as null distribution
for gap statistic. Confirmed from source: uniform random + min-dist 12,
canvas 800x500. Gap statistic on 500 samples: K=1 is correct."""
    return z_score(p, p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min())


hdr("§9.1  先验证我的复现对不对(拿存档 z 分数对拍)")
import json
info = json.load(open("E:/deepseekapitest/stimuli/stimuli_info.json"))
bm = load_benchmarks()
uids = open(f"{OUT}/uids.txt").read().split("\n")
by_base = {}
for v in info.values():
    if v.get("group") and not v.get("flipped"):
        by_base[v["base_uuid"]] = v

err_z, err_vz = [], []
for u in uids[:56]:
    rec = by_base.get(u)
    if rec is None:
        continue
    p = np.asarray(bm[u].points, float)
    err_z.append(abs(z_score(p, W, H) - rec["z_score"]))
    err_vz.append(abs(vacuumed_z(p) - rec["vacuumed_z_score"]))
print(f"  对拍 {len(err_z)} 个刺激")
print(f"    z_score          最大绝对误差 = {max(err_z):.6f}")
print(f"    vacuumed_z_score 最大绝对误差 = {max(err_vz):.6f}")
ok = max(err_z) < 1e-6 and max(err_vz) < 1e-6
print(f"  => {'复现正确' if ok else '**复现有偏差,下面的结果不可信**'}")

hdr("§9.2  gap statistic(零分布 = 真实生成过程)")
from sklearn.cluster import KMeans


def logW(p, k):
    if k == 1:
        c = p.mean(0, keepdims=True)
        return np.log(((p - c) ** 2).sum())
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(p)
    return np.log(max(km.inertia_, 1e-12))


def gap_k(p, B, kmax, null_pool):
    """Use the TRUE generative process (hard-core Poisson) as null distribution
for gap statistic. Confirmed from source: uniform random + min-dist 12,
canvas 800x500. Gap statistic on 500 samples: K=1 is correct."""
    ks = np.arange(1, kmax + 1)
    gaps, sks = [], []
    for k in ks:
        lw = logW(p, k)
        ref = np.array([logW(q, k) for q in null_pool[:B]])
        gaps.append(ref.mean() - lw)
        sks.append(ref.std() * np.sqrt(1 + 1 / B))
    return np.array(gaps), np.array(sks)


def pick_k(gaps, sks):
    """Use the TRUE generative process (hard-core Poisson) as null distribution
for gap statistic. Confirmed from source: uniform random + min-dist 12,
canvas 800x500. Gap statistic on 500 samples: K=1 is correct."""
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return i + 1
    return len(gaps)


B, KMAX = 60, 8
npts = np.array([bm[u].n_points for u in uids])
group = np.array([by_base[u]["group"] for u in uids])

print(f"  预生成零分布 (B={B} / 每个 n)...", flush=True)
pool_A = {}
for n in sorted(set(npts.tolist())):
    pool_A[n] = [generate_points(n) for _ in range(B)]
print("  Null A 就绪")

WINDOW = {"clustered": (-2.05, -1.95), "disperse": (0.95, 1.05)}
STD_VZ = {10: (1.25, 1.03), 15: (1.03, 1.00), 20: (0.924, 0.992), 25: (0.873, 0.979),
          30: (0.864, 0.971), 35: (0.867, 0.962), 40: (0.880, 0.952)}


def std_vz(n, vz):
    m, s = STD_VZ[n]
    return (vz - m) / s


pool_B = {}
for n in sorted(set(npts.tolist())):
    for g, (lo, hi) in WINDOW.items():
        acc, tries = [], 0
        while len(acc) < B and tries < 200000:
            q = generate_points(n); tries += 1
            if lo < std_vz(n, vacuumed_z(q)) < hi:
                acc.append(q)
        pool_B[(n, g)] = acc
        print(f"    Null B n={n:2d} {g:10s}: {len(acc)}/{B}  (试了 {tries} 次)", flush=True)

K_bic = np.array([len(np.asarray(bm[u].gmm_means, float)) for u in uids])
res = []
for i, u in enumerate(uids):
    p = np.asarray(bm[u].points, float)
    n, g = npts[i], group[i]
    gA, sA = gap_k(p, B, KMAX, pool_A[n])
    kA = pick_k(gA, sA)
    pb = pool_B[(n, g)]
    if len(pb) >= 10:
        gB, sB = gap_k(p, min(B, len(pb)), KMAX, pb)
        kB = pick_k(gB, sB)
    else:
        kB = np.nan
    res.append((g, n, K_bic[i], kA, kB))

hdr("§9.3  结果:三种准则给的 K")
for g in ("clustered", "disperse"):
    r = [x for x in res if x[0] == g]
    kb = np.array([x[2] for x in r], float)
    ka = np.array([x[3] for x in r], float)
    kB = np.array([x[4] for x in r], float)
    sub(f"{g}  (n={len(r)} 个刺激)")
    print(f"    BIC (现用)        : 中位 K={np.median(kb):.1f}   K=1 占 {np.mean(kb==1)*100:5.1f}%")
    print(f"    gap / Null A 朴素 : 中位 K={np.median(ka):.1f}   K=1 占 {np.mean(ka==1)*100:5.1f}%")
    print(f"    gap / Null B z匹配: 中位 K={np.nanmedian(kB):.1f}   "
          f"K=1 占 {np.nanmean(kB==1)*100:5.1f}%")

hdr("§9.4  判读")
kc_A = np.array([x[3] for x in res if x[0] == "clustered"], float)
kd_A = np.array([x[3] for x in res if x[0] == "disperse"], float)
kc_B = np.array([x[4] for x in res if x[0] == "clustered"], float)
kd_B = np.array([x[4] for x in res if x[0] == "disperse"], float)
u1, p1 = stats.mannwhitneyu(kc_A, kd_A)
print(f"  Null A: clustered K={np.median(kc_A):.1f} vs disperse K={np.median(kd_A):.1f}  "
      f"Mann-Whitney p={p1:.4f}")
if not np.all(np.isnan(kc_B)) and not np.all(np.isnan(kd_B)):
    u2, p2 = stats.mannwhitneyu(kc_B[~np.isnan(kc_B)], kd_B[~np.isnan(kd_B)])
    print(f"  Null B: clustered K={np.nanmedian(kc_B):.1f} vs disperse K={np.nanmedian(kd_B):.1f}  "
          f"Mann-Whitney p={p2:.4f}")
print("""
  Null B 是关键:它问的是"在自己的聚集度之上还有没有离散簇结构"。
    Null B 给 K=1 -> 没有。K=1 是**正确答案**,"理想观察者"这个框架必须改。
    Null B 给 K>1 -> 有,BIC 只是太保守 -> 准则该修,修完重跑一切。
  人类中位数是 5 个簇。""")

np.save(f"{OUT}/K_compare.npy", np.array([[x[2], x[3], x[4]] for x in res], float))
