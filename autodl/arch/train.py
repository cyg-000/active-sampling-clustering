"""
Training loop for unattended GPU runs.
10 termination conditions: epoch limit, early stopping (val ARI, not FM),
max-hours timeout, NaN loss abort, etc. Every exit path writes status.json.
"""
import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

import config as C
from data import load_npz, make_splits
from gpu_data import batches, build_tensors, epoch_order
from losses import total_loss
from metrics import adjusted_rand, fowlkes_mallows
from model2 import AssignNet2, soft_assign

STOP = False


def _sigterm(sig, frm):
    """
    Training loop for unattended GPU runs.
    10 termination conditions: epoch limit, early stopping (val ARI, not FM),
    max-hours timeout, NaN loss abort, etc. Every exit path writes status.json.
    """
    global STOP
    if STOP:
        print("  second signal received, force-exit", flush=True)
        os._exit(130)
    STOP = True
    print("  received SIGTERM, will save after this epoch (press again to force-exit)", flush=True)


def finish(out, status, **kw):
    json.dump(dict(status=status, **kw), open(out / "status.json", "w"), indent=2)
    print(f"[{out.name}] status={status} {kw}", flush=True)
    return 0 if status in ("ok", "timeout", "interrupted") else 1


@torch.no_grad()
def evaluate(net, T, batch, amp):
    net.eval()
    fms, aris, losses = [], [], []
    for sel in batches(T, batch, False):
        x, y, m, o = T["X"][sel], T["Y"][sel], T["M"][sel], T["O"][sel]
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            logits = net(x, m, o)
        P = soft_assign(logits.float())
        l, _ = total_loss(P, x, y, m)
        losses.append(float(l))
        hard = P.argmax(-1).cpu().numpy()
        yy, mm = y.cpu().numpy(), m.cpu().numpy()
        for b in range(len(hard)):
            pr = np.where(mm[b], hard[b], -1)
            f, r = fowlkes_mallows(yy[b], pr), adjusted_rand(yy[b], pr)
            if np.isfinite(f):
                fms.append(f)
            if np.isfinite(r):
                aris.append(r)
    vl = float(np.mean(losses)) if losses else float("nan")
    vfm = float(np.mean(fms)) if fms else float("nan")
    vari = float(np.mean(aris)) if aris else float("nan")
    return vl, vfm, vari


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="human",
                    choices=["human", "gmm", "dbscan", "shuffle"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--scanpath", default="full", choices=["full", "none"])
    ap.add_argument("--split-by", default=C.SPLIT_BY, choices=["base_uuid", "sid"])
    ap.add_argument("--subset", default="all",
                    help="all | exp1..exp4 | lasso | voronoi | full | funnel")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--pairing", default="meanmax", choices=["meanmax", "dpool", "attn"],
                    help="meanmax=baseline (no pairwise); dpool=fixed-kernel distance pooling + local density; attn=distance-biased self-attention")
    ap.add_argument("--ncount", action="store_true",
                    help="provide explicit point-count feature -- second candidate cause of slope_k failure")
    ap.add_argument("--max-hours", type=float, default=C.MAX_HOURS)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    tag = a.tag or f"{a.target}_{a.subset}_{a.scanpath}_s{a.seed}"
    out = C.OUTPUT / tag
    out.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cuda = torch.cuda.is_available()
    dev = C.DEVICE if cuda else "cpu"
    amp = cuda and not a.no_amp
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"[{tag}] dev={dev} amp={amp} target={a.target} scanpath={a.scanpath} "
          f"batch={a.batch}", flush=True)

    d = load_npz()
    keep = np.ones(len(d["exp"]), bool)
    if a.subset in ("exp1", "exp2", "exp3", "exp4"):
        keep = d["exp"] == a.subset
    elif a.subset in ("lasso", "voronoi"):
        keep = d["response_type"] == a.subset
    elif a.subset in ("full", "funnel"):
        keep = d["visibility"] == a.subset
    elif a.subset != "all":
        return finish(out, "bad_args", msg=f"unknown subset {a.subset}")
    sp = {k: v[keep[v]] for k, v in make_splits(d, a.split_by).items()}
    print("  split:", {k: len(v) for k, v in sp.items()}, flush=True)

    for k, v in sp.items():
        if len(v) < a.batch // 4 or len(v) == 0:
            return finish(out, "empty_split", split=k, n=len(v))

    Ttr = build_tensors(d, sp["train"], a.target, a.scanpath, a.seed, dev)
    Tva = build_tensors(d, sp["val"], a.target, a.scanpath, a.seed, dev)
    Tte = build_tensors(d, sp["test"], a.target, a.scanpath, a.seed, dev)
    steps = math.ceil(Ttr["n"] / a.batch)
    print(f"  {steps} steps/epoch   VRAM resident "
          f"{sum(t.element_size()*t.nelement() for t in Ttr.values() if torch.is_tensor(t))/1e6:.1f} MB",
          flush=True)

    net = AssignNet2(pairing=a.pairing, use_ncount=a.ncount,
                     use_gru=not a.no_gru).to(dev)
    print(f"  arch: pairing={a.pairing}  ncount={a.ncount}  "
          f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=a.epochs * steps, pct_start=0.1)

    ep0, best, bad, nan_run, fm0 = 0, -1.0, 0, 0, None
    ck_path = out / "last.pt"
    if a.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location=dev)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        ep0, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        if ep0 >= a.epochs:
            return finish(out, "already_done", epoch=ep0, best_val_ARI=best)
        for _ in range(ep0 * steps):
            sched.step()
        print(f"  resume from ep{ep0} (best FM {best:.4f}), LR fast-forward to "
              f"{sched.get_last_lr()[0]:.2e})", flush=True)

    json.dump(vars(a), open(out / "config.json", "w"), indent=2)
    log = open(out / "log.jsonl", "a" if a.resume else "w")
    t_start, saved_best = time.time(), (out / "best.pt").exists()

    for ep in range(ep0, a.epochs):
        net.train()
        t0, tot, pacc = time.time(), [], {}
        O = epoch_order(Ttr, None)
        for sel in batches(Ttr, a.batch, True):
            x, y, m, o = Ttr["X"][sel], Ttr["Y"][sel], Ttr["M"][sel], O[sel]
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                logits = net(x, m, o)
            P = soft_assign(logits.float())
            loss, parts = total_loss(P, x, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP)
            opt.step(); sched.step()
            tot.append(float(loss))
            for k, v in parts.items():
                pacc[k] = pacc.get(k, 0.0) + v

        tl = float(np.mean(tot))
        if not math.isfinite(tl):
            return finish(out, "nan_loss", epoch=ep, best_val_ARI=best)

        vl, vfm, vari = evaluate(net, Tva, a.batch, amp)
        if fm0 is None:
            fm0 = vari
        nan_run = nan_run + 1 if not math.isfinite(vari) else 0
        if nan_run >= 10:
            return finish(out, "nan_eval", epoch=ep, best_val_ARI=best)

        rec = dict(epoch=ep, train_loss=tl, val_loss=vl, val_ARI=vari, val_FM=vfm,
                   lr=sched.get_last_lr()[0], sec=time.time() - t0,
                   **{f"p_{k}": v / steps for k, v in pacc.items()})
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"  ep{ep:3d}  train {tl:.4f}  val {vl:.4f}  ARI {vari:.4f}  "
              f"FM {vfm:.4f}  ({rec['sec']:.1f}s)", flush=True)

        if math.isfinite(vari) and vari > best + 1e-5:
            best, bad, saved_best = vari, 0, True
            torch.save(dict(model=net.state_dict(), epoch=ep, val_ARI=vari,
                            val_FM=vfm, args=vars(a)), out / "best.pt")
        else:
            bad += 1
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep, best=best, bad=bad),
                   ck_path)

        if ep >= C.SADDLE_BY and abs(vari - fm0) < 1e-9 and vl > 0.69:
            return finish(out, "stuck_saddle", epoch=ep, val_loss=vl, val_FM=vfm,
                          hint="reduce batch (or increase LR): escaping the saddle needs gradient update count")
        hrs = (time.time() - t_start) / 3600
        if bad >= C.PATIENCE and ep >= C.MIN_EPOCHS:
            print(f"  early stop @ ep{ep} (best {best:.4f})", flush=True); break
        if hrs > a.max_hours:
            return finish(out, "timeout", epoch=ep, hours=hrs, best_val_ARI=best)
        if STOP:
            return finish(out, "interrupted", epoch=ep, best_val_ARI=best)

    if not saved_best:
        return finish(out, "no_best", best_val_ARI=best)
    net.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    tl, tfm, tari = evaluate(net, Tte, a.batch, amp)
    print(f"[{tag}] TEST loss {tl:.4f} ARI {tari:.4f} FM {tfm:.4f}  "
          f"(best val ARI {best:.4f})", flush=True)
    return finish(out, "ok", test_loss=tl, test_ARI=tari, test_FM=tfm, best_val_ARI=best,
                  hours=(time.time() - t_start) / 3600)


if __name__ == "__main__":
    raise SystemExit(main())
