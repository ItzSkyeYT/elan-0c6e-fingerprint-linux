"""
Descriptor matching v5 = v4 with a wider alignment search, plus fusion.

Two changes over v4:

  * v4 picked the rotation by descriptor vote mass and only then computed the
    top-K local-correlation score.  If the vote picked the wrong rotation the
    good score was never computed.  v5 evaluates the final score at every
    candidate alignment and maximises over them.
  * the Hough accumulator's top `npeak` peaks are all tried, not just the
    highest, so a genuine pair whose true offset came second is not lost.

Scores produced per pair:
    sqrt   vote mass / sqrt(nt*np)                (v1-style, for reference)
    topK   mean of the K best local patch correlations at the best alignment
    topKq  same, but each local correlation is taken over a +/- ltol snap
           window (elastic) -- this is the headline score
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desc_v3 import build_views, DEF3, H, W
from desc_v4 import _kp_index
from proto import load_all, scenario_A, scenario_B, summarise

TOPKS = (8, 12, 16, 24, 32)


def _peaks(pooled, npeak, sep=2):
    """Top `npeak` local maxima of the pooled accumulator, greedily separated."""
    flat = np.argsort(pooled, axis=None)[::-1]
    out = []
    for k in flat[:60]:
        by, bx = divmod(int(k), pooled.shape[1])
        if pooled[by, bx] <= 0:
            break
        if all(abs(by - y) >= sep or abs(bx - x) >= sep for y, x in out):
            out.append((by, bx))
            if len(out) >= npeak:
                break
    return out


def local_scores(S, vp, vt, TG, tyc, txc, tol):
    """Best descriptor similarity for each probe keypoint among template
    keypoints within `tol` px of its predicted position.  Returns the values for
    probe keypoints that actually land inside the template's covered area."""
    npb = vp.n
    py = vp.P[:, 0] + tyc
    px = vp.P[:, 1] + txc
    loc = np.full(npb, -2.0, np.float32)
    hitany = np.zeros(npb, bool)
    rows = np.arange(npb)
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            if dy * dy + dx * dx > tol * tol:
                continue
            yy, xx = py + dy, px + dx
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if not ok.any():
                continue
            tid = np.where(ok, TG[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)], -1)
            hit = tid >= 0
            if not hit.any():
                continue
            v = S[rows[hit], tid[hit]]
            loc[hit] = np.maximum(loc[hit], v)
            hitany |= hit
    return loc[hitany]


def match_pair(vt, vp, DT2, TG, cfg):
    nt, npb = vt.n, vp.n
    if nt < 5 or npb < 5:
        return None
    S = vp.D @ vt.D.T
    best = np.argmax(S, axis=1)
    sbest = S[np.arange(npb), best]

    if cfg["ratio"] < 1.0:
        S2 = np.where(DT2[best] < cfg["excl"] ** 2, -2.0, S)
        keep = (sbest > cfg["simmin"]) & (S2.max(axis=1) < cfg["ratio"] * sbest)
    else:
        keep = sbest > cfg["simmin"]
    if keep.sum() < 3:
        return None

    ty = vt.P[best][:, 0] - vp.P[:, 0]
    tx = vt.P[best][:, 1] - vp.P[:, 1]
    b = cfg["bin"]
    ny, nx = (2 * H) // b + 3, (2 * W) // b + 3
    iy = np.clip((ty[keep] + H) // b, 0, ny - 1).astype(np.int32)
    ix = np.clip((tx[keep] + W) // b, 0, nx - 1).astype(np.int32)
    wts = (sbest[keep] - cfg["simmin"]).astype(np.float32)
    acc = np.bincount(iy * nx + ix, weights=wts,
                      minlength=ny * nx).astype(np.float32).reshape(ny, nx)
    pooled = (acc[:-2, :-2] + acc[:-2, 1:-1] + acc[:-2, 2:] +
              acc[1:-1, :-2] + acc[1:-1, 1:-1] + acc[1:-1, 2:] +
              acc[2:, :-2] + acc[2:, 1:-1] + acc[2:, 2:])

    out = {"mass": 0.0, "sqrt": 0.0}
    for K in TOPKS:
        out[f"top{K}"] = 0.0
    kidx_all = np.nonzero(keep)[0]
    for by, bx in _peaks(pooled, cfg["npeak"]):
        mass = float(pooled[by, bx])
        sel = (iy >= by) & (iy <= by + 2) & (ix >= bx) & (ix <= bx + 2)
        if sel.sum() == 0:
            continue
        kidx = kidx_all[sel]
        tyc = int(round(float(ty[kidx].mean())))
        txc = int(round(float(tx[kidx].mean())))
        if mass > out["mass"]:
            out["mass"] = mass
            out["sqrt"] = mass / math.sqrt(nt * npb)
        vals = local_scores(S, vp, vt, TG, tyc, txc, cfg["ltol"])
        if vals.size == 0:
            continue
        v = np.sort(vals)[::-1]
        for K in TOPKS:
            pad = np.zeros(K, np.float32)
            pad[:min(K, v.size)] = v[:K]
            s = float(pad.mean())
            if s > out[f"top{K}"]:
                out[f"top{K}"] = s
    return out


VARIANTS = ["sqrt"] + [f"top{K}" for K in TOPKS]

DEF5 = dict(DEF3, desc="patch", R=8, sstep=2, excl=12, simmin=0.30, step=4,
            margin=8, ltol=6, bin=5, npeak=3)


def build(cfg, imgs=None, idx=None, verbose=True):
    if imgs is None:
        imgs, _, idx = load_all()
    t0 = time.time()
    views = [build_views(im, cfg) for im in imgs]
    r0 = len(cfg["rots"]) // 2
    counts = [v[r0].n for v in views]
    if verbose:
        print(f"  keypoints/image mean {np.mean(counts):.0f} "
              f"[{min(counts)}..{max(counts)}]  dim {views[0][0].D.shape[1]}  "
              f"({time.time()-t0:.0f}s extract)", flush=True)
    DT2 = [((v[r0].P[None, :, :] - v[r0].P[:, None, :]) ** 2).sum(-1).astype(np.float32)
           for v in views]
    TG = [_kp_index(v[r0].P) for v in views]
    n = len(imgs)
    Ms = {v: np.zeros((n, n), np.float32) for v in VARIANTS}
    for i in range(n):
        for j in range(n):
            bestv = {v: 0.0 for v in VARIANTS}
            for r in range(len(cfg["rots"])):
                res = match_pair(views[i][r0], views[j][r], DT2[i], TG[i], cfg)
                if res is None:
                    continue
                for v in VARIANTS:
                    if res[v] > bestv[v]:
                        bestv[v] = res[v]
            for v in VARIANTS:
                Ms[v][i, j] = bestv[v]
    if verbose:
        print(f"  {n*n} pairs ({time.time()-t0:.0f}s)", flush=True)
    return Ms, idx, n, counts


def report(Ms, idx, n, tag, top=99):
    rows = []
    for v, M in Ms.items():
        sc = lambda i, j: float(M[i, j])
        bad = sum(1 for i in range(n)
                  if M[i, i] < max(M[j, i] for j in range(n) if j != i))
        tA, _, _ = scenario_A(sc, idx)
        mB, _, _ = scenario_B(sc, idx)
        rows.append(dict(v=v, bad=bad,
                         dA=float(np.mean([t["dprime"] for t in tA])),
                         sdA=float(np.std([t["dprime"] for t in tA])),
                         eA=float(np.mean([t["eer"] for t in tA])),
                         fA=float(np.mean([t["far10"] for t in tA])),
                         dB=mB["dprime"], eB=mB["eer"], fB=mB["far10"]))
    rows.sort(key=lambda r: -(r["dA"] + r["dB"]))
    for r in rows[:top]:
        print(f"  {tag:14s} {r['v']:8s} selffail {r['bad']:2d}  "
              f"A d'={r['dA']:5.2f}+/-{r['sdA']:.2f} EER={r['eA']*100:5.1f}% "
              f"FAR10={r['fA']*100:5.1f}%  |  B d'={r['dB']:5.2f} "
              f"EER={r['eB']*100:5.1f}% FAR10={r['fB']*100:5.1f}%", flush=True)
    return rows


if __name__ == "__main__":
    imgs, _, idx = load_all()
    cfgs = [
        ("v5 t9 p3", dict(DEF5, ltol=9)),
        ("v5 t12 p3", dict(DEF5, ltol=12)),
        ("v5 t9 p1", dict(DEF5, ltol=9, npeak=1)),
        ("v5 t16 p3", dict(DEF5, ltol=16)),
        ("v5 t9 R12", dict(DEF5, ltol=9, R=12, margin=12, excl=14)),
        ("v5 t9 s3", dict(DEF5, ltol=9, step=3)),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, cfg in cfgs:
        if only and only not in tag:
            continue
        print(f"\n===== {tag} =====", flush=True)
        Ms, _, n, counts = build(cfg, imgs, idx)
        report(Ms, idx, n, tag)
        np.savez(f"V5_{tag.replace(' ', '_')}.npz", **Ms)
