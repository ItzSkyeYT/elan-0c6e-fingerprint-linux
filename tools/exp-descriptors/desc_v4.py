"""
Descriptor matching v4 -- vote for the alignment, then SCORE it with local
elastic correlation.

What v1-v3 measured: counting geometrically consistent descriptor matches
reproduces the correlation baseline and no more.  The count is a poor statistic:
it throws away how WELL each patch matched, and it scales with overlap area, so
a 30%-overlap genuine pair is punished exactly like an impostor.

v4 keeps the descriptor stage only as an alignment finder (it tolerates partial
overlap and elastic distortion, which is what it is good at) and then computes a
different score:

    for every probe keypoint, take the best descriptor similarity among the
    template keypoints lying within `tol` px of its predicted position;
    the score is the mean of the top K of those.

Fixed K makes the score comparable across pairs regardless of overlap area --
a genuine pair sharing 30% of its area still supplies K good local matches,
whereas an impostor does not, no matter how much area it shares.  Patches are
allowed to snap to the best neighbour within `tol`, so the alignment is locally
elastic rather than a single rigid transform.

All of this is a handful of dot products, a small integer accumulator and a
partial sort -- portable to C without an FFT.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desc_v3 import build_views, DEF3, H, W
from proto import load_all, scenario_A, scenario_B, summarise

TOPKS = (8, 16, 24, 32)


def _kp_index(P):
    """Dense (H, W) map from pixel position to keypoint id, -1 where none."""
    g = np.full((H, W), -1, np.int32)
    g[P[:, 0], P[:, 1]] = np.arange(len(P), dtype=np.int32)
    return g


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
    k = int(np.argmax(pooled))
    by, bx = divmod(k, pooled.shape[1])
    mass = float(pooled[by, bx])
    sel = (iy >= by) & (iy <= by + 2) & (ix >= bx) & (ix <= bx + 2)
    if sel.sum() == 0:
        return None
    kidx = np.nonzero(keep)[0][sel]
    tyc = int(round(float(ty[kidx].mean())))
    txc = int(round(float(tx[kidx].mean())))

    # ---- local elastic correlation at the proposed alignment ----------------
    tol = cfg["ltol"]
    offs = [(dy, dx) for dy in range(-tol, tol + 1) for dx in range(-tol, tol + 1)
            if dy * dy + dx * dx <= tol * tol]
    py = vp.P[:, 0] + tyc
    px = vp.P[:, 1] + txc
    loc = np.full(npb, -2.0, np.float32)
    any_hit = np.zeros(npb, bool)
    rows = np.arange(npb)
    for dy, dx in offs:
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
        any_hit |= hit

    out = {"mass": mass, "sqrt": mass / math.sqrt(nt * npb),
           "n_ov": int(any_hit.sum()), "ty": tyc, "tx": txc}
    vals = loc[any_hit]
    for K in TOPKS:
        if vals.size == 0:
            out[f"top{K}"] = 0.0
            out[f"top{K}p"] = 0.0
            continue
        v = np.sort(vals)[::-1]
        pad = np.zeros(K, np.float32)
        pad[:min(K, v.size)] = v[:K]
        out[f"top{K}"] = float(pad.mean())            # zero-padded: needs K matches
        out[f"top{K}p"] = float(v[:K].mean())         # unpadded: pure quality
    return out


VARIANTS = ["sqrt"] + [f"top{K}" for K in TOPKS] + [f"top{K}p" for K in TOPKS]

# margin = R: keypoints get full support.  Allowing border keypoints with
# partially-masked support was measured (run_patch.log, patchR8) to LOSE about
# 0.4 d-prime -- a half-empty patch is a noisy descriptor, not a useful one.
DEF4 = dict(DEF3, desc="patch", R=8, sstep=2, excl=12, simmin=0.30, step=4,
            margin=8, ltol=6, bin=5)


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
            bestmass, bestres = -1.0, None
            for r in range(len(cfg["rots"])):
                res = match_pair(views[i][r0], views[j][r], DT2[i], TG[i], cfg)
                if res is not None and res["mass"] > bestmass:
                    bestmass, bestres = res["mass"], res
            for v in VARIANTS:
                Ms[v][i, j] = bestres[v] if bestres else 0.0
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
        ("v4 R8", dict(DEF4)),
        ("v4 R12", dict(DEF4, R=12, sstep=2, excl=14, margin=12)),
        ("v4 R8 t9", dict(DEF4, ltol=9)),
        ("v4 R8 gab", dict(DEF4, enh="gabor")),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, cfg in cfgs:
        if only and only not in tag:
            continue
        print(f"\n===== {tag} =====", flush=True)
        Ms, _, n, counts = build(cfg, imgs, idx)
        report(Ms, idx, n, tag)
        np.savez(f"V4_{tag.replace(' ', '_')}.npz", **Ms)
