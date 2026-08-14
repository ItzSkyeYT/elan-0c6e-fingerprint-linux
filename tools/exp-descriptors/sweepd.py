"""
Sweep the descriptor design.  The (np x nt) cosine-similarity matrix is the
dominant cost, so it is computed once per (pair, rotation) and every matching
variant is evaluated on it.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desc_v1 import build_views, H, W, DEFAULT
from proto import load_all, scenario_A, scenario_B, summarise, selfcheck, _metrics


def vote(S, Pt, Pp, DT2, mv, acc):
    """mv = (ratio, excl, simmin, bin, norm).  Returns score."""
    ratio, excl, simmin, b, norm = mv
    npb, nt = S.shape
    if npb < 4 or nt < 4:
        return 0.0
    best = np.argmax(S, axis=1)
    sbest = S[np.arange(npb), best]
    if ratio < 1.0:
        S2 = np.where(DT2[best] < excl * excl, -2.0, S)
        s2 = S2.max(axis=1)
        keep = (sbest > simmin) & (s2 < ratio * sbest)
    else:
        keep = sbest > simmin
    k = int(keep.sum())
    if k < 3:
        return 0.0
    bt = Pt[best][keep]
    ty = bt[:, 0] - Pp[keep, 0]
    tx = bt[:, 1] - Pp[keep, 1]
    wts = (sbest[keep] - simmin).astype(np.float32)

    acc[:] = 0
    iy = np.clip((ty + H) // b, 0, acc.shape[0] - 1).astype(np.int32)
    ix = np.clip((tx + W) // b, 0, acc.shape[1] - 1).astype(np.int32)
    np.add.at(acc, (iy, ix), wts)
    pooled = (acc[:-2, :-2] + acc[:-2, 1:-1] + acc[:-2, 2:] +
              acc[1:-1, :-2] + acc[1:-1, 1:-1] + acc[1:-1, 2:] +
              acc[2:, :-2] + acc[2:, 1:-1] + acc[2:, 2:])
    kk = int(np.argmax(pooled))
    by, bx = divmod(kk, pooled.shape[1])
    mass = float(pooled[by, bx])
    sel = (iy >= by) & (iy <= by + 2) & (ix >= bx) & (ix <= bx + 2)
    cnt = float(sel.sum())

    if norm == "sqrt":
        return mass / math.sqrt(nt * npb)
    if norm == "probe":
        return mass / npb
    if norm == "cnt":
        return cnt / math.sqrt(nt * npb)
    if norm == "raw":
        return mass
    raise ValueError(norm)


def evaluate(cfg, match_variants, imgs=None, idx=None, verbose=True):
    if imgs is None:
        imgs, _, idx = load_all()
    t0 = time.time()
    rots = cfg["rots"]
    views = [build_views(im, cfg, rots) for im in imgs]
    r0 = len(rots) // 2
    counts = [v[r0].n for v in views]
    dims = views[0][0].D.shape[1]
    if verbose:
        print(f"  keypoints/image mean {np.mean(counts):.0f} "
              f"[{min(counts)}..{max(counts)}] dim {dims} ({time.time()-t0:.0f}s)")

    # squared distances between template keypoints, for the ratio-test exclusion
    DT2 = [((v[r0].P[None, :, :] - v[r0].P[:, None, :]) ** 2).sum(-1).astype(np.float32)
           for v in views]

    n = len(imgs)
    maxb = max(mv[3] for mv in match_variants)
    acc = np.zeros(((2 * H) // min(mv[3] for mv in match_variants) + 3,
                    (2 * W) // min(mv[3] for mv in match_variants) + 3), np.float32)
    Ms = {mv: np.zeros((n, n), np.float32) for mv in match_variants}

    for i in range(n):
        vt = views[i][r0]
        for j in range(n):
            best = {mv: 0.0 for mv in match_variants}
            for r in range(len(rots)):
                vp = views[j][r]
                if vp.n < 4 or vt.n < 4:
                    continue
                S = vp.D @ vt.D.T
                for mv in match_variants:
                    bsz = mv[3]
                    a = acc[:(2 * H) // bsz + 3, :(2 * W) // bsz + 3]
                    s = vote(S, vt.P, vp.P, DT2[i], mv, a)
                    if s > best[mv]:
                        best[mv] = s
            for mv in match_variants:
                Ms[mv][i, j] = best[mv]
    if verbose:
        print(f"  {n*n} pairs x {len(rots)} rots x {len(match_variants)} variants "
              f"({time.time()-t0:.0f}s)")
    return Ms, idx, n, counts, dims


def score_all(Ms, idx, n, tag, top=4):
    rows = []
    for mv, M in Ms.items():
        sc = lambda i, j: float(M[i, j])
        bad = sum(1 for i in range(n)
                  if M[i, i] < max(M[j, i] for j in range(n) if j != i))
        tA, gA, iA = scenario_A(sc, idx)
        mB, _, _ = scenario_B(sc, idx)
        dA = float(np.mean([t["dprime"] for t in tA]))
        rows.append(dict(mv=mv, bad=bad,
                         dA=dA, sdA=float(np.std([t["dprime"] for t in tA])),
                         eA=float(np.mean([t["eer"] for t in tA])),
                         fA=float(np.mean([t["far10"] for t in tA])),
                         dB=mB["dprime"], eB=mB["eer"], fB=mB["far10"]))
    rows.sort(key=lambda r: -(r["dA"] + r["dB"]))
    for r in rows[:top]:
        print(f"  {tag:22s} {str(r['mv']):34s} self-fail {r['bad']:2d}  "
              f"A d'={r['dA']:5.2f}+/-{r['sdA']:.2f} EER={r['eA']*100:5.1f}% "
              f"FAR10={r['fA']*100:5.1f}%  |  B d'={r['dB']:5.2f} "
              f"EER={r['eB']*100:5.1f}% FAR10={r['fB']*100:5.1f}%")
    return rows


MATCH_VARIANTS = [
    (0.92, 12, 0.30, 5, "sqrt"),
    (0.92, 12, 0.30, 5, "cnt"),
    (0.85, 12, 0.35, 5, "sqrt"),
    (0.98, 12, 0.40, 5, "sqrt"),
    (1.00, 0, 0.45, 5, "sqrt"),
    (1.00, 0, 0.55, 5, "sqrt"),
    (0.92, 20, 0.30, 8, "sqrt"),
    (0.92, 12, 0.30, 3, "sqrt"),
]

if __name__ == "__main__":
    imgs, _, idx = load_all()
    grids = [
        ("lcn/patch/grid r8 s4", dict(DEFAULT, enh="lcn", desc="patch", kp="grid", r=8, step=4)),
        ("lcn/orient/grid r8 s4", dict(DEFAULT, enh="lcn", desc="orient", kp="grid", r=8, step=4)),
        ("lcn/both/grid r8 s4", dict(DEFAULT, enh="lcn", desc="both", kp="grid", r=8, step=4)),
        ("gabor/patch/grid r8s4", dict(DEFAULT, enh="gabor", desc="patch", kp="grid", r=8, step=4)),
        ("lcn/patch/grid r12s4", dict(DEFAULT, enh="lcn", desc="patch", kp="grid", r=12, sub=3, step=4)),
        ("lcn/patch/harris", dict(DEFAULT, enh="lcn", desc="patch", kp="harris", r=8, nmax=160, nms=5)),
        ("lcn/both/harris", dict(DEFAULT, enh="lcn", desc="both", kp="harris", r=8, nmax=160, nms=5)),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, cfg in grids:
        if only and only not in tag:
            continue
        print(f"\n===== {tag} =====", flush=True)
        Ms, idx2, n, counts, dims = evaluate(cfg, MATCH_VARIANTS, imgs, idx)
        score_all(Ms, idx, n, tag)
