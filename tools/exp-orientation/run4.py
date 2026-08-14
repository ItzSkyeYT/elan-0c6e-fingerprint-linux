#!/usr/bin/env python3
"""
Joint search over reliability-WEIGHTED NCC and the orientation-field score.

    M_w[t,p] = max over (dx,dy,rot) of  (1-w)*WNCC(dx,dy,rot) + w*ORIENT(dx,dy,rot)

Both cues share the same per-pixel reliability weights and are maximised at a
COMMON alignment, so a high fused score means the two cues agree about WHERE the
match is -- an impostor that scrapes a good intensity correlation at one shift
and a good flow correlation at another gets no credit.

    python3 run4.py <tag> [key=value ...]
    python3 run4.py --eval            # just re-evaluate everything cached
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                       # noqa: F401,F403
from oflib import _corr_planes
from wncc import reliability, wncc_planes_a, wncc_planes_b, wncc_map, shrink
from run import load_flat, show, CACHE     # noqa

WEIGHTS = np.round(np.arange(0.0, 1.001, 0.1), 3)


def matrices(imgs, block=8, energy_q=0.40, rel_pow=1.0, lcn_sigma=6.0,
             max_dx=24, max_dy=10, max_rot=12, rot_step=3,
             min_blocks=25, min_w=200.0, kappa=0.0, weights=WEIGHTS):
    n = len(imgs)
    ph, pw, iy, ix = _corr_planes(max_dx, max_dy)
    pres = [local_contrast_norm(im, lcn_sigma) for im in imgs]
    kw = dict(block=block, energy_q=energy_q, rel_pow=rel_pow)

    PA, FA = [], []
    for p in pres:
        g, r, f = reliability(None, pre=p, deg=0, **kw)
        PA.append(wncc_planes_a(g, r, ph, pw))
        FA.append(fft_pack(f, ph, pw))

    rots = [0] if max_rot == 0 else list(range(-max_rot, max_rot + 1, rot_step))
    M = np.full((len(weights), n, n), -1.0)
    for j in range(n):
        PB, FB = [], []
        for d in rots:
            g, r, f = reliability(None, pre=pres[j], deg=d, **kw)
            PB.append(wncc_planes_b(g, r, ph, pw))
            FB.append(fft_pack(f, ph, pw, conj=True))
        for i in range(n):
            best = np.full(len(weights), -1.0)
            for k in range(len(rots)):
                wm, Ww = wncc_map(PA[i], PB[k], ph, pw, iy, ix, min_w)
                om = orient_map(FA[i], FB[k], ph, pw, iy, ix, min_blocks, block)
                good = np.isfinite(wm) & np.isfinite(om)
                if not good.any():
                    continue
                if kappa > 0:
                    f_ = (Ww / (Ww + kappa))[good]
                    a = wm[good] * f_
                    b = om[good] * f_
                else:
                    a = wm[good]
                    b = om[good]
                for wi, w in enumerate(weights):
                    v = float(np.max((1 - w) * a + w * b))
                    if v > best[wi]:
                        best[wi] = v
            M[:, i, j] = best
    return M


def selfcheck(imgs, **kw):
    """An image against itself must score ~+1 for both cues at zero shift."""
    ph, pw, iy, ix = _corr_planes(kw.get("max_dx", 24), kw.get("max_dy", 10))
    bad = []
    for im in imgs[:6]:
        p = local_contrast_norm(im, 6.0)
        g, r, f = reliability(None, pre=p, deg=0, block=kw.get("block", 8))
        wm, Ww = wncc_map(wncc_planes_a(g, r, ph, pw),
                          wncc_planes_b(g, r, ph, pw), ph, pw, iy, ix, 200.0)
        om = orient_map(fft_pack(f, ph, pw), fft_pack(f, ph, pw, conj=True),
                        ph, pw, iy, ix, 25, kw.get("block", 8))
        bad.append((float(np.max(wm)), float(np.max(om))))
    w0 = min(b[0] for b in bad)
    o0 = min(b[1] for b in bad)
    print(f"  self-check: min WNCC={w0:.4f}  min ORIENT={o0:.4f}  "
          f"{'OK' if w0 > 0.999 and o0 > 0.999 else 'FAIL'}")
    return w0 > 0.999 and o0 > 0.999


def report(tag, M, labels):
    res = [show(f"{tag} w={w:.1f}", M[wi], labels, quiet=True)
           for wi, w in enumerate(WEIGHTS)]
    print(f"\n== {tag}: (1-w)*weightedNCC + w*orientation, common alignment ==")
    print(f"  {'w':>5}{'A d-prime':>12}{'A EER':>9}{'A FAR@10':>10}"
          f"{'B d-prime':>12}{'B EER':>8}{'B FAR@10':>10}")
    for r, w in zip(res, WEIGHTS):
        print(f"  {w:5.1f}{r['A'][0][0]:12.2f}{r['A'][0][1]*100:8.1f}%"
              f"{r['A'][0][2]*100:9.1f}%{r['B'][0]:12.2f}{r['B'][1]*100:7.1f}%"
              f"{r['B'][2]*100:9.1f}%")
    return res


def main():
    imgs, labels = load_flat()
    args = sys.argv[1:]
    if args and args[0] == "--eval":
        for f in sorted(os.listdir(CACHE)):
            if f.startswith("w4_") and f.endswith(".npy"):
                report(f[:-4], np.load(os.path.join(CACHE, f)), labels)
        return
    tag = args[0] if args else "w4_b8"
    kw = {}
    for a in args[1:]:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        M = np.load(path)
    else:
        selfcheck(imgs, **kw)
        t0 = time.time()
        M = matrices(imgs, **kw)
        np.save(path, M)
        print(f"  computed in {time.time()-t0:.0f}s", file=sys.stderr)
    res = report(tag, M, labels)
    bi = int(np.argmax([r["B"][0] for r in res]))
    show(f"{tag} best w={WEIGHTS[bi]:.1f}", M[bi], labels)


if __name__ == "__main__":
    main()
