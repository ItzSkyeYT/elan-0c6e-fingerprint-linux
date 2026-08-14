#!/usr/bin/env python3
"""
Three-cue joint search, all maximised at a COMMON alignment:

    cue 0  reliability-weighted NCC on the contrast-normalised image
    cue 1  centred orientation correlation at a FINE block scale
    cue 2  centred orientation correlation at a COARSE block scale

The two orientation scales are not redundant: the fine field follows individual
ridge wiggles (high information, low robustness), the coarse field follows the
overall flow pattern (low information, high robustness).  Weighting them
separately lets the sweep decide how much of each is worth having.

The weight simplex is swept at 0.05 resolution and the WHOLE surface is printed,
so it is visible whether the optimum is a broad plateau (real) or a single spike
(a fit to 26 scores in scenario B).

    python3 run5.py <tag> [key=value ...]
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                       # noqa: F401,F403
from oflib import _corr_planes
from wncc import reliability, wncc_planes_a, wncc_planes_b, wncc_map
from run import load_flat, show, CACHE     # noqa


def simplex(step=0.1):
    n = int(round(1.0 / step))
    out = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            out.append((a / n, b / n, c / n))
    return out


def matrices(imgs, bf=4, bc=12, energy_q=0.40, rel_pow=1.0, lcn_sigma=6.0,
             max_dx=24, max_dy=10, max_rot=12, rot_step=3,
             min_blocks=25, min_w=200.0, weights=None):
    weights = weights if weights is not None else simplex(0.1)
    Wm = np.array(weights)
    n = len(imgs)
    ph, pw, iy, ix = _corr_planes(max_dx, max_dy)
    pres = [local_contrast_norm(im, lcn_sigma) for im in imgs]

    PA, FAf, FAc = [], [], []
    for p in pres:
        g, r, ff = reliability(None, pre=p, deg=0, block=bf,
                               energy_q=energy_q, rel_pow=rel_pow)
        _, _, fc = reliability(None, pre=p, deg=0, block=bc,
                               energy_q=energy_q, rel_pow=rel_pow)
        PA.append(wncc_planes_a(g, r, ph, pw))
        FAf.append(fft_pack(ff, ph, pw))
        FAc.append(fft_pack(fc, ph, pw))

    rots = [0] if max_rot == 0 else list(range(-max_rot, max_rot + 1, rot_step))
    M = np.full((len(weights), n, n), -1.0)
    for j in range(n):
        PB, FBf, FBc = [], [], []
        for d in rots:
            g, r, ff = reliability(None, pre=pres[j], deg=d, block=bf,
                                   energy_q=energy_q, rel_pow=rel_pow)
            _, _, fc = reliability(None, pre=pres[j], deg=d, block=bc,
                                   energy_q=energy_q, rel_pow=rel_pow)
            PB.append(wncc_planes_b(g, r, ph, pw))
            FBf.append(fft_pack(ff, ph, pw, conj=True))
            FBc.append(fft_pack(fc, ph, pw, conj=True))
        for i in range(n):
            best = np.full(len(weights), -1.0)
            for k in range(len(rots)):
                wm, _ = wncc_map(PA[i], PB[k], ph, pw, iy, ix, min_w)
                of = orient_map(FAf[i], FBf[k], ph, pw, iy, ix, min_blocks, bf)
                oc = orient_map(FAc[i], FBc[k], ph, pw, iy, ix, min_blocks, bc)
                good = np.isfinite(wm) & np.isfinite(of) & np.isfinite(oc)
                if not good.any():
                    continue
                C = np.stack([wm[good], of[good], oc[good]])   # 3 x nshift
                v = (Wm @ C).max(axis=1)
                best = np.maximum(best, v)
            M[:, i, j] = best
    return M


def main():
    imgs, labels = load_flat()
    args = sys.argv[1:]
    tag = args[0] if args else "w5_b4_b12"
    kw = {}
    for a in args[1:]:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
    weights = simplex(0.1)
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        M = np.load(path)
    else:
        t0 = time.time()
        M = matrices(imgs, weights=weights, **kw)
        np.save(path, M)
        print(f"  computed in {time.time()-t0:.0f}s", file=sys.stderr)

    res = [show("", M[i], labels, quiet=True) for i in range(len(weights))]
    A = np.array([r["A"][0][0] for r in res])
    B = np.array([r["B"][0] for r in res])
    print(f"\n== {tag}: w0*weightedNCC + w1*orient(fine) + w2*orient(coarse) ==")
    print(f"  {'w_ncc':>6}{'w_fine':>8}{'w_coarse':>9}"
          f"{'A d':>7}{'A EER':>8}{'A FAR10':>9}{'B d':>7}{'B EER':>8}{'B FAR10':>9}")
    order = np.argsort(-(A + B))          # rank by the SUM so neither protocol alone drives it
    for k in order[:20]:
        w = weights[k]
        r = res[k]
        print(f"  {w[0]:6.2f}{w[1]:8.2f}{w[2]:9.2f}"
              f"{r['A'][0][0]:7.2f}{r['A'][0][1]*100:7.1f}%{r['A'][0][2]*100:8.1f}%"
              f"{r['B'][0]:7.2f}{r['B'][1]*100:7.1f}%{r['B'][2]*100:8.1f}%")
    kb = int(np.argmax(A + B))
    print(f"\n  corners:  pure WNCC A={A[weights.index((1.0,0.0,0.0))]:.2f} "
          f"B={B[weights.index((1.0,0.0,0.0))]:.2f} | "
          f"pure fine A={A[weights.index((0.0,1.0,0.0))]:.2f} "
          f"B={B[weights.index((0.0,1.0,0.0))]:.2f} | "
          f"pure coarse A={A[weights.index((0.0,0.0,1.0))]:.2f} "
          f"B={B[weights.index((0.0,0.0,1.0))]:.2f}")
    show(f"{tag} best {weights[kb]}", M[kb], labels)


if __name__ == "__main__":
    main()
