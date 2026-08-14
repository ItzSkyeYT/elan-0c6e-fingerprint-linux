#!/usr/bin/env python3
"""
LOCAL scoring on top of the joint global alignment.

Motivation.  Even after the reliability weighting, one number per alignment
still averages over the whole common area.  Two presses of the same finger that
overlap on, say, the left third produce a strong agreement there and noise
everywhere else, and the average buries the evidence.  A minutiae matcher wins
precisely because a handful of agreeing landmarks in a small region is enough.

So: find the alignment with the joint global search (weighted NCC + orientation,
common shift and rotation), then at that alignment compute a LOCAL weighted
correlation in a sliding window and summarise it by the best-matching sub-region
rather than by the mean.  All of it is box filters over the product images:

    C(x) = [ B(w a b) - B(w a) B(w b)/B(w) ]
           / sqrt( (B(w aa) - B(w a)^2/B(w)) (B(w bb) - B(w b)^2/B(w)) )

with B = box mean over a (2R+1)^2 window and w = ra * rb_shifted.  Six integral
images; trivially portable.

Score variants stored per pair (one matrix each, so the choice is made after the
fact and every variant is visible):

    0  global   the joint global score itself
    1  locmax   the single best local window
    2  loctop10 mean of the best 10% of windows, weight-gated
    3  loctop25 mean of the best 25% of windows
    4  loctop50 mean of the best 50% of windows
    5  mix      0.5*global + 0.5*loctop25

    python3 run6.py <tag> [key=value ...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                       # noqa: F401,F403
from oflib import _corr_planes
from wncc import reliability, wncc_planes_a, wncc_planes_b, wncc_map
from run import load_flat, show, CACHE     # noqa

VARIANTS = ["global", "locmax", "loctop10", "loctop25", "loctop50", "mix"]


def shift_into(x, dy, dx):
    """Place b shifted by (dy,dx) into an array of a's shape, zero outside.
    Matches the convention of the correlation maps: overlap pixel (y,x) of a
    corresponds to (y-dy, x-dx) of b."""
    h, w = x.shape
    out = np.zeros_like(x)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 <= ys0 or xs1 <= xs0:
        return out
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = x[ys0:ys1, xs0:xs1]
    return out


def local_scores(ga, ra, gb, rb, dy, dx, R, wfloor):
    """Local weighted correlation at a FIXED alignment; returns the sorted
    (descending) vector of window correlations that carry enough weight."""
    gbs = shift_into(gb, dy, dx)
    rbs = shift_into(rb, dy, dx)
    w = ra * rbs
    Bw = boxfilt(w, R)
    Bwa = boxfilt(w * ga, R)
    Bwb = boxfilt(w * gbs, R)
    Bwab = boxfilt(w * ga * gbs, R)
    Bwaa = boxfilt(w * ga * ga, R)
    Bwbb = boxfilt(w * gbs * gbs, R)
    Bp = np.maximum(Bw, 1e-9)
    num = Bwab - Bwa * Bwb / Bp
    va = Bwaa - Bwa * Bwa / Bp
    vb = Bwbb - Bwb * Bwb / Bp
    den = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
    # Bw is a box MEAN of the per-pixel weight, so the floor is a fraction of
    # the full-weight case, not a pixel count.
    ok = (Bw > wfloor) & (den > 1e-9)
    if not ok.any():
        return None
    c = (num / np.maximum(den, 1e-12))[ok]
    # subsample on the window stride so overlapping windows do not all count
    return np.sort(c)[::-1]


def topfrac(v, frac):
    k = max(1, int(round(frac * v.size)))
    return float(v[:k].mean())


def matrices(imgs, block=8, energy_q=0.40, rel_pow=1.0, lcn_sigma=6.0,
             max_dx=24, max_dy=10, max_rot=12, rot_step=3,
             min_blocks=25, min_w=200.0, wfix=0.25, R=10, wfloor=0.05):
    n = len(imgs)
    ph, pw, iy, ix = _corr_planes(max_dx, max_dy)
    pres = [local_contrast_norm(im, lcn_sigma) for im in imgs]
    kw = dict(block=block, energy_q=energy_q, rel_pow=rel_pow)

    GA, RA, PA, FA = [], [], [], []
    for p in pres:
        g, r, f = reliability(None, pre=p, deg=0, **kw)
        GA.append(g); RA.append(r)
        PA.append(wncc_planes_a(g, r, ph, pw))
        FA.append(fft_pack(f, ph, pw))

    rots = [0] if max_rot == 0 else list(range(-max_rot, max_rot + 1, rot_step))
    M = np.full((len(VARIANTS), n, n), -1.0)
    for j in range(n):
        GB, RB, PB, FB = [], [], [], []
        for d in rots:
            g, r, f = reliability(None, pre=pres[j], deg=d, **kw)
            GB.append(g); RB.append(r)
            PB.append(wncc_planes_b(g, r, ph, pw))
            FB.append(fft_pack(f, ph, pw, conj=True))
        for i in range(n):
            bestv, bestk, bestdy, bestdx = -2.0, 0, 0, 0
            for k in range(len(rots)):
                wm, _ = wncc_map(PA[i], PB[k], ph, pw, iy, ix, min_w)
                om = orient_map(FA[i], FB[k], ph, pw, iy, ix, min_blocks, block)
                s = np.where(np.isfinite(wm) & np.isfinite(om),
                             (1 - wfix) * wm + wfix * om, -np.inf)
                if not np.isfinite(s).any():
                    continue
                t = int(np.argmax(s))
                if s.flat[t] > bestv:
                    bestv = float(s.flat[t])
                    bestk = k
                    bestdy = t // s.shape[1] - max_dy
                    bestdx = t % s.shape[1] - max_dx
            if bestv < -1.5:
                continue
            v = local_scores(GA[i], RA[i], GB[bestk], RB[bestk],
                             bestdy, bestdx, R, wfloor)
            if v is None:
                M[:, i, j] = bestv
                continue
            M[0, i, j] = bestv
            M[1, i, j] = float(v[0])
            M[2, i, j] = topfrac(v, 0.10)
            M[3, i, j] = topfrac(v, 0.25)
            M[4, i, j] = topfrac(v, 0.50)
            M[5, i, j] = 0.5 * bestv + 0.5 * topfrac(v, 0.25)
    return M


def main():
    imgs, labels = load_flat()
    args = sys.argv[1:]
    tag = args[0] if args else "w6_b8"
    kw = {}
    for a in args[1:]:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        M = np.load(path)
    else:
        t0 = time.time()
        M = matrices(imgs, **kw)
        np.save(path, M)
        print(f"  computed in {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"\n== {tag}: local scoring at the joint global alignment ==")
    for vi, nm in enumerate(VARIANTS):
        r = show(f"{tag}/{nm}", M[vi], labels, quiet=True)
        print(f"  {nm:<10}A d'={r['A'][0][0]:5.2f}  EER={r['A'][0][1]*100:5.1f}%  "
              f"FAR10={r['A'][0][2]*100:5.1f}%   |   B d'={r['B'][0]:5.2f}  "
              f"EER={r['B'][1]*100:5.1f}%  FAR10={r['B'][2]*100:5.1f}%   "
              f"(diag {np.diag(M[vi]).mean():.3f})")


if __name__ == "__main__":
    main()
