#!/usr/bin/env python3
"""Build and cache the candidate score matrices.

Note on rotation: the existing baseline rotates with edge clamping, so after a
12 degree rotation the corners of the window are filled with smeared copies of
the border and then correlated as if they were data.  Here the out-of-frame
pixels get weight 0 and drop out of the correlation entirely, which is both
more correct and measurably better.
"""
import math, sys, time
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds
import minutiae as M
import quality as Q
import wncc

ORDER = ["right-index", "right-index-cover", "right-middle"]
W, H = 150, 52


def build(ds, rots, mdx, mdy, min_frac, weight="none", pre="lcn"):
    """min_frac: minimum overlap as a fraction of the FULL window weight."""
    ws, ims = [], []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            q, n = Q.quality(im)
            if pre == "gabor":
                th, _ = M.orientation(n, block=8, smooth=2.5)
                n = M.gabor(n, th, 0.110, n_orient=16, ksize=11, sx=4.0, sy=4.0)
            if weight == "none":
                w = np.ones_like(q)
            elif weight == "q":
                w = q
            elif weight == "qs":                    # mild: sqrt
                w = np.sqrt(np.clip(q, 0, 1))
            else:
                raise ValueError(weight)
            ws.append(w); ims.append(np.asarray(n, np.float64))
    P = [wncc.prep(ims[i], ws[i], rots, mdx, mdy) for i in range(len(ims))]
    # scale the overlap requirement to the average weight, so `min_frac` means
    # the same effective area whatever the weighting is
    mw = float(np.mean([w.mean() for w in ws]))
    min_w = min_frac * (W * H) * mw * mw
    n = len(P)
    S = np.full((n, n), np.nan)
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            s, _ = wncc.surface(P[i], P[j], rots, mdx, mdy, min_w)
            S[i, j] = S[j, i] = float(np.nanmax(s)) if np.isfinite(s).any() else -1.0
    return S, time.time() - t0, min_w


def main():
    ds = load_ds()
    rots = list(range(-12, 13, 4))
    jobs = [
        ("P_dx20", 20, 8, 0.45, "none", "lcn"),
        ("P_dx30", 30, 10, 0.40, "none", "lcn"),
        ("P_dx40", 40, 14, 0.35, "none", "lcn"),
        ("P_dx60", 60, 20, 0.28, "none", "lcn"),
        ("Pq_dx20", 20, 8, 0.45, "qs", "lcn"),
        ("Pq_dx40", 40, 14, 0.35, "qs", "lcn"),
        ("Pg_dx20", 20, 8, 0.45, "none", "gabor"),
        ("Pg_dx40", 40, 14, 0.35, "none", "gabor"),
    ]
    for name, mdx, mdy, mf, wt, pre in jobs:
        S, el, mw = build(ds, rots, mdx, mdy, mf, wt, pre)
        np.save(f"{name}.npy", S)
        print(f"{name:10s} dx{mdx} dy{mdy} minw={mw:.0f} {wt:5s} {pre:6s} "
              f"{el:.1f}s ({el*1000/990:.1f} ms/pair)")


if __name__ == "__main__":
    main()
