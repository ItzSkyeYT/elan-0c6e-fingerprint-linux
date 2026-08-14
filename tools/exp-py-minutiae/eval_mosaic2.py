#!/usr/bin/env python3
"""Mosaic, second pass.

The first pass registered all 19 enrolment captures unconditionally and the
last few went in at correlation 0.12 -- i.e. essentially at random -- which
smears the canvas and destroys the ridge detail the matcher needs.  Here a
capture is only merged if it registers above `accept`; the rest are kept as
ordinary standalone templates and scored the usual max-over-templates way.
The final score is the max of (mosaic score, best standalone score), so the
mosaic can only help.
"""
import math, sys, time
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr
import quality as Q
import mosaic as MO
import wncc

ORDER = ["right-index", "right-index-cover", "right-middle"]
ROTS = list(range(-12, 13, 4))
W, H = 150, 52


def prep_all(ds):
    ims, wts, labels = [], [], []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            q, n = Q.quality(im)
            ims.append(np.asarray(n, np.float64))
            wts.append(np.asarray(q, np.float64))
            labels.append(lbl)
    return ims, wts, labels


def pair_matrix(ims, mdx=20, mdy=8, min_frac=0.45):
    ones = [np.ones((H, W)) for _ in ims]
    P = [wncc.prep(ims[i], ones[i], ROTS, mdx, mdy) for i in range(len(ims))]
    min_w = min_frac * W * H
    n = len(P)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            s, _ = wncc.surface(P[i], P[j], ROTS, mdx, mdy, min_w)
            S[i, j] = S[j, i] = float(np.nanmax(s)) if np.isfinite(s).any() else -1.0
    return S


def rep(tag, g, i):
    d = dprime(g, i); e, _ = eer(g, i); f, _ = far_at_frr(g, i, 0.10)
    print(f"  {tag:30s} d'={d:6.2f}  EER={e*100:5.1f}%  FAR@10%FRR={f*100:5.1f}%")
    return d, e, f


def run(accept, ims, wts, labels, Spair, trials=12, n_enroll=6, seed=0,
        pad_x=70, pad_y=24):
    gi = [k for k, l in enumerate(labels) if l != "right-middle"]
    ii = [k for k, l in enumerate(labels) if l == "right-middle"]
    cov = [k for k, l in enumerate(labels) if l == "right-index-cover"]
    idx = [k for k, l in enumerate(labels) if l == "right-index"]

    def score_set(T, probes):
        mo, order = MO.build([ims[k] for k in T], [wts[k] for k in T],
                             pad_x=pad_x, pad_y=pad_y, accept=accept)
        mos = [MO.score_probe(mo, ims[p], wts[p]) for p in probes]
        std = [max(Spair[t, p] for t in T) for p in probes]
        return mo.placed, np.array(mos), np.array(std)

    # B
    nb, gm, gs = score_set(cov, idx)
    _, mm, ms = score_set(cov, ii)
    print(f"\n--- accept={accept}: scenario B (mosaic merged {nb}/19) ---")
    rep("B mosaic only", gm, mm)
    rep("B standalone max (reference)", gs, ms)
    rB = rep("B max(mosaic, standalone)", np.maximum(gm, gs), np.maximum(mm, ms))

    # A
    rng = np.random.default_rng(seed)
    GM, MM, GS, MS = [], [], [], []
    for _ in range(trials):
        T = list(rng.choice(gi, size=n_enroll, replace=False))
        probes = [k for k in gi if k not in T]
        _, gm2, gs2 = score_set(T, probes)
        _, mm2, ms2 = score_set(T, ii)
        GM += list(gm2); GS += list(gs2); MM += list(mm2); MS += list(ms2)
    GM, GS, MM, MS = map(np.array, (GM, GS, MM, MS))
    print(f"--- accept={accept}: scenario A ---")
    rep("A mosaic only", GM, MM)
    rep("A standalone max (reference)", GS, MS)
    rA = rep("A max(mosaic, standalone)", np.maximum(GM, GS), np.maximum(MM, MS))
    return rA, rB


def main():
    ds = load_ds()
    ims, wts, labels = prep_all(ds)
    t0 = time.time()
    Spair = pair_matrix(ims)
    print(f"pairwise reference matrix {time.time()-t0:.1f}s")
    for accept in (0.45, 0.35):
        run(accept, ims, wts, labels, Spair)


if __name__ == "__main__":
    main()
