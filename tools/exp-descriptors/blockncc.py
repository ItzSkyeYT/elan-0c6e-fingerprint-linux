#!/usr/bin/env python3
"""v5: elastic local-patch matching with a TRIMMED aggregate.

v4 (blockdesc) reduces every patch to a hard in/out decision and counts.  At a
17x17 patch size -- two ridge periods -- each decision is close to a coin flip,
so the count is dominated by its own variance.  v5 keeps the graded evidence:

  score = max over global alignments (tx, ty, theta) of
            AGG_k [ max over a small local displacement of
                      NCC( patch k of A , B at that place ) ]

with AGG a TRIMMED aggregate -- the mean of the best K patches, not of all of
them.  That is the partial-overlap fix stated as a statistic rather than as a
threshold: a probe sharing 40% of its area with the template contributes ~40%
of its patches at high correlation and the rest at noise level, and the trimmed
mean reads the former while whole-image NCC averages the two together.

The inner "max over a small local displacement" is elastic matching: skin
deforms between presses, so patches sit a few pixels off the rigid alignment.

AGG over all patches (trim=1.0, elastic=0) degenerates to ordinary
overlap-restricted NCC, which is the baseline -- so the baseline is nested
inside this model and is reported alongside as a control.

The heavy step is the same patch-vs-all-positions correlation as v4, so this
shares blockdesc's templates.
"""

import math
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blockdesc as BD

W, H = 150, 52

DEFAULT = dict(
    enh="gabor", osmooth=6.0,
    r=8, sub=2,
    q_step=4, q_margin=None,
    min_coh=0.0, max_amb=1.01, keep_best=0, amb_excl=14.0,
    rots=(-12.0, -6.0, 0.0, 6.0, 12.0),
    elastic=2,                      # local displacement tolerance, px
    max_tx=55, max_ty=22,           # global translation search range
    trims=(4, 8, 12, 20, 30, 45, 0),   # K for the trimmed mean (0 = all valid)
    fracs=(0.15, 0.25, 0.35, 0.5),  # K as a FRACTION of the overlapping patches
    sels=(0, 30, 12),               # K used to CHOOSE the alignment
    min_valid=40,                   # minimum usable patches for an alignment
    psr_excls=(10, 18, 28),         # sidelobe exclusion radii, px
    variants=None,                  # filled in below
)


def _vnames(cfg):
    t = list(cfg["trims"]) + [f"f{int(f*100)}" for f in cfg["fracs"]]
    n = [f"max{k}" for k in t]
    n += [f"sel{s}sc{k}" for s in cfg["sels"] for k in t]
    n += [f"psr{k}e{e}" for k in t for e in cfg["psr_excls"]]
    return tuple(n)


DEFAULT["variants"] = _vnames(DEFAULT)

make_template = BD.make_template


def _dilate(S, d):
    """Max filter over a (2d+1)^2 window, edge-replicated, on the last 2 axes."""
    if d <= 0:
        return S
    p = np.pad(S, ((0, 0), (0, 0), (d, d), (d, d)), mode="edge")
    p = sliding_window_view(p, 2 * d + 1, axis=2).max(axis=-1)
    p = sliding_window_view(p, 2 * d + 1, axis=3).max(axis=-1)
    return p


def score_one(ta, tb, cfg):
    rots = np.asarray(cfg["rots"], np.float32)
    nrot, nq = len(rots), ta.nq
    variants = list(cfg["variants"])
    if nq < 4:
        return {v: 0.0 for v in variants}

    S = (ta.qd.reshape(nrot * nq, -1) @ tb.db.T).reshape(nrot, nq, tb.ny, tb.nx)
    S = _dilate(S, cfg["elastic"])
    ny, nx = S.shape[2], S.shape[3]

    tyv = np.arange(-cfg["max_ty"], cfg["max_ty"] + 1)
    txv = np.arange(-cfg["max_tx"], cfg["max_tx"] + 1)
    kidx = np.arange(nq)[:, None, None]
    trims = list(cfg["trims"]) + [f"f{int(f*100)}" for f in cfg["fracs"]]
    nty, ntx = len(tyv), len(txv)
    AGG = {k: np.full((nrot, nty, ntx), -1.0, np.float32) for k in trims}

    for ri in range(nrot):
        th = math.radians(float(rots[ri]))
        ct, st = math.cos(th), math.sin(th)
        ax, ay = ta.qc[:, 0], ta.qc[:, 1]
        bx = np.rint(ct * ax + st * ay).astype(int) - tb.r
        by = np.rint(-st * ax + ct * ay).astype(int) - tb.r

        iy = by[:, None, None] + tyv[None, :, None]
        ix = bx[:, None, None] + txv[None, None, :]
        ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        V = S[ri][kidx, np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
        V = np.where(ok, V, -1.0).astype(np.float32)

        nval = ok.sum(axis=0)

        # descending sort once: the cumulative mean then gives the trimmed mean
        # of the best K patches for EVERY K at no extra cost
        Vs = -np.sort(-V, axis=0)
        C = np.cumsum(Vs, axis=0)
        denom = np.arange(1, nq + 1, dtype=np.float32)[:, None, None]
        M = C / denom                     # M[K-1] = mean of the best K

        void = nval < cfg["min_valid"]
        for k in trims:
            if k == 0:                    # mean over every overlapping patch
                agg = np.where(ok, V, 0.0).sum(axis=0) / np.maximum(nval, 1)
            elif isinstance(k, str):      # K as a fraction of the overlap
                f = int(k[1:]) / 100.0
                ke = np.clip(np.rint(f * nval).astype(int), 1, nq) - 1
                agg = np.take_along_axis(M, ke[None], axis=0)[0]
            else:
                agg = M[min(k, nq) - 1]
            AGG[k][ri] = np.where(void, -1.0, agg)

    out = {}
    for k in trims:
        A = AGG[k]
        flat = int(A.argmax())
        out[f"max{k}"] = float(A.flat[flat])
        # peak-to-sidelobe: a genuine pair has ONE sharp alignment; an impostor's
        # best alignment is one of many mediocre ones, so the drop-off between
        # the peak and the best well-separated rival is itself a discriminant.
        _, py, px = np.unravel_index(flat, A.shape)
        for ex in cfg["psr_excls"]:
            B = A.copy()
            B[:, max(0, py - ex):py + ex + 1, max(0, px - ex):px + ex + 1] = -1.0
            out[f"psr{k}e{ex}"] = float(A.flat[flat]) - float(B.max())

    for s in cfg["sels"]:
        # choose the alignment with one statistic, read the score off another:
        # picking the alignment that maximises the SAME trimmed mean gives an
        # impostor thousands of chances to get lucky.
        pos = int(AGG[s].argmax())
        for k in trims:
            out[f"sel{s}sc{k}"] = float(AGG[k].flat[pos])
    return out
