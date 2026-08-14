#!/usr/bin/env python3
"""Local-descriptor matching, v4: rotated square patches + exhaustive NN +
Hough consistency, with overlap-normalised scoring.

Why this differs from v3 (which measured d'=0.77 / 1.42, i.e. LOST to the
plain-NCC baseline):

  * v3 rotation-normalised each descriptor to the LOCAL ridge direction.  On a
    fingerprint that is close to fatal: away from a minutia the local ridge
    pattern is a set of parallel stripes, so after rotating every patch onto its
    own ridge direction almost all patches look alike and the descriptor stops
    discriminating.  Here the patch is rotated by a small set of GLOBAL angles
    instead (the press-to-press rotation on this sensor is small), so the
    orientation of the ridges relative to the frame is retained as signal, and
    the rotation still becomes a Hough dimension rather than an assumption.

  * the descriptor is the enhanced ridge image sampled on a square grid, and the
    nearest-neighbour search over the other capture is EXHAUSTIVE over every
    pixel position (not a sparse keypoint set), so keypoint repeatability --
    which killed v2/v3 -- cannot matter.  A query patch always has a candidate
    within half a sample step of its true correspondent.

  * the score is normalised by the OVERLAP implied by the winning alignment,
    not by the number of query patches.  That is the whole point of a
    partial-overlap matcher: a probe that shares 40% of its area with the
    template should not be penalised for the 60% it cannot possibly match.

Similarity is a zero-mean unit-norm dot product, i.e. exactly normalised cross
correlation of the patch, so it is the same quantity the NCC baseline uses --
the difference is purely that each patch aligns independently and then has to
agree with the others.

Portability: per comparison this is nrot x nq patch correlations over all
positions.  In C that is either a direct loop (~0.2-0.5 GFLOP at the default
settings) or the existing integral-image NCC; no library needed.
"""

import math
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enh

W, H = 150, 52

DEFAULT = dict(
    enh="gabor", osmooth=6.0,
    r=8, sub=2,                     # patch half-size, sample step inside it
    q_step=6, q_margin=None,        # query grid stride; margin defaults to ceil(r*sqrt2)
    min_coh=0.30,                   # drop query patches over incoherent ridge flow
    max_amb=1.01,                   # keep only patches that are DISTINCTIVE within
    keep_best=0,                    #   their own image (see make_template); or best N
    amb_excl=14.0,
    rots=(-12.0, -6.0, 0.0, 6.0, 12.0),
    topk=1, nms=6,                  # candidate peaks used per (patch, rotation)
    ratio_gap=0.0,                  # require peak - background >= this (ratio test)
    weight="sim",                   # vote weight: "sim" | "margin"
    min_sim=0.35,                   # candidate correlation floor
    sim_floor=0.30,                 # vote weight = sim - sim_floor
    bin_t=6.0, tol_t=6.0, rot_slack=1,
    min_ov=6.0,
    variants=("cnt", "frac", "ov", "sumov", "ovsim"),
)


# ------------------------------------------------------------------ helpers --

def _bilinear(field, x, y):
    xc = np.clip(x, 0, W - 1)
    yc = np.clip(y, 0, H - 1)
    x0 = np.floor(xc).astype(np.int32)
    y0 = np.floor(yc).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (xc - x0).astype(np.float32)
    fy = (yc - y0).astype(np.float32)
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy) +
            field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)


def _norm_rows(M):
    """Zero-mean, unit-L2 each row; degenerate rows become zero (NCC -> 0)."""
    M = M - M.mean(axis=1, keepdims=True)
    n = np.sqrt((M * M).sum(axis=1, keepdims=True))
    good = (n > 1e-6)
    return np.where(good, M / np.maximum(n, 1e-9), 0.0).astype(np.float32), good.ravel()


_ECACHE = {}


def _enhanced(img, cfg):
    # NB: keyed on CONTENT, not id().  id() is reused once a temporary array is
    # freed, which silently served stale enhancements and made every synthetic
    # transform test look identical.
    key = (hash(img.tobytes()), cfg["enh"], cfg["osmooth"])
    hit = _ECACHE.get(key)
    if hit is None:
        e, c2, s2, coh = enh.enhance(img, cfg["enh"], osmooth=cfg["osmooth"])
        hit = (e.astype(np.float32), coh.astype(np.float32))
        _ECACHE[key] = hit
    return hit


class Template:
    __slots__ = ("db", "ny", "nx", "r", "qc", "qd", "nq", "coh", "dbok")


def make_template(img, cfg):
    e, coh = _enhanced(img, cfg)
    r, sub = cfg["r"], cfg["sub"]
    off = np.arange(-r, r + 1, sub, dtype=np.float32)
    P = len(off)

    # --- database side: every pixel position whose r-window fits in the image
    win = sliding_window_view(e, (2 * r + 1, 2 * r + 1))          # (ny,nx,P0,P0)
    ny, nx = win.shape[0], win.shape[1]
    cols = win[:, :, ::sub, ::sub].reshape(ny * nx, P * P)
    db, dbok = _norm_rows(cols.astype(np.float32))

    # --- query side: a coarse grid, patches sampled at each candidate rotation
    # a square half-size r rotated by theta needs r*(|cos|+|sin|) of room, which
    # for the small angles used here is far less than the r*sqrt(2) worst case
    mx = max(abs(math.cos(math.radians(d))) + abs(math.sin(math.radians(d)))
             for d in cfg["rots"])
    m = cfg["q_margin"] or int(math.ceil(r * mx)) + 1
    ys = np.arange(m, H - m, cfg["q_step"])
    xs = np.arange(m, W - m, cfg["q_step"])
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    qc = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    if cfg["min_coh"] > 0:
        keep = coh[qc[:, 1].astype(int), qc[:, 0].astype(int)] >= cfg["min_coh"]
        qc = qc[keep]
    nq = len(qc)

    ou, ov = np.meshgrid(off, off, indexing="ij")         # (P,P): ou=dy, ov=dx
    ou, ov = ou.ravel(), ov.ravel()
    qd = np.empty((len(cfg["rots"]), nq, P * P), np.float32)
    for ti, deg in enumerate(cfg["rots"]):
        t = math.radians(deg)
        ct, st = math.cos(t), math.sin(t)
        sx = qc[:, 0:1] + ct * ov[None, :] - st * ou[None, :]
        sy = qc[:, 1:2] + st * ov[None, :] + ct * ou[None, :]
        qd[ti] = _norm_rows(_bilinear(e, sx, sy))[0]

    # --- distinctiveness: a patch of plain parallel ridges matches almost
    # anywhere on any finger, so it can only add noise to the vote.  Measure
    # each patch's best correlation against its OWN image away from its own
    # location; a low value means the patch is locally unique (curvature, a
    # minutia, a scar) and is the only kind worth matching.
    if nq and (cfg["max_amb"] < 1.0 or cfg["keep_best"]):
        Sself = (qd[len(cfg["rots"]) // 2] @ db.T).reshape(nq, ny, nx)
        yy = np.arange(ny)[None, :] + r
        xx = np.arange(nx)[None, :] + r
        near_y = np.abs(yy - qc[:, 1:2]) <= cfg["amb_excl"]
        near_x = np.abs(xx - qc[:, 0:1]) <= cfg["amb_excl"]
        Sself[near_y[:, :, None] & near_x[:, None, :]] = -2.0
        amb = Sself.reshape(nq, -1).max(axis=1)
        keep = amb <= cfg["max_amb"]
        if cfg["keep_best"]:
            order = np.argsort(amb)[:cfg["keep_best"]]
            k2 = np.zeros(nq, bool); k2[order] = True
            keep &= k2
        if keep.sum() >= 4:
            qc, qd, nq = qc[keep], qd[:, keep], int(keep.sum())

    t = Template()
    t.db, t.ny, t.nx, t.r = db, ny, nx, r
    t.qc, t.qd, t.nq, t.coh, t.dbok = qc, qd, nq, coh, dbok
    return t


# ------------------------------------------------------------------ scoring --

def score_one(ta, tb, cfg):
    """Directed: query patches of ta matched into the dense position set of tb."""
    rots = np.asarray(cfg["rots"], np.float32)
    nrot, nq = len(rots), ta.nq
    zero = {v: 0.0 for v in cfg["variants"]}
    if nq < 4:
        return zero

    S = (ta.qd.reshape(nrot * nq, -1) @ tb.db.T).reshape(nrot * nq, tb.ny, tb.nx)

    # top-k peaks per (rotation, patch) with non-maximum suppression
    nrows = nrot * nq
    rows = np.arange(nrows)[:, None, None]
    dd = np.arange(-cfg["nms"], cfg["nms"] + 1)
    cand_p, cand_s = [], []
    flat = S.reshape(nrows, -1)
    ar = np.arange(nrows)
    for _ in range(cfg["topk"] + 1):        # one extra peak: the background level
        k = flat.argmax(axis=1)
        cand_p.append(k)
        cand_s.append(flat[ar, k])
        py, px = np.divmod(k, tb.nx)
        yy = np.clip(py[:, None, None] + dd[None, :, None], 0, tb.ny - 1)
        xx = np.clip(px[:, None, None] + dd[None, None, :], 0, tb.nx - 1)
        S[rows, yy, xx] = -2.0

    bg = cand_s[-1]                          # best rival peak, well away from the rest
    K = cfg["topk"]
    src = np.concatenate([np.tile(np.arange(nq, dtype=np.int32), nrot)] * K)
    rid = np.concatenate([np.repeat(np.arange(nrot, dtype=np.int32), nq)] * K)
    pos = np.concatenate(cand_p[:K])
    sim = np.concatenate(cand_s[:K]).astype(np.float32)
    marg = sim - np.concatenate([bg] * K).astype(np.float32)

    keep = (sim >= cfg["min_sim"]) & (marg >= cfg["ratio_gap"])
    if keep.sum() < 3:
        return zero
    src, rid, pos, sim, marg = (src[keep], rid[keep], pos[keep], sim[keep],
                                marg[keep])

    py, px = np.divmod(pos, tb.nx)
    pbx = (px + tb.r).astype(np.float32)
    pby = (py + tb.r).astype(np.float32)

    # patch content was sampled from A rotated by +theta and matched at pb, so
    # the implied map is  x -> R(-theta) x + tvec.
    th = np.radians(rots[rid])
    ct, st = np.cos(th), np.sin(th)
    ax, ay = ta.qc[src, 0], ta.qc[src, 1]
    rx = ct * ax + st * ay
    ry = -st * ax + ct * ay
    tx, ty = pbx - rx, pby - ry
    base = marg if cfg["weight"] == "margin" else sim - cfg["sim_floor"]
    wt = np.maximum(base, 1e-3).astype(np.float32)

    # --- Hough vote over (tx, ty, rotation), bilinear in translation
    bt = cfg["bin_t"]
    nbx = int(2 * W / bt) + 3
    nby = int(2 * H / bt) + 3
    fx = (tx + W) / bt
    fy = (ty + H) / bt
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    gx, gy = fx - ix, fy - iy
    acc = np.zeros(nbx * nby * nrot, np.float32)
    for dx in (0, 1):
        wx = gx if dx else 1 - gx
        jx = np.clip(ix + dx, 0, nbx - 1)
        for dy in (0, 1):
            wy = gy if dy else 1 - gy
            jy = np.clip(iy + dy, 0, nby - 1)
            np.add.at(acc, (jx * nby + jy) * nrot + rid, wt * wx * wy)
    best = int(np.argmax(acc))
    cx = -W + (best // (nrot * nby) + 0.5) * bt
    cy = -H + ((best // nrot) % nby + 0.5) * bt
    cr = best % nrot

    # refine the alignment: the Hough bin is coarse, so recentre on the inliers
    rok = np.abs(rid - cr) <= cfg["rot_slack"]
    inl = ((np.abs(tx - cx) <= cfg["tol_t"]) & (np.abs(ty - cy) <= cfg["tol_t"]) & rok)
    for _ in range(cfg.get("refine", 2)):
        if not inl.any():
            break
        cx = float((tx[inl] * wt[inl]).sum() / wt[inl].sum())
        cy = float((ty[inl] * wt[inl]).sum() / wt[inl].sum())
        inl = ((np.abs(tx - cx) <= cfg["tol_t"]) &
               (np.abs(ty - cy) <= cfg["tol_t"]) & rok)
    if not inl.any():
        return zero

    # one correspondence per query patch, the strongest
    bestw = np.zeros(nq, np.float32)
    bests = np.zeros(nq, np.float32)
    np.maximum.at(bestw, src[inl], wt[inl])
    np.maximum.at(bests, src[inl], sim[inl])
    matched = bestw > 0
    cnt = float(matched.sum())

    # --- overlap implied by the winning alignment: how many query patches could
    # have had a correspondent at all
    th0 = math.radians(float(rots[cr]))
    c0, s0 = math.cos(th0), math.sin(th0)
    mx = c0 * ta.qc[:, 0] + s0 * ta.qc[:, 1] + cx
    my = -s0 * ta.qc[:, 0] + c0 * ta.qc[:, 1] + cy
    ov = float(((mx >= tb.r) & (mx <= W - 1 - tb.r) &
                (my >= tb.r) & (my <= H - 1 - tb.r)).sum())
    den = max(ov, cfg["min_ov"])

    out = {}
    for v in cfg["variants"]:
        if v == "cnt":
            out[v] = cnt
        elif v == "frac":
            out[v] = cnt / max(nq, 1)
        elif v == "ov":
            out[v] = cnt / den
        elif v == "sumov":
            out[v] = float(bestw.sum()) / den
        elif v == "ovsim":
            out[v] = float(bests.sum()) / den
        elif v == "cntsim":
            out[v] = float(bests.sum())
        else:
            raise KeyError(v)
    return out
