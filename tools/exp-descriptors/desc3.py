#!/usr/bin/env python3
"""
Local-descriptor fingerprint matching, v3.

What v2 got wrong: keypoint repeatability.  A dense grid in BOTH images means a
grid point of A generally falls between grid points of B, and at an 8.8 px ridge
period a 2 px localisation error rotates the sampled ridge phase by ~80 degrees,
so the descriptor no longer matches.  Measured: translating one capture by
(6, 2) px already lost 60% of its own matches.  Corner/curvature detectors were
worse still -- on this sensor they are not repeatable across presses.

v3 makes the comparison asymmetric:
    query side  -- sparse keypoints (stride ~6), one canonical frame,
    database side -- DENSE keypoints (stride 2), both reference frames,
so every query patch has a database sample within ~1 px of its true
correspondent and localisation error stops mattering.  This is patch matching
by exhaustive search, expressed as a descriptor dot product.

Scoring is unchanged in spirit: top-K candidates per query keypoint, a
generalised Hough vote over (tx, ty, dtheta) with trilinear spreading, then a
refinement pass that keeps at most one correspondence per query keypoint.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enh

W, H = 150, 52


def _bilinear(field, x, y):
    ok = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    xc = np.clip(x, 0, W - 1)
    yc = np.clip(y, 0, H - 1)
    x0 = np.floor(xc).astype(np.int32)
    y0 = np.floor(yc).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (xc - x0).astype(np.float32)
    fy = (yc - y0).astype(np.float32)
    v = (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy) +
         field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)
    return v, ok


def build_polar(spec):
    off = [(0.0, 0.0)]
    for r, na in spec:
        for k in range(na):
            a = 2 * math.pi * k / na
            off.append((r * math.cos(a), r * math.sin(a)))
    return np.array(off, np.float32)


def grid_pts(coh, step, margin, min_coh):
    ys = np.arange(margin, H - margin, step)
    xs = np.arange(margin, W - margin, step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    gy = gy.ravel().astype(np.float32)
    gx = gx.ravel().astype(np.float32)
    if min_coh > 0:
        m = coh[gy.astype(int), gx.astype(int)] >= min_coh
        gy, gx = gy[m], gx[m]
    return np.stack([gx, gy], axis=1)


def describe(pts, e, c2, s2, coh, offsets, cfg, both_frames):
    """Rotation-normalised descriptor.

    alpha = local ridge direction (from the coherence-weighted doubled-angle
    field).  Sampling offsets are rotated by alpha and, if orientation channels
    are used, the sampled doubled angles are expressed relative to 2*alpha.
    Rotating the input image by beta rotates alpha and every sampled orientation
    by beta and rotates the sampling positions with the image, so the descriptor
    is unchanged: genuinely rotation-normalised, not merely tolerant.

    alpha is defined only mod pi; both frames (alpha, alpha+pi) are emitted on
    the database side so the query's single canonical frame always has a
    counterpart.
    """
    n_off = len(offsets)
    if len(pts) == 0:
        z = np.zeros
        return z((0, 1), np.float32), z((0, 2), np.float32), z(0, np.float32), z(0, np.int32)

    rc, _ = _bilinear(enh._sep_blur(c2 * coh, cfg["ref_sigma"]), pts[:, 0], pts[:, 1])
    rs, _ = _bilinear(enh._sep_blur(s2 * coh, cfg["ref_sigma"]), pts[:, 0], pts[:, 1])
    alpha = 0.5 * np.arctan2(rs, rc)
    if not cfg["rot_norm"]:
        alpha = np.zeros_like(alpha)
        both_frames = False

    frames = [0.0, math.pi] if both_frames else [0.0]
    px, py = pts[:, 0], pts[:, 1]
    ox, oy = offsets[None, :, 0], offsets[None, :, 1]

    D, P, A, K = [], [], [], []
    for extra in frames:
        a = alpha + extra
        ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
        sx = (px[:, None] + ca * ox - sa * oy).ravel()
        sy = (py[:, None] + sa * ox + ca * oy).ravel()
        fe, ok = _bilinear(e, sx, sy)
        ok = ok.reshape(-1, n_off).astype(np.float32)
        fe = fe.reshape(-1, n_off) * ok

        parts = []
        if cfg["w_ridge"] > 0:
            cnt = np.maximum(ok.sum(axis=1, keepdims=True), 1.0)
            v = (fe - (fe * ok).sum(axis=1, keepdims=True) / cnt) * ok
            v /= np.maximum(np.sqrt((v * v).sum(axis=1, keepdims=True)), 1e-6)
            parts.append(v * cfg["w_ridge"])
        if cfg["w_orient"] > 0:
            fco, _ = _bilinear(coh, sx, sy)
            fco = fco.reshape(-1, n_off) * ok
            fc2, _ = _bilinear(c2, sx, sy); fc2 = fc2.reshape(-1, n_off)
            fs2, _ = _bilinear(s2, sx, sy); fs2 = fs2.reshape(-1, n_off)
            c2a, s2a = np.cos(2 * a)[:, None], np.sin(2 * a)[:, None]
            v = np.concatenate([(fc2 * c2a + fs2 * s2a) * fco,
                                (-fc2 * s2a + fs2 * c2a) * fco], axis=1)
            v /= np.maximum(np.sqrt((v * v).sum(axis=1, keepdims=True)), 1e-6)
            parts.append(v * cfg["w_orient"])
        d = np.concatenate(parts, axis=1)
        d /= np.maximum(np.sqrt((d * d).sum(axis=1, keepdims=True)), 1e-6)
        D.append(d.astype(np.float32))
        P.append(pts)
        A.append(a.astype(np.float32))
        K.append(np.arange(len(pts), dtype=np.int32))
    return (np.concatenate(D), np.concatenate(P), np.concatenate(A),
            np.concatenate(K))


class Template:
    __slots__ = ("qd", "qp", "qa", "nq", "dd", "dp", "da", "ndb")


def make_template(img, cfg):
    e, c2, s2, coh = enh.enhance(img, cfg["enh"], osmooth=cfg["osmooth"])
    e = e.astype(np.float32)
    offs = build_polar(cfg["rings"])
    t = Template()
    qp = grid_pts(coh, cfg["q_step"], cfg["margin"], cfg["min_coh"])
    dp = grid_pts(coh, cfg["db_step"], cfg["db_margin"], cfg["db_min_coh"])
    t.qd, t.qp, t.qa, _ = describe(qp, e, c2, s2, coh, offs, cfg, False)
    t.dd, t.dp, t.da, _ = describe(dp, e, c2, s2, coh, offs, cfg, True)
    t.nq, t.ndb = len(qp), len(dp)
    return t


def _score_one(ta, tb, cfg):
    """Directed score: query keypoints of ta against the dense database of tb."""
    if ta.nq < 4 or tb.ndb < 8:
        return 0.0
    S = ta.qd @ tb.dd.T
    K = min(cfg["topk"], S.shape[1])
    idx = np.argpartition(-S, K - 1, axis=1)[:, :K]
    sim = np.take_along_axis(S, idx, axis=1)

    na = S.shape[0]
    src = np.repeat(np.arange(na, dtype=np.int32), K)
    dst = idx.ravel()
    sim = sim.ravel()
    m = sim >= cfg["min_sim"]
    if m.sum() < cfg["min_pairs"]:
        return 0.0
    src, dst, sim = src[m], dst[m], sim[m]

    dth = (tb.da[dst] - ta.qa[src] + math.pi) % (2 * math.pi) - math.pi
    m = np.abs(dth) <= math.radians(cfg["max_rot"])
    if m.sum() < cfg["min_pairs"]:
        return 0.0
    src, dst, sim, dth = src[m], dst[m], sim[m], dth[m]

    pa, pb = ta.qp[src], tb.dp[dst]
    ct, st = np.cos(dth), np.sin(dth)
    tx = pb[:, 0] - (ct * pa[:, 0] - st * pa[:, 1])
    ty = pb[:, 1] - (st * pa[:, 0] + ct * pa[:, 1])
    thd = np.degrees(dth)
    wt = np.maximum(sim - cfg["sim_floor"], 0.0).astype(np.float32)

    bt, br, mr = cfg["bin_t"], cfg["bin_r"], cfg["max_rot"]
    nx = int(2 * W / bt) + 2
    ny = int(2 * H / bt) + 2
    nr = int(2 * mr / br) + 2
    fx = (tx + W) / bt
    fy = (ty + H) / bt
    fr = (thd + mr) / br
    ix, iy, ir = np.floor(fx).astype(int), np.floor(fy).astype(int), np.floor(fr).astype(int)
    gx, gy, gr = fx - ix, fy - iy, fr - ir
    acc = np.zeros(nx * ny * nr, np.float32)
    for dx in (0, 1):
        wx = gx if dx else 1 - gx
        jx = np.clip(ix + dx, 0, nx - 1)
        for dy in (0, 1):
            wy = gy if dy else 1 - gy
            jy = np.clip(iy + dy, 0, ny - 1)
            for dr in (0, 1):
                wr = gr if dr else 1 - gr
                jr = np.clip(ir + dr, 0, nr - 1)
                np.add.at(acc, (jx * ny + jy) * nr + jr, wt * wx * wy * wr)
    best = int(np.argmax(acc))
    cx = -W + (best // (nr * ny) + 0.5) * bt
    cy = -H + ((best // nr) % ny + 0.5) * bt
    cr = -mr + (best % nr + 0.5) * br

    m = ((np.abs(tx - cx) <= cfg["tol_t"]) & (np.abs(ty - cy) <= cfg["tol_t"]) &
         (np.abs(thd - cr) <= cfg["tol_r"]))
    if not m.any():
        return 0.0
    uniq = np.zeros(na, np.float32)
    np.maximum.at(uniq, src[m], wt[m])
    if cfg["norm"] == "count":
        return float((uniq > 0).sum()) / max(1, na) * 100.0
    if cfg["norm"] == "rawcount":
        return float((uniq > 0).sum())
    if cfg["norm"] == "raw":
        return float(uniq.sum())
    return float(uniq.sum()) / max(1, na) * 100.0


def match(ta, tb, cfg):
    s = _score_one(ta, tb, cfg)
    if cfg["symmetrise"] == "max":
        s = max(s, _score_one(tb, ta, cfg))
    elif cfg["symmetrise"] == "mean":
        s = 0.5 * (s + _score_one(tb, ta, cfg))
    return s


DEFAULT = dict(
    enh="gabor", osmooth=6.0,
    q_step=6, margin=8, min_coh=0.0,
    db_step=2, db_margin=4, db_min_coh=0.0,
    rings=((3, 8), (6, 12), (9, 16), (12, 16), (15, 16)),
    ref_sigma=2.0, rot_norm=True,
    w_ridge=1.0, w_orient=0.0,
    topk=3, min_sim=0.0, sim_floor=0.0, min_pairs=3,
    max_rot=20.0, bin_r=8.0, bin_t=4.0, tol_t=5.0, tol_r=10.0,
    norm="sqrtn", symmetrise="max",
)
