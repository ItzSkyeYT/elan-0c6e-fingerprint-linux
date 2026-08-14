#!/usr/bin/env python3
"""
Local-descriptor fingerprint matching, v2.

Changes from v1 (which measured d' ~ 0.3, i.e. useless):
  * the descriptor samples the ENHANCED RIDGE IMAGE on a rotation-normalised
    log-polar grid, optionally concatenated with relative ridge orientation.
    v1 used orientation alone and was not distinctive (median Lowe ratio 0.95).
  * no Lowe ratio test.  Fingerprint patches are self-similar by construction --
    parallel ridges everywhere -- so the ratio test rejects nearly every correct
    correspondence.  Instead every keypoint proposes its top-K candidates and a
    generalised Hough vote over (tx, ty, dtheta) sorts out which are consistent.
  * the vote is a real 3-D accumulator with soft (trilinear) spreading, then a
    refinement pass that keeps at most one correspondence per source keypoint.

All operations are separable blurs, bilinear samples, dot products and
histogram increments -- a direct C port with no library beyond libm.
"""

import math
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(TOOLS, "matcher-lab.py")).read().split("def main(")[0])

W, H = 150, 52


# --------------------------------------------------------------- ridge maps --

import enh as _enh


def ridge_maps(img, mode="gabor", osmooth=6.0):
    """(enhanced, cos2phi, sin2phi, coherence). phi = ridge direction.

    Uses exp-descriptors/enh.py, not matcher-lab's gabor_enhance() -- the latter
    under-smooths the orientation field and visibly shreds the ridges on this
    sensor (see enh.py).
    """
    e, c2, s2, coh = _enh.enhance(img, mode, osmooth=osmooth)
    return e.astype(np.float32), c2, s2, coh


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


# ---------------------------------------------------------------- keypoints --

def keypoints_grid(coh, step, margin, min_coh):
    ys = np.arange(margin, H - margin, step)
    xs = np.arange(margin, W - margin, step)
    return np.array([(float(x), float(y)) for y in ys for x in xs
                     if coh[int(y), int(x)] >= min_coh], np.float32).reshape(-1, 2)


def _harris(e, sigma=2.0):
    gx = np.gradient(e, axis=1)
    gy = np.gradient(e, axis=0)
    a = _sep_blur(gx * gx, sigma)
    b = _sep_blur(gy * gy, sigma)
    c = _sep_blur(gx * gy, sigma)
    return (a * b - c * c) - 0.04 * (a + b) ** 2


def _curvature(c2, s2, sigma=1.5):
    """Magnitude of the spatial derivative of the doubled-angle field: high
    where ridge flow bends or terminates -- i.e. near minutiae."""
    gxc = np.gradient(c2, axis=1); gyc = np.gradient(c2, axis=0)
    gxs = np.gradient(s2, axis=1); gys = np.gradient(s2, axis=0)
    k = gxc ** 2 + gyc ** 2 + gxs ** 2 + gys ** 2
    return _sep_blur(k, sigma)


def keypoints_peaks(resp, coh, n, margin, min_coh, nms):
    r = np.where(coh >= min_coh, resp, -np.inf).copy()
    r[:margin, :] = -np.inf; r[-margin:, :] = -np.inf
    r[:, :margin] = -np.inf; r[:, -margin:] = -np.inf
    order = np.argsort(r, axis=None)[::-1]
    taken = np.zeros((H, W), bool)
    pts = []
    for idx in order:
        y, x = divmod(int(idx), W)
        if not np.isfinite(r[y, x]):
            break
        if taken[max(0, y - nms):y + nms + 1, max(0, x - nms):x + nms + 1].any():
            continue
        taken[y, x] = True
        pts.append((float(x), float(y)))
        if len(pts) >= n:
            break
    return np.array(pts, np.float32).reshape(-1, 2)


# -------------------------------------------------------------- descriptors --

def build_polar(spec):
    """spec = ((radius, n_angles), ...) -> offsets (n,2) in the local frame."""
    off = [(0.0, 0.0)]
    for r, na in spec:
        for k in range(na):
            a = 2 * math.pi * k / na
            off.append((r * math.cos(a), r * math.sin(a)))
    return np.array(off, np.float32)


def describe(pts, e, c2, s2, coh, offsets, cfg):
    """Rotation-normalised descriptors, two reference frames per keypoint.

    alpha = local ridge direction.  Sampling positions are rotated by alpha and
    ridge orientations are expressed relative to alpha, so rotating the input
    image leaves the descriptor bit-identical (up to interpolation).  alpha is
    only defined mod pi, so the frame is 2-fold ambiguous; frames alpha and
    alpha+pi are both emitted, tagged with the same keypoint id.
    """
    n_off = len(offsets)
    if len(pts) == 0:
        return (np.zeros((0, 0), np.float32), np.zeros((0, 2), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.int32))

    ws = cfg["ref_sigma"]
    rc, _ = _bilinear(_sep_blur(c2 * coh, ws), pts[:, 0], pts[:, 1])
    rs, _ = _bilinear(_sep_blur(s2 * coh, ws), pts[:, 0], pts[:, 1])
    alpha = 0.5 * np.arctan2(rs, rc)
    if not cfg["rot_norm"]:
        alpha = np.zeros_like(alpha)

    frames = [0.0, math.pi] if (cfg["rot_norm"] and cfg["both_frames"]) else [0.0]
    px, py = pts[:, 0], pts[:, 1]
    ox, oy = offsets[None, :, 0], offsets[None, :, 1]

    D, P, A, K = [], [], [], []
    for extra in frames:
        a = alpha + extra
        ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
        sx = (px[:, None] + ca * ox - sa * oy).ravel()
        sy = (py[:, None] + sa * ox + ca * oy).ravel()
        fe, ok = _bilinear(e, sx, sy)
        fco, _ = _bilinear(coh, sx, sy)
        ok = ok.reshape(-1, n_off)
        fe = fe.reshape(-1, n_off) * ok
        fco = fco.reshape(-1, n_off) * ok

        parts = []
        if cfg["use_ridge"]:
            cnt = np.maximum(ok.sum(axis=1, keepdims=True), 1)
            m = (fe * ok).sum(axis=1, keepdims=True) / cnt
            v = (fe - m) * ok
            v = v / np.maximum(np.sqrt((v * v).sum(axis=1, keepdims=True)), 1e-6)
            parts.append(v * cfg["w_ridge"])
        if cfg["use_orient"]:
            fc2, _ = _bilinear(c2, sx, sy); fc2 = fc2.reshape(-1, n_off)
            fs2, _ = _bilinear(s2, sx, sy); fs2 = fs2.reshape(-1, n_off)
            c2a, s2a = np.cos(2 * a)[:, None], np.sin(2 * a)[:, None]
            rc2 = (fc2 * c2a + fs2 * s2a) * fco
            rs2 = (-fc2 * s2a + fs2 * c2a) * fco
            v = np.concatenate([rc2, rs2], axis=1)
            v = v / np.maximum(np.sqrt((v * v).sum(axis=1, keepdims=True)), 1e-6)
            parts.append(v * cfg["w_orient"])
        d = np.concatenate(parts, axis=1)
        d = d / np.maximum(np.sqrt((d * d).sum(axis=1, keepdims=True)), 1e-6)
        D.append(d.astype(np.float32))
        P.append(pts)
        A.append(a.astype(np.float32))
        K.append(np.arange(len(pts), dtype=np.int32))

    return (np.concatenate(D), np.concatenate(P), np.concatenate(A),
            np.concatenate(K))


class Template:
    __slots__ = ("desc", "pts", "ang", "kid", "n_kp", "desc1", "pts1", "ang1", "kid1")


def make_template(img, cfg):
    e, c2, s2, coh = ridge_maps(img, cfg.get("enh", "gabor"), cfg.get("osmooth", 6.0))
    if cfg["kp"] == "grid":
        pts = keypoints_grid(coh, cfg["grid_step"], cfg["margin"], cfg["min_coh"])
    elif cfg["kp"] == "harris":
        pts = keypoints_peaks(_harris(e), coh, cfg["n_kp_max"], cfg["margin"],
                              cfg["min_coh"], cfg["nms"])
    else:  # curvature
        pts = keypoints_peaks(_curvature(c2, s2), coh, cfg["n_kp_max"],
                              cfg["margin"], cfg["min_coh"], cfg["nms"])
    offs = build_polar(cfg["rings"])
    t = Template()
    # both frames -- used as the DATABASE side of a comparison
    d, p, a, k = describe(pts, e, c2, s2, coh, offs, cfg)
    t.desc, t.pts, t.ang, t.kid = d, p, a, k
    # single canonical frame -- used as the QUERY side, so each keypoint
    # proposes candidates once rather than twice
    c1 = dict(cfg); c1["both_frames"] = False
    d1, p1, a1, k1 = describe(pts, e, c2, s2, coh, offs, c1)
    t.desc1, t.pts1, t.ang1, t.kid1 = d1, p1, a1, k1
    t.n_kp = len(pts)
    return t


# ----------------------------------------------------------------- matching --

def match(ta, tb, cfg):
    """Generalised Hough vote over similarity transforms.

    Every query keypoint of A proposes its top-K descriptor matches in B.  A
    correspondence implies a transform (tx, ty, dtheta); true correspondences
    concentrate in one accumulator cell, self-similarity noise does not.
    """
    if ta.n_kp < 4 or tb.n_kp < 4:
        return 0.0
    S = ta.desc1 @ tb.desc.T                       # (na, 2*nb)
    if S.size == 0:
        return 0.0
    K = min(cfg["topk"], S.shape[1])
    idx = np.argpartition(-S, K - 1, axis=1)[:, :K]
    sim = np.take_along_axis(S, idx, axis=1)

    na = S.shape[0]
    src = np.repeat(np.arange(na, dtype=np.int32), K)
    dst = idx.ravel()
    sim = sim.ravel()

    good = sim >= cfg["min_sim"]
    if good.sum() < cfg["min_pairs"]:
        return 0.0
    src, dst, sim = src[good], dst[good], sim[good]

    dth = (tb.ang[dst] - ta.ang1[src] + math.pi) % (2 * math.pi) - math.pi
    ok = np.abs(dth) <= math.radians(cfg["max_rot"])
    if ok.sum() < cfg["min_pairs"]:
        return 0.0
    src, dst, sim, dth = src[ok], dst[ok], sim[ok], dth[ok]

    pa = ta.pts1[src]
    pb = tb.pts[dst]
    ct, st = np.cos(dth), np.sin(dth)
    tx = pb[:, 0] - (ct * pa[:, 0] - st * pa[:, 1])
    ty = pb[:, 1] - (st * pa[:, 0] + ct * pa[:, 1])
    thd = np.degrees(dth)
    wt = np.maximum(sim - cfg["sim_floor"], 0.0).astype(np.float32)

    # ---- 3-D accumulator with trilinear spreading -------------------------
    bt, br = cfg["bin_t"], cfg["bin_r"]
    TXLO, TYLO = -W, -H
    nx = int(2 * W / bt) + 2
    ny = int(2 * H / bt) + 2
    nr = int(2 * cfg["max_rot"] / br) + 2
    fx = (tx - TXLO) / bt
    fy = (ty - TYLO) / bt
    fr = (thd + cfg["max_rot"]) / br
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
    bjr = best % nr
    bjy = (best // nr) % ny
    bjx = best // (nr * ny)
    cx = TXLO + (bjx + 0.5) * bt
    cy = TYLO + (bjy + 0.5) * bt
    cr = -cfg["max_rot"] + (bjr + 0.5) * br

    # ---- refinement: one correspondence per source keypoint --------------
    m = ((np.abs(tx - cx) <= cfg["tol_t"]) & (np.abs(ty - cy) <= cfg["tol_t"]) &
         (np.abs(thd - cr) <= cfg["tol_r"]))
    if not m.any():
        return 0.0
    ms, mw = src[m], wt[m]
    uniq = np.zeros(na, np.float32)
    np.maximum.at(uniq, ms, mw)
    score = float(uniq.sum())
    cnt = float((uniq > 0).sum())

    n_eff = math.sqrt(max(1.0, ta.n_kp) * max(1.0, tb.n_kp))
    if cfg["norm"] == "count":
        return cnt / n_eff * 100.0
    if cfg["norm"] == "raw":
        return score
    if cfg["norm"] == "rawcount":
        return cnt
    return score / n_eff * 100.0


DEFAULT = dict(
    kp="grid", grid_step=5, margin=4, min_coh=0.30, n_kp_max=150, nms=3,
    rings=((3, 8), (6, 12), (9, 12), (12, 16), (15, 16)),
    ref_sigma=2.0, rot_norm=True, both_frames=True,
    use_ridge=True, w_ridge=1.0, use_orient=False, w_orient=0.5,
    topk=3, min_sim=0.0, sim_floor=0.0, min_pairs=3,
    max_rot=24.0, bin_r=8.0, bin_t=4.0, tol_t=5.0, tol_r=10.0,
    norm="sqrtn", enh="gabor", osmooth=6.0,
)
