#!/usr/bin/env python3
"""
Local-descriptor fingerprint matching for the 150x52 ELAN press sensor.

Idea: correlation needs broad overlap between two captures; local descriptors
need only a handful of corresponding patches.  We therefore

  1. enhance the image and compute a doubled-angle ridge orientation field,
  2. place keypoints (dense grid, or a corner-response peak picker),
  3. build a rotation-normalised descriptor per keypoint by sampling the ridge
     orientation on concentric rings in a frame aligned to the local ridge
     direction (the "Tico-Kuosmanen" style descriptor),
  4. match with nearest-neighbour + Lowe ratio test,
  5. enforce geometric consistency with a Hough vote over (dx, dy, dtheta),
  6. score by the number / quality of geometrically consistent matches.

Everything is plain array arithmetic (bilinear sampling, separable blurs,
histogram votes) so the port to C is mechanical: no FFT, no scipy, no OpenCV.
"""

import math
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(TOOLS, "matcher-lab.py")).read().split("def main(")[0])

W, H = 150, 52


# --------------------------------------------------------------- ridge maps --

def ridge_maps(img, block_smooth=2.5, coh_smooth=3.0):
    """Return (enhanced, c2, s2, coh).

    c2/s2 are the doubled-angle ridge-DIRECTION unit vector components, i.e.
    (cos 2phi, sin 2phi) where phi is the direction along the ridge.  Doubled
    angles are the only sane representation: orientation is mod pi.
    coh in [0,1] is the local orientation coherence, used as a reliability
    weight so smeared / empty regions contribute nothing.
    """
    e = gabor_enhance(img)

    gx = np.gradient(e, axis=1)
    gy = np.gradient(e, axis=0)
    vx = 2.0 * gx * gy                 # sin of doubled gradient angle * mag^2
    vy = gx ** 2 - gy ** 2             # cos of doubled gradient angle * mag^2
    sx = _sep_blur(vx, block_smooth)
    sy = _sep_blur(vy, block_smooth)
    mag = _sep_blur(gx ** 2 + gy ** 2, block_smooth)

    r = np.sqrt(sx ** 2 + sy ** 2)
    coh = r / (mag + 1e-6)
    coh = np.clip(_sep_blur(coh, coh_smooth), 0.0, 1.0).astype(np.float32)

    # gradient-normal doubled angle -> ridge direction is +90 deg, which in
    # doubled-angle space is a negation of the vector.
    inv = 1.0 / (r + 1e-9)
    c2 = (-sy * inv).astype(np.float32)
    s2 = (-sx * inv).astype(np.float32)
    return e.astype(np.float32), c2, s2, coh


def _bilinear(field, x, y):
    """Bilinear sample with an out-of-bounds mask. x,y are float arrays."""
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

def keypoints_grid(coh, step=5, margin=4, min_coh=0.30):
    ys = np.arange(margin, H - margin, step)
    xs = np.arange(margin, W - margin, step)
    pts = [(float(x), float(y)) for y in ys for x in xs if coh[int(y), int(x)] >= min_coh]
    return np.array(pts, dtype=np.float32).reshape(-1, 2)


def _corner_response(e, sigma=2.0):
    """Harris response of the enhanced ridge image."""
    gx = np.gradient(e, axis=1)
    gy = np.gradient(e, axis=0)
    a = _sep_blur(gx * gx, sigma)
    b = _sep_blur(gy * gy, sigma)
    c = _sep_blur(gx * gy, sigma)
    det = a * b - c * c
    tr = a + b
    return det - 0.04 * tr * tr


def keypoints_harris(e, coh, n=120, margin=5, min_coh=0.30, nms=3):
    r = _corner_response(e)
    r = np.where(coh >= min_coh, r, -np.inf)
    r[:margin, :] = -np.inf
    r[-margin:, :] = -np.inf
    r[:, :margin] = -np.inf
    r[:, -margin:] = -np.inf
    order = np.argsort(r, axis=None)[::-1]
    taken = np.zeros((H, W), dtype=bool)
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
    return np.array(pts, dtype=np.float32).reshape(-1, 2)


# -------------------------------------------------------------- descriptors --

def build_rings(radii, n_ang):
    """Sampling offsets in the keypoint's own frame: list of (dx, dy)."""
    off = []
    for r in radii:
        for k in range(n_ang):
            a = 2 * math.pi * k / n_ang
            off.append((r * math.cos(a), r * math.sin(a)))
    return np.array(off, dtype=np.float32)


def describe(pts, c2, s2, coh, offsets, ref_sigma=2.0, rot_norm=True,
             both_frames=True, min_valid=0.55):
    """Rotation-normalised orientation descriptors.

    For keypoint p with reference ridge direction alpha, we sample the ridge
    orientation phi at p + R(alpha)*offset and store (cos 2(phi-alpha),
    sin 2(phi-alpha)) weighted by coherence.  Rotating the whole image by beta
    rotates both the sampling positions and every phi and alpha by beta, so the
    stored values are unchanged -- the descriptor is genuinely rotation
    invariant, not merely rotation tolerant.

    alpha is only defined mod pi, so the frame has a 2-fold ambiguity.  We emit
    BOTH frames (alpha and alpha+pi) as separate descriptors tagged with the
    same keypoint id, and the ratio test later ignores the sibling.
    """
    if len(pts) == 0:
        return (np.zeros((0, 2 * len(offsets)), np.float32),
                np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int32))

    # local reference angle: coherence-weighted mean doubled-angle vector
    wc2 = _sep_blur(c2 * coh, ref_sigma)
    ws2 = _sep_blur(s2 * coh, ref_sigma)
    px, py = pts[:, 0], pts[:, 1]
    rc, _ = _bilinear(wc2, px, py)
    rs, _ = _bilinear(ws2, px, py)
    alpha = 0.5 * np.arctan2(rs, rc)                     # ridge direction

    frames = [0.0, math.pi] if (rot_norm and both_frames) else [0.0]
    if not rot_norm:
        alpha = np.zeros_like(alpha)
        frames = [0.0]

    D, P, A, K = [], [], [], []
    n_off = len(offsets)
    for fi, extra in enumerate(frames):
        a = alpha + extra
        ca, sa = np.cos(a), np.sin(a)
        # sample positions: (n_pts, n_off)
        ox = offsets[None, :, 0]
        oy = offsets[None, :, 1]
        sx = px[:, None] + ca[:, None] * ox - sa[:, None] * oy
        sy = py[:, None] + sa[:, None] * ox + ca[:, None] * oy
        fc2, ok = _bilinear(c2, sx.ravel(), sy.ravel())
        fs2, _ = _bilinear(s2, sx.ravel(), sy.ravel())
        fco, _ = _bilinear(coh, sx.ravel(), sy.ravel())
        fc2 = fc2.reshape(-1, n_off)
        fs2 = fs2.reshape(-1, n_off)
        fco = (fco.reshape(-1, n_off) * ok.reshape(-1, n_off)).astype(np.float32)

        # rotate the doubled-angle vectors into the keypoint frame:
        # phi - alpha  =>  doubled angle rotates by -2*alpha
        c2a = np.cos(2 * a)[:, None]
        s2a = np.sin(2 * a)[:, None]
        rc2 = fc2 * c2a + fs2 * s2a
        rs2 = -fc2 * s2a + fs2 * c2a

        d = np.concatenate([rc2 * fco, rs2 * fco], axis=1)
        nrm = np.sqrt((d * d).sum(axis=1, keepdims=True))
        valid = fco.mean(axis=1) / max(1e-6, 1.0)
        keep = (nrm[:, 0] > 1e-6)
        d = d / np.maximum(nrm, 1e-6)

        D.append(d)
        P.append(pts)
        A.append(a % (2 * math.pi))
        K.append(np.arange(len(pts), dtype=np.int32))
        _ = valid, keep, min_valid, fi

    return (np.concatenate(D).astype(np.float32),
            np.concatenate(P).astype(np.float32),
            np.concatenate(A).astype(np.float32),
            np.concatenate(K).astype(np.int32))


# ------------------------------------------------------------------ per-img --

class Template:
    __slots__ = ("desc", "pts", "ang", "kid", "n_kp")

    def __init__(self, desc, pts, ang, kid, n_kp):
        self.desc, self.pts, self.ang, self.kid, self.n_kp = desc, pts, ang, kid, n_kp


def make_template(img, cfg):
    e, c2, s2, coh = ridge_maps(img)
    if cfg["kp"] == "grid":
        pts = keypoints_grid(coh, cfg["grid_step"], cfg["margin"], cfg["min_coh"])
    else:
        pts = keypoints_harris(e, coh, cfg["n_harris"], cfg["margin"], cfg["min_coh"])
    offs = build_rings(cfg["radii"], cfg["n_ang"])
    d, p, a, k = describe(pts, c2, s2, coh, offs,
                          ref_sigma=cfg["ref_sigma"],
                          rot_norm=cfg["rot_norm"],
                          both_frames=cfg["both_frames"])
    return Template(d, p, a, k, len(pts))


# ----------------------------------------------------------------- matching --

def match(ta, tb, cfg):
    """Score two templates: ratio-tested NN correspondences filtered by a Hough
    vote over similarity-transform parameters (dx, dy, dtheta)."""
    if ta.n_kp < 4 or tb.n_kp < 4:
        return 0.0

    # cosine similarity; descriptors are unit norm so this is 1 - d^2/2
    S = ta.desc @ tb.desc.T                      # (na, nb)
    if S.size == 0:
        return 0.0

    # For each descriptor in A, best match in B, plus the best match belonging
    # to a DIFFERENT keypoint of B that is also spatially far away -- otherwise
    # the ratio test is defeated by the sibling frame and by grid neighbours.
    nb = S.shape[1]
    order = np.argsort(-S, axis=1)
    bx, by = tb.pts[:, 0], tb.pts[:, 1]

    pairs = []
    top = min(nb, cfg["ratio_scan"])
    for i in range(S.shape[0]):
        oi = order[i]
        j1 = oi[0]
        s1 = S[i, j1]
        if s1 < cfg["min_sim"]:
            continue
        s2v = None
        for jj in oi[1:top]:
            if tb.kid[jj] == tb.kid[j1]:
                continue
            if (bx[jj] - bx[j1]) ** 2 + (by[jj] - by[j1]) ** 2 < cfg["nn_excl"] ** 2:
                continue
            s2v = S[i, jj]
            break
        if s2v is None:
            continue
        # cosine -> euclidean distance on unit vectors: d = sqrt(2-2s)
        d1 = math.sqrt(max(0.0, 2 - 2 * s1))
        d2 = math.sqrt(max(0.0, 2 - 2 * s2v))
        if d2 <= 1e-9 or d1 / d2 > cfg["ratio"]:
            continue
        pairs.append((i, int(j1), float(s1), d1 / d2))

    # Both reference frames of the same keypoint are in the descriptor set, so a
    # single true correspondence can appear twice.  Keep the best entry per
    # source keypoint so the vote counts keypoints, not descriptor rows.
    if pairs:
        bykp = {}
        for p in pairs:
            k = int(ta.kid[p[0]])
            if k not in bykp or p[2] > bykp[k][2]:
                bykp[k] = p
        pairs = list(bykp.values())

    if len(pairs) < cfg["min_pairs"]:
        return 0.0

    ai = np.array([p[0] for p in pairs])
    bj = np.array([p[1] for p in pairs])
    sim = np.array([p[2] for p in pairs], dtype=np.float32)
    rat = np.array([p[3] for p in pairs], dtype=np.float32)

    dth = (tb.ang[bj] - ta.ang[ai] + math.pi) % (2 * math.pi) - math.pi
    # rotation limited: reject wild ones
    keep = np.abs(dth) <= math.radians(cfg["max_rot"])
    if keep.sum() < cfg["min_pairs"]:
        return 0.0
    ai, bj, sim, rat, dth = ai[keep], bj[keep], sim[keep], rat[keep], dth[keep]

    pa = ta.pts[ai]
    pb = tb.pts[bj]
    ct, st = np.cos(dth), np.sin(dth)
    # translation implied if B = R(dth) * A + t
    tx = pb[:, 0] - (ct * pa[:, 0] - st * pa[:, 1])
    ty = pb[:, 1] - (st * pa[:, 0] + ct * pa[:, 1])

    wt = (1.0 - rat)                       # confident matches vote harder
    best = _hough(tx, ty, np.degrees(dth), wt, cfg)
    n_eff = math.sqrt(max(1.0, ta.n_kp) * max(1.0, tb.n_kp))
    if cfg["norm"] == "sqrtn":
        return float(best / n_eff * 100.0)
    if cfg["norm"] == "none":
        return float(best)
    return float(best / n_eff * 100.0)


def _hough(tx, ty, dthdeg, wt, cfg):
    """Soft-binned vote over (tx, ty, dtheta); returns the best bin mass.

    Instead of a real 3-D accumulator we cluster greedily: for every candidate
    correspondence, count the mass of all correspondences within the tolerance
    box of it.  With <200 candidates this is cheaper than allocating a grid and
    it avoids bin-boundary artefacts entirely.  (Trivially portable to C.)
    """
    n = len(tx)
    if n == 0:
        return 0.0
    dt = cfg["tol_t"]
    da = cfg["tol_r"]
    best = 0.0
    for i in range(n):
        m = ((np.abs(tx - tx[i]) <= dt) & (np.abs(ty - ty[i]) <= dt) &
             (np.abs(dthdeg - dthdeg[i]) <= da))
        v = float(wt[m].sum())
        if v > best:
            best = v
    return best


DEFAULT = dict(
    kp="grid", grid_step=5, margin=4, min_coh=0.30, n_harris=120,
    radii=(3.0, 6.0, 9.0, 12.0), n_ang=8, ref_sigma=2.0,
    rot_norm=True, both_frames=True,
    ratio=0.85, ratio_scan=40, nn_excl=8.0, min_sim=0.0,
    min_pairs=3, max_rot=25.0, tol_t=6.0, tol_r=12.0, norm="sqrtn",
)
