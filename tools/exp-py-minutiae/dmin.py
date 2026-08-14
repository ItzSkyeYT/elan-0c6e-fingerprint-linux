#!/usr/bin/env python3
"""Minutiae with quality gating and a local orientation descriptor.

Two things are wrong with the plain crossing-number pipeline on this sensor:

  * it detects MORE minutiae on the impostor finger (20.8/image) than on the
    genuine one (14.0), because several captures have large regions with no
    usable ridge at all and the thinned skeleton there is pure noise.  Any
    geometric matcher then finds more "agreeing" points for impostors, which
    is exactly what the ceiling analysis measured (oracle d' = -0.30).
  * with ~15 points in a 150x52 window and an 8-10 px position tolerance, the
    chance of a spurious agreement under some alignment is high, so the Hough
    vote is dominated by coincidences.

Fixes here: gate minutiae on the ridge-quality map, and attach to each minutia
a rotation-normalised descriptor of the surrounding orientation field, so a
correspondence has to agree on local ridge FLOW, not just on a point position.
"""
import math
import numpy as np

import minutiae as M
import quality as Q

W, H = 150, 52

# descriptor sampling grid (polar, in the minutia's own frame)
D_RADII = (5.0, 9.0, 13.0, 17.0)
D_NANG = 10


class DTemplate:
    __slots__ = ("m", "desc", "dw", "q", "n", "theta", "lcn")

    def __init__(self, m, desc, dw, q, theta, lcn):
        self.m = m
        self.desc = desc          # (N, 2*len(R)*NANG)
        self.dw = dw              # (N,) descriptor reliability
        self.q = q
        self.n = len(m)
        self.theta = theta
        self.lcn = lcn


def _bilinear(field, X, Y):
    x0 = np.clip(np.floor(X).astype(int), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(Y).astype(int), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = np.clip(X - x0, 0, 1); fy = np.clip(Y - y0, 0, 1)
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy) +
            field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)


def descriptors(m, theta, q):
    """For each minutia, sample cos/sin of the DOUBLED orientation difference
    relative to the minutia direction, on a polar grid in the minutia frame.

    Doubled angles because ridge orientation is defined mod pi.  Each sample is
    weighted by the local quality; samples off the platen weigh nothing.
    """
    ox = np.cos(2.0 * theta)
    oy = np.sin(2.0 * theta)
    npt = len(D_RADII) * D_NANG
    out = np.zeros((len(m), 2 * npt), np.float32)
    wgt = np.zeros(len(m), np.float32)
    if len(m) == 0:
        return out, wgt
    angs = np.linspace(0, 2 * math.pi, D_NANG, endpoint=False)
    for i in range(len(m)):
        mx, my, md = float(m[i, 0]), float(m[i, 1]), float(m[i, 2])
        px, py, ws = [], [], []
        for r in D_RADII:
            for a in angs:
                px.append(mx + r * math.cos(a + md))
                py.append(my + r * math.sin(a + md))
        px = np.array(px); py = np.array(py)
        inb = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        wq = np.where(inb, _bilinear(q, np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)), 0.0)
        cx = _bilinear(ox, np.clip(px, 0, W - 1), np.clip(py, 0, H - 1))
        cy = _bilinear(oy, np.clip(px, 0, W - 1), np.clip(py, 0, H - 1))
        # rotate the doubled-angle vector into the minutia frame
        c2, s2 = math.cos(2 * md), math.sin(2 * md)
        rx = cx * c2 + cy * s2
        ry = -cx * s2 + cy * c2
        v = np.concatenate([rx * wq, ry * wq]).astype(np.float32)
        nrm = float(np.linalg.norm(v))
        out[i] = v / nrm if nrm > 1e-6 else v
        wgt[i] = float(wq.mean())
    return out, wgt


def make_template(img, cfg=None):
    cfg = cfg or {}
    q, lcn = Q.quality(img, energy_ref=cfg.get("energy_ref", 0.45))
    qthr = cfg.get("qthr", 0.35)
    mask = q >= qthr
    if mask.mean() < 0.15:                      # never mask everything away
        mask = q >= np.quantile(q, 0.5)

    th, coh = M.orientation(lcn, block=8, smooth=2.5)
    enh = M.gabor(lcn, th, cfg.get("freq", 0.110), n_orient=16, ksize=11,
                  sx=4.0, sy=4.0)
    b = M.binarise(enh, mask, k=cfg.get("bin_k", 7))
    sk = M.thin(b)
    m = M.extract_minutiae(sk, th, mask,
                           ridge_period=1.0 / cfg.get("freq", 0.110),
                           border=cfg.get("border", 4),
                           merge_dist=cfg.get("merge_dist", None),
                           spur_len=cfg.get("spur_len", 6),
                           prune_rounds=3,
                           min_ridge=cfg.get("min_ridge", 8))
    # drop minutiae in low-quality neighbourhoods
    if len(m):
        qv = np.array([q[int(round(y)), int(round(x))] for x, y, _, _ in m])
        m = m[qv >= cfg.get("min_q", 0.45)]
    # cap to the strongest N by local quality, so a noisy capture cannot
    # outvote a clean one purely by having more points
    cap = cfg.get("cap", 0)
    if cap and len(m) > cap:
        qv = np.array([q[int(round(y)), int(round(x))] for x, y, _, _ in m])
        m = m[np.argsort(-qv)[:cap]]
    d, dw = descriptors(m, th, q)
    return DTemplate(m, d, dw, q, th, lcn)


# ------------------------------------------------------------------ match --

def match(ta, tb, rot_max=20.0, rot_bin=5.0, tr_bin=8.0, pos_tol=9.0,
          dir_tol=math.radians(25.0), desc_thr=0.55, top_k=6,
          min_denom=6.0, use_desc=True):
    A, B = ta.m, tb.m
    if len(A) == 0 or len(B) == 0:
        return 0.0
    Dsim = ta.desc @ tb.desc.T if use_desc else np.ones((len(A), len(B)), np.float32)

    nrot = int(2 * rot_max / rot_bin) + 1
    rots = np.linspace(-rot_max, rot_max, nrot)
    votes = {}
    for ri, rdeg in enumerate(rots):
        t = math.radians(rdeg)
        ct, st = math.cos(t), math.sin(t)
        rax = ct * A[:, 0] - st * A[:, 1]
        ray = st * A[:, 0] + ct * A[:, 1]
        for i in range(len(A)):
            dd = (B[:, 2] - (A[i, 2] + t)) % math.pi
            dd = np.minimum(dd, math.pi - dd)
            ok = (dd <= dir_tol) & (Dsim[i] >= desc_thr)
            if not ok.any():
                continue
            for j in np.nonzero(ok)[0]:
                X = B[j, 0] - rax[i]
                Y = B[j, 1] - ray[i]
                key = (ri, int(round(X / tr_bin)), int(round(Y / tr_bin)))
                votes[key] = votes.get(key, 0.0) + float(Dsim[i, j])
    if not votes:
        return 0.0
    best = sorted(votes.items(), key=lambda kv: -kv[1])[:top_k]
    out = 0.0
    for (ri, ix, iy), _v in best:
        t = math.radians(rots[ri])
        for sx in (-0.5, 0.0, 0.5):
            for sy in (-0.5, 0.0, 0.5):
                s = _score(ta, tb, Dsim, t, (ix + sx) * tr_bin, (iy + sy) * tr_bin,
                           pos_tol, dir_tol, desc_thr, min_denom)
                if s > out:
                    out = s
    return out


def _inside(x, y, margin=4.0):
    return (x >= margin) & (x < W - margin) & (y >= margin) & (y < H - margin)


def _score(ta, tb, Dsim, t, tx, ty, pos_tol, dir_tol, desc_thr, min_denom):
    A, B = ta.m, tb.m
    ct, st = math.cos(t), math.sin(t)
    ax = ct * A[:, 0] - st * A[:, 1] + tx
    ay = st * A[:, 0] + ct * A[:, 1] + ty
    ad = A[:, 2] + t
    bx = ct * (B[:, 0] - tx) + st * (B[:, 1] - ty)
    by = -st * (B[:, 0] - tx) + ct * (B[:, 1] - ty)

    inA = _inside(ax, ay)
    inB = _inside(bx, by)
    nA, nB = int(inA.sum()), int(inB.sum())
    if nA == 0 or nB == 0:
        return 0.0
    ia = np.nonzero(inA)[0]; ib = np.nonzero(inB)[0]
    dx = ax[ia][:, None] - B[ib, 0][None, :]
    dy = ay[ia][:, None] - B[ib, 1][None, :]
    d2 = dx * dx + dy * dy
    dd = (ad[ia][:, None] - B[ib, 2][None, :]) % math.pi
    dd = np.minimum(dd, math.pi - dd)
    ds = Dsim[np.ix_(ia, ib)]
    cand = (d2 <= pos_tol ** 2) & (dd <= dir_tol) & (ds >= desc_thr)
    if not cand.any():
        return 0.0
    cost = np.where(cand, d2 - 1000.0 * ds, np.inf)
    ua = np.zeros(len(ia), bool); ub = np.zeros(len(ib), bool)
    tot = 0.0
    for k in np.argsort(cost, axis=None):
        if not np.isfinite(cost.flat[k]):
            break
        i, j = divmod(int(k), cost.shape[1])
        if ua[i] or ub[j]:
            continue
        ua[i] = ub[j] = True
        # each accepted pair contributes its descriptor agreement, not just 1
        tot += float(ds[i, j])
    den = math.sqrt(max(nA, min_denom) * max(nB, min_denom))
    return tot / den
