#!/usr/bin/env python3
"""Quality-WEIGHTED normalised cross-correlation over (rotation, translation).

Same exact per-shift statistics as a plain NCC, but every pixel carries the
ridge-quality weight of BOTH captures, so regions where either capture has no
usable ridge simply drop out of the correlation instead of diluting it.  All
six weighted sums come from FFT correlations, and the per-image FFTs are cached
so a full 45x45 matrix is cheap.
"""
import math
import numpy as np

W, H = 150, 52


class WPrep:
    __slots__ = ("F", "ph", "pw", "rots")

    def __init__(self):
        self.F = {}


def _rotate(img, deg, fill=0.0):
    if deg == 0:
        return np.asarray(img, np.float64)
    t = math.radians(deg)
    ct, st = math.cos(t), math.sin(t)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    xr = ct * (xx - cx) + st * (yy - cy) + cx
    yr = -st * (xx - cx) + ct * (yy - cy) + cy
    inb = (xr >= 0) & (xr <= W - 1) & (yr >= 0) & (yr <= H - 1)
    x0 = np.clip(np.floor(xr).astype(int), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(yr).astype(int), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = np.clip(xr - x0, 0, 1)
    fy = np.clip(yr - y0, 0, 1)
    out = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
           img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return np.where(inb, out, fill)


def prep(a, q, rots, max_dx, max_dy):
    """Cache the five FFTs needed per rotation: q, q*a, q*a^2 (and the plain
    q, q*a again for the other side -- the same three suffice both roles)."""
    p = WPrep()
    p.ph, p.pw = H + 2 * max_dy + 1, W + 2 * max_dx + 1
    p.rots = list(rots)
    for d in rots:
        qd = _rotate(q, d, 0.0) if d else np.asarray(q, np.float64)
        ad = _rotate(a, d, 0.0) if d else np.asarray(a, np.float64)
        qd = np.clip(qd, 0.0, 1.0)
        f = np.fft.rfft2
        p.F[d] = (f(qd, s=(p.ph, p.pw)),
                  f(qd * ad, s=(p.ph, p.pw)),
                  f(qd * ad * ad, s=(p.ph, p.pw)))
    return p


def surface(pa, pb, rots, max_dx, max_dy, min_w):
    """Weighted-NCC surface (len(rots), 2*max_dy+1, 2*max_dx+1) and the
    weight mass per shift.  A is at rotation 0, B is rotated."""
    ph, pw = pa.ph, pa.pw
    Qa, Aa, A2a = pa.F[0]
    dys = np.arange(-max_dy, max_dy + 1)
    dxs = np.arange(-max_dx, max_dx + 1)
    sel = np.ix_(dys % ph, dxs % pw)

    out = np.full((len(rots), len(dys), len(dxs)), np.nan)
    wm = np.zeros((len(rots), len(dys), len(dxs)))
    irf = np.fft.irfft2
    for k, d in enumerate(rots):
        Qb, Ab, A2b = pb.F[d]
        n = irf(Qa * np.conj(Qb), s=(ph, pw))[sel]
        sab = irf(Aa * np.conj(Ab), s=(ph, pw))[sel]
        sa = irf(Aa * np.conj(Qb), s=(ph, pw))[sel]
        sb = irf(Qa * np.conj(Ab), s=(ph, pw))[sel]
        sa2 = irf(A2a * np.conj(Qb), s=(ph, pw))[sel]
        sb2 = irf(Qa * np.conj(A2b), s=(ph, pw))[sel]
        ok = n >= min_w
        with np.errstate(invalid="ignore", divide="ignore"):
            nn = np.where(ok, n, 1.0)
            num = sab - sa * sb / nn
            va = np.maximum(sa2 - sa * sa / nn, 0.0)
            vb = np.maximum(sb2 - sb * sb / nn, 0.0)
            den = np.sqrt(va * vb)
            s = num / np.where(den > 1e-12, den, np.nan)
        out[k] = np.where(ok, s, np.nan)
        wm[k] = n
    return out, wm
