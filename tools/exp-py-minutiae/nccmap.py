#!/usr/bin/env python3
"""Full NCC surface over (rotation, dx, dy) with cached FFTs, plus several
ways of turning that surface into ONE score.

The baseline takes the plain argmax.  That is exactly the wrong statistic when
the search window is wide: an impostor gets to pick the luckiest of hundreds of
alignments, and a small overlap gives a noisier -- hence on average larger --
maximum.  The alternatives here correct for both effects:

  rmax   plain max NCC                       (the baseline statistic)
  z      max over alignments of Fisher-z * sqrt(n_eff-3), where n_eff is the
         overlap area divided by the ridge correlation area (~9x9 px).  Trades
         correlation strength against overlap size on a common scale, so a wide
         search cannot be won by a tiny fluke overlap.
  psr    peak-to-sidelobe ratio: (peak - mean) / sd of the whole surface.
         Measures how much the best alignment stands out from the rest, which
         is what actually distinguishes a real registration from a lucky one.
"""
import math
import numpy as np

W, H = 150, 52


def _integral(x):
    return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


class Prep:
    """Per-image cached data for the correlation surface."""
    __slots__ = ("img", "rot", "F", "ia", "ia2", "ph", "pw")

    def __init__(self, img):
        self.img = img


def prep(img, rots, max_dx, max_dy, rotate):
    ph, pw = H + 2 * max_dy + 1, W + 2 * max_dx + 1
    p = Prep(img)
    p.ph, p.pw = ph, pw
    p.rot, p.F, p.ia, p.ia2 = {}, {}, {}, {}
    for d in rots:
        r = rotate(img, d) if d else img
        r = np.asarray(r, np.float64)
        p.rot[d] = r
        p.F[d] = np.fft.rfft2(r, s=(ph, pw))
        p.ia[d] = _integral(r)
        p.ia2[d] = _integral(r * r)
    return p


def _rect(ii, y0, y1, x0, x1):
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def surface(pa, pb, rots, max_dx, max_dy, min_overlap):
    """NCC surface, shape (len(rots), 2*max_dy+1, 2*max_dx+1).

    Image A is held at rotation 0; image B is rotated by each `rots` entry.
    Convention a[y,x] <-> b[y-dy, x-dx].  Invalid entries are NaN.
    Also returns the overlap pixel count for each (dy,dx).
    """
    ph, pw = pa.ph, pa.pw
    dys = np.arange(-max_dy, max_dy + 1)
    dxs = np.arange(-max_dx, max_dx + 1)

    # overlap rectangle bounds, per (dy, dx)
    y0 = np.maximum(0, dys); y1 = np.minimum(H, H + dys)
    x0 = np.maximum(0, dxs); x1 = np.minimum(W, W + dxs)
    n = np.outer(y1 - y0, x1 - x0).astype(np.float64)
    valid = n >= min_overlap

    ia, ia2 = pa.ia[0], pa.ia2[0]
    SA = np.zeros_like(n); SA2 = np.zeros_like(n)
    for i, dy in enumerate(dys):
        for j, dx in enumerate(dxs):
            if not valid[i, j]:
                continue
            SA[i, j] = _rect(ia, y0[i], y1[i], x0[j], x1[j])
            SA2[i, j] = _rect(ia2, y0[i], y1[i], x0[j], x1[j])

    out = np.full((len(rots), len(dys), len(dxs)), np.nan)
    for k, d in enumerate(rots):
        corr = np.fft.irfft2(pa.F[0] * np.conj(pb.F[d]), s=(ph, pw))
        ib, ib2 = pb.ia[d], pb.ia2[d]
        SB = np.zeros_like(n); SB2 = np.zeros_like(n)
        for i, dy in enumerate(dys):
            for j, dx in enumerate(dxs):
                if not valid[i, j]:
                    continue
                SB[i, j] = _rect(ib, y0[i] - dy, y1[i] - dy, x0[j] - dx, x1[j] - dx)
                SB2[i, j] = _rect(ib2, y0[i] - dy, y1[i] - dy, x0[j] - dx, x1[j] - dx)
        cross = corr[np.ix_(dys % ph, dxs % pw)]
        with np.errstate(invalid="ignore", divide="ignore"):
            nn = np.where(valid, n, 1.0)
            num = cross - SA * SB / nn
            va = np.maximum(SA2 - SA * SA / nn, 0.0)
            vb = np.maximum(SB2 - SB * SB / nn, 0.0)
            den = np.sqrt(va * vb)
            s = num / np.where(den > 1e-9, den, np.nan)
        out[k] = np.where(valid, s, np.nan)
    return out, n


def stats(S, n, corr_area=81.0, clip=0.999):
    """Turn a surface into the candidate scalar scores."""
    fin = np.isfinite(S)
    if not fin.any():
        return dict(rmax=-1.0, z=0.0, psr=0.0, zpsr=0.0)
    v = S[fin]
    rmax = float(v.max())

    neff = np.maximum(n / corr_area, 4.0)             # independent samples
    neff3 = np.broadcast_to(neff, S.shape)
    r = np.clip(S, -clip, clip)
    with np.errstate(invalid="ignore"):
        Z = np.arctanh(r) * np.sqrt(np.maximum(neff3 - 3.0, 1.0))
    Z = np.where(fin, Z, -np.inf)
    z = float(Z.max())

    mu, sd = float(v.mean()), float(v.std())
    psr = (rmax - mu) / sd if sd > 1e-9 else 0.0

    zv = Z[np.isfinite(Z)]
    zmu, zsd = float(zv.mean()), float(zv.std())
    zpsr = (z - zmu) / zsd if zsd > 1e-9 else 0.0
    return dict(rmax=rmax, z=z, psr=float(psr), zpsr=float(zpsr))
