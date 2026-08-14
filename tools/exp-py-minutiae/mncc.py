#!/usr/bin/env python3
"""Masked normalised cross-correlation over all (rotation, translation).

The existing baseline correlates the whole rectangular overlap, including
pixels where one capture has no finger on the platen at all.  On a press
sensor the contact area is a fraction of the window and varies press to press,
so that background dilutes every genuine score.  Here the correlation is
restricted to pixels that are FOREGROUND IN BOTH captures, with the per-shift
sums obtained exactly via FFT correlations (Padfield 2010).

Returns the full score map so the caller can pick several candidate
alignments, not just the argmax.
"""
import math
import numpy as np

W, H = 150, 52


def _corr_maps(A, B, pad_y, pad_x):
    """corr[dy, dx] = sum_p A[p] * B[p - (dy,dx)], for all lags, via FFT."""
    ph, pw = H + 2 * pad_y + 1, W + 2 * pad_x + 1
    fa = np.fft.rfft2(A, s=(ph, pw))
    fb = np.fft.rfft2(B, s=(ph, pw))
    return np.fft.irfft2(fa * np.conj(fb), s=(ph, pw)), ph, pw


def masked_ncc_map(a, ma, b, mb, max_dx, max_dy, min_px):
    """NCC restricted to the intersection of the two foreground masks.

    a, b: float images.  ma, mb: bool masks.  Convention a[y,x] <-> b[y-dy,x-dx].
    Returns (score[2*max_dy+1, 2*max_dx+1], npix[...]) with -1 where invalid.
    """
    a = np.asarray(a, np.float64) * ma
    b = np.asarray(b, np.float64) * mb
    fa = ma.astype(np.float64)
    fb = mb.astype(np.float64)

    py, px = max_dy, max_dx
    Cn, ph, pw = _corr_maps(fa, fb, py, px)          # overlap pixel count
    Cab, _, _ = _corr_maps(a, b, py, px)
    Ca, _, _ = _corr_maps(a, fb, py, px)
    Cb, _, _ = _corr_maps(fa, b, py, px)
    Ca2, _, _ = _corr_maps(a * a, fb, py, px)
    Cb2, _, _ = _corr_maps(fa, b * b, py, px)

    dys = np.arange(-max_dy, max_dy + 1)
    dxs = np.arange(-max_dx, max_dx + 1)
    iy = dys % ph
    ix = dxs % pw
    sel = np.ix_(iy, ix)

    n = Cn[sel]
    sab = Cab[sel]
    sa = Ca[sel]
    sb = Cb[sel]
    sa2 = Ca2[sel]
    sb2 = Cb2[sel]

    ok = n >= min_px
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.where(ok, n, 1.0)
        num = sab - sa * sb / nn
        va = np.maximum(sa2 - sa * sa / nn, 0.0)
        vb = np.maximum(sb2 - sb * sb / nn, 0.0)
        den = np.sqrt(va * vb)
        s = np.where((den > 1e-9) & ok, num / np.where(den > 1e-9, den, 1.0), -1.0)
    return s.astype(np.float64), n


def rotate(img, deg, fill=None, order_nearest=False):
    if deg == 0:
        return img
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
    if fill is not None:
        out = np.where(inb, out, fill)
    return out


def rotate_mask(m, deg):
    if deg == 0:
        return m
    return rotate(m.astype(np.float64), deg, fill=0.0) > 0.5
