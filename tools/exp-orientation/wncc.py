"""
Reliability-WEIGHTED normalised cross-correlation, and its joint search with the
orientation-field score.

Why this exists
---------------
Plain NCC over a shift (dx,dy) averages over the ENTIRE overlap rectangle.  On a
150x52 press sensor two genuine captures often share only part of that
rectangle: outside the common contact patch one or both images hold no ridge
signal at all (dry edge, no contact, smear).  Those pixels are pure noise but
they carry full weight, so they dilute a real match towards zero.  That is the
mechanism behind "only 6 of 66 genuine pairs outscore the best impostor pair" --
not a lack of signal, a lack of *masking*.

The orientation matcher in oflib.py already weights by per-block reliability.
Applying the same weighting to the LCN intensity is the obvious missing
experiment, and it costs six correlations instead of three:

    w(x,d) = ra(x) * rb(x-d)

    S(d) = [ <w,ab> - <w,a><w,b>/<w> ]
           / sqrt( (<w,aa> - <w,a>^2/<w>) (<w,bb> - <w,b>^2/<w>) )

with

    <w,ab>(d) = corr(ra*a , rb*b)      <w,a>(d) = corr(ra*a , rb)
    <w,aa>(d) = corr(ra*a*a, rb)       <w,b>(d) = corr(ra , rb*b)
    <w,bb>(d) = corr(ra , rb*b*b)      <w>(d)   = corr(ra , rb)

Six plane pairs, six inverse transforms -- or six direct integer loops in C.
Nothing here needs an FFT in principle; the FFT is only a speed-up for the
python sweep.

The same reliability planes drive the orientation score, so both cues can be
maximised at a COMMON alignment, which is the only honest way to fuse them.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                       # noqa: F401,F403
from oflib import _corr_planes

W, H = 150, 52


# ------------------------------------------------------------ reliability --

def reliability(img, block=8, lcn_sigma=6.0, energy_q=0.40, deg=0, pre=None,
                rel_pow=1.0):
    """Per-pixel ridge-quality weight in [0,1] plus the contrast-normalised
    image it was measured on, both in the rotated frame.

    Identical construction to oflib.make_field so the two cues agree on what
    'this pixel carries usable ridge information' means: coherence of the
    gradient structure tensor times an energy gate.
    """
    g = pre if pre is not None else local_contrast_norm(img, lcn_sigma)
    g, ok = rotate_valid(np.asarray(g, dtype=np.float64), deg)

    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5

    r = max(1, block // 2)
    Gxx = boxfilt(gx * gx, r)
    Gyy = boxfilt(gy * gy, r)
    Gxy = boxfilt(gx * gy, r)
    Vx = Gxx - Gyy
    Vy = 2.0 * Gxy
    mag = np.sqrt(Vx * Vx + Vy * Vy)
    en = Gxx + Gyy
    coh = mag / (en + 1e-12)

    inside = ok
    if deg != 0:
        inside = boxsum_valid(ok.astype(np.float64), r + 1) >= (2 * (r + 1) + 1) ** 2 - 0.5
        if inside.sum() < 50:
            inside = ok
    q = float(np.quantile(en[inside], energy_q)) if inside.any() else 1.0
    rel = coh * (en / (en + q + 1e-12))
    if rel_pow != 1.0:
        rel = rel ** rel_pow
    rel = np.where(inside, rel, 0.0)

    inv = 1.0 / (mag + 1e-12)
    fld = Field(rel * Vx * inv, rel * Vy * inv, rel, (rel > 0.05).astype(np.float64))
    return np.where(inside, g, 0.0), rel, fld


# ------------------------------------------------------- weighted NCC map --

def wncc_planes_a(g, r, ph, pw):
    """FFTs of the three template-side planes."""
    return [np.fft.rfft2(p, s=(ph, pw)) for p in (r * g, r * g * g, r)]


def wncc_planes_b(g, r, ph, pw):
    """Conjugated FFTs of the three probe-side planes."""
    return [np.conj(np.fft.rfft2(p, s=(ph, pw))) for p in (r * g, r * g * g, r)]


def wncc_map(PA, PB, ph, pw, iy, ix, min_w):
    """Weighted correlation coefficient for every shift in the window.

    PA = wncc_planes_a(...)  -> [F(ra*a), F(ra*a*a), F(ra)]
    PB = wncc_planes_b(...)  -> conj of the same three for b
    """
    def c(u, v):
        return np.fft.irfft2(u * v, s=(ph, pw))[iy, ix]

    Wab = c(PA[0], PB[0])
    Wa = c(PA[0], PB[2])
    Wb = c(PA[2], PB[0])
    Waa = c(PA[1], PB[2])
    Wbb = c(PA[2], PB[1])
    Ww = c(PA[2], PB[2])

    Wp = np.maximum(Ww, 1e-9)
    num = Wab - Wa * Wb / Wp
    va = Waa - Wa * Wa / Wp
    vb = Wbb - Wb * Wb / Wp
    den = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
    ok = (Ww >= min_w) & (den > 1e-9)
    return np.where(ok, num / np.maximum(den, 1e-12), -np.inf), Ww


# --------------------------------------------------- overlap-aware shrink --

def shrink(smap, Ww, kappa):
    """Penalise correlations measured on few effective samples.

    A correlation coefficient estimated from n_eff independent samples has
    standard error ~1/sqrt(n_eff) even when the truth is zero, so a wide
    translation search is systematically won by the smallest admissible overlap.
    Multiplying by n/(n+kappa) is the standard shrinkage of a sample correlation
    towards zero and removes that bias without a hard cut-off.  kappa=0 disables.
    """
    if kappa <= 0:
        return smap
    return smap * (Ww / (Ww + kappa))
