#!/usr/bin/env python3
"""Ridge enhancement tuned for this sensor.

matcher-lab's gabor_enhance() measurably *degrades* the ridges here: it
estimates orientation with a 2 px smoothing radius, which on an 8.8 px ridge
period is far too local, so the per-pixel "nearest quantised orientation"
selection flips between neighbouring filters and shreds the ridge flow.

Measured ridge period (radial power spectrum, all 45 captures): 8.8 px,
f = 0.113 cycles/px.  Orientation is therefore smoothed over ~1.5 ridge
periods, and the Gabor kernel is elongated along the ridge.
"""

import math
import os

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(TOOLS, "matcher-lab.py")).read().split("def main(")[0])

RIDGE_F = 0.113


def orient(img, pre=1.0, smooth=6.0):
    g = _sep_blur(img, pre) if pre else img
    gx = np.gradient(g, axis=1)
    gy = np.gradient(g, axis=0)
    sx = _sep_blur(2.0 * gx * gy, smooth)
    sy = _sep_blur(gx ** 2 - gy ** 2, smooth)
    mag = _sep_blur(gx ** 2 + gy ** 2, smooth)
    r = np.sqrt(sx ** 2 + sy ** 2)
    coh = np.clip(r / (mag + 1e-9), 0.0, 1.0).astype(np.float32)
    # gradient-normal doubled angle -> ridge direction: negate the vector
    inv = 1.0 / (r + 1e-12)
    return (-sy * inv).astype(np.float32), (-sx * inv).astype(np.float32), coh


def gabor(img, freq=RIDGE_F, n_orient=16, ksize=15, sx=3.0, sy=5.0,
          osmooth=6.0):
    """Orientation-steered Gabor, elongated along the ridge."""
    x = local_contrast_norm(img)
    c2, s2, coh = orient(x, 1.0, osmooth)
    phi = 0.5 * np.arctan2(s2, c2)            # ridge direction, mod pi

    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    p = np.pad(x, r, mode="edge")
    H, W = x.shape
    responses = np.empty((n_orient, H, W), np.float32)
    for ai in range(n_orient):
        a = math.pi * ai / n_orient           # ridge direction
        # along-ridge axis u, across-ridge axis v
        u = xx * math.cos(a) + yy * math.sin(a)
        v = -xx * math.sin(a) + yy * math.cos(a)
        k = np.exp(-(v ** 2) / (2 * sx ** 2) - (u ** 2) / (2 * sy ** 2)) \
            * np.cos(2 * math.pi * freq * v)
        k -= k.mean()
        acc = np.zeros_like(x)
        for dy in range(ksize):
            for dx in range(ksize):
                if k[dy, dx] != 0.0:
                    acc += k[dy, dx] * p[dy:dy + H, dx:dx + W]
        responses[ai] = acc
    idx = np.rint((phi % math.pi) / math.pi * n_orient).astype(np.int32) % n_orient
    out = np.take_along_axis(responses, idx[None], axis=0)[0]
    return normalize_global(out), c2, s2, coh


def enhance(img, mode="gabor", **kw):
    if mode == "lcn":
        e = local_contrast_norm(img)
        c2, s2, coh = orient(e, 1.0, kw.get("osmooth", 6.0))
        return normalize_global(e), c2, s2, coh
    if mode == "bp":
        e = bandpass(img, 1.0, 4.0)
        c2, s2, coh = orient(e, 1.0, kw.get("osmooth", 6.0))
        return normalize_global(e), c2, s2, coh
    return gabor(img, **kw)
