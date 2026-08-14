#!/usr/bin/env python3
"""Per-pixel ridge-quality weight.

Looking at the captures directly, a large fraction of several images is not
ridge at all: the fingertip does not cover the whole platen, so part of the
window is a smooth gradient or sensor noise.  The block-variance segmentation
already in minutiae.py calls 99% of every image foreground, which is why
masking it made no difference.

Quality here is the product of two things that actually distinguish ridge from
non-ridge:
  coherence  how consistently the local gradients point one way (structure
             tensor); parallel ridges give ~1, noise and flat areas give ~0.
  energy     the local variance in the ridge frequency band, soft-limited so a
             merely-strong region does not dominate a merely-good one.

The weight then multiplies into a weighted NCC, so blank patches contribute
nothing instead of contributing zero-mean noise that dilutes the score.
"""
import math
import numpy as np

import minutiae as M

W, H = 150, 52


def bandpass(img, lo=1.0, hi=4.0):
    return M._sep_blur(img, lo) - M._sep_blur(img, hi)


def quality(img, tensor_sigma=4.0, smooth=3.0, energy_ref=0.45, gamma=1.0):
    """Return (weight HxW in [0,1], lcn image)."""
    n = M.local_contrast_norm(img, sigma=6.0)
    bp = bandpass(n)

    g = M._sep_blur(n, 1.0)
    gx = np.gradient(g, axis=1)
    gy = np.gradient(g, axis=0)
    Gxx = M._sep_blur(gx * gx, tensor_sigma)
    Gyy = M._sep_blur(gy * gy, tensor_sigma)
    Gxy = M._sep_blur(gx * gy, tensor_sigma)
    num = np.sqrt((Gxx - Gyy) ** 2 + 4.0 * Gxy ** 2)
    den = Gxx + Gyy + 1e-9
    coh = np.clip(num / den, 0.0, 1.0)

    en = np.sqrt(np.maximum(M._sep_blur(bp * bp, tensor_sigma), 0.0))
    enr = en / (np.median(en) + 1e-9)
    esc = enr / (enr + energy_ref)                # soft saturation in [0,1)

    q = M._sep_blur((coh * esc).astype(np.float32), smooth)
    q = np.clip(q, 0.0, 1.0) ** gamma
    return q.astype(np.float64), n.astype(np.float64)
