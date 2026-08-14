#!/usr/bin/env python3
"""Enrolment-time MOSAICKING.

The whole difficulty on this sensor is that a 150x52 window sees a small patch
of fingertip and two presses often share little area.  Every matcher so far
attacks that at VERIFY time, which cannot work: the information simply is not
in the pair.  The fix is to attack it at ENROL time -- register the enrolled
captures to each other once, into a canvas larger than the sensor, so the
stored template covers much more of the finger than any single press does.
A probe then has a large overlap with the template even though it has a small
overlap with each individual enrolled capture.

Enrolment cost is irrelevant (it happens once, off the critical path);
verification is a single weighted correlation against the canvas.

Greedy growth: start from the capture that agrees best with the others, then
repeatedly add whichever remaining capture registers most confidently to the
mosaic so far.  Weights are the ridge-quality map, so blank areas of a capture
neither contribute to nor corrupt the canvas.
"""
import math
import numpy as np

W, H = 150, 52


def rotate_pair(img, wgt, deg):
    """Rotate image and weight together; out-of-frame gets weight 0."""
    if deg == 0:
        return np.asarray(img, np.float64), np.asarray(wgt, np.float64)
    t = math.radians(deg)
    ct, st = math.cos(t), math.sin(t)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    xr = ct * (xx - cx) + st * (yy - cy) + cx
    yr = -st * (xx - cx) + ct * (yy - cy) + cy
    inb = (xr >= 0) & (xr <= W - 1) & (yr >= 0) & (yr <= H - 1)
    x0 = np.clip(np.floor(xr).astype(int), 0, W - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(yr).astype(int), 0, H - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    fx = np.clip(xr - x0, 0, 1); fy = np.clip(yr - y0, 0, 1)

    def samp(f):
        return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x1] * fx * (1 - fy) +
                f[y1, x0] * (1 - fx) * fy + f[y1, x1] * fx * fy)
    return np.where(inb, samp(img), 0.0), np.where(inb, samp(wgt), 0.0)


# ------------------------------------------------------- weighted full NCC --

def _f(x, s):
    return np.fft.rfft2(x, s=s)


def wncc_place(Ai, Aw, Bi, Bw, min_w):
    """Weighted NCC of the small patch B placed at every offset inside the
    canvas A.  Returns (score[oy, ox], mass[oy, ox]) for oy in 0..CH-H,
    ox in 0..CW-W.  Invalid entries are NaN.
    """
    CH, CW = Ai.shape
    h, w = Bi.shape
    s = (CH + h, CW + w)
    a1 = Aw
    aa = Aw * Ai
    a2 = Aw * Ai * Ai
    b1 = Bw
    bb = Bw * Bi
    b2 = Bw * Bi * Bi
    F = lambda x: _f(x, s)
    Fa1, Faa, Fa2 = F(a1), F(aa), F(a2)
    Cb1, Cbb, Cb2 = np.conj(F(b1)), np.conj(F(bb)), np.conj(F(b2))
    irf = np.fft.irfft2
    n = irf(Fa1 * Cb1, s=s)[:CH - h + 1, :CW - w + 1]
    sab = irf(Faa * Cbb, s=s)[:CH - h + 1, :CW - w + 1]
    sa = irf(Faa * Cb1, s=s)[:CH - h + 1, :CW - w + 1]
    sb = irf(Fa1 * Cbb, s=s)[:CH - h + 1, :CW - w + 1]
    sa2 = irf(Fa2 * Cb1, s=s)[:CH - h + 1, :CW - w + 1]
    sb2 = irf(Fa1 * Cb2, s=s)[:CH - h + 1, :CW - w + 1]
    ok = n >= min_w
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.where(ok, n, 1.0)
        num = sab - sa * sb / nn
        va = np.maximum(sa2 - sa * sa / nn, 0.0)
        vb = np.maximum(sb2 - sb * sb / nn, 0.0)
        den = np.sqrt(va * vb)
        sc = num / np.where(den > 1e-12, den, np.nan)
    return np.where(ok, sc, np.nan), n


# ------------------------------------------------------------- the mosaic --

class Mosaic:
    def __init__(self, pad_x=70, pad_y=24, rots=range(-12, 13, 4)):
        self.px, self.py = pad_x, pad_y
        self.CW, self.CH = W + 2 * pad_x, H + 2 * pad_y
        self.rots = list(rots)
        self.sw = np.zeros((self.CH, self.CW))
        self.swi = np.zeros((self.CH, self.CW))
        self.placed = 0

    def image(self):
        m = self.sw > 1e-6
        out = np.zeros_like(self.swi)
        out[m] = self.swi[m] / self.sw[m]
        return out

    def weight(self, cap=1.0):
        return np.minimum(self.sw, cap)

    def add(self, img, wgt, oy, ox, deg):
        i, q = rotate_pair(img, wgt, deg)
        self.sw[oy:oy + H, ox:ox + W] += q
        self.swi[oy:oy + H, ox:ox + W] += q * i
        self.placed += 1

    def register(self, img, wgt, min_w):
        """Best (score, oy, ox, deg) for this capture against the mosaic."""
        A = self.image()
        Aw = self.weight()
        best = (-2.0, 0, 0, 0)
        for d in self.rots:
            i, q = rotate_pair(img, wgt, d)
            sc, n = wncc_place(A, Aw, i, q, min_w)
            if not np.isfinite(sc).any():
                continue
            k = int(np.nanargmax(sc))
            oy, ox = divmod(k, sc.shape[1])
            v = float(sc[oy, ox])
            if v > best[0]:
                best = (v, oy, ox, d)
        return best


def build(imgs, wgts, min_w_frac=0.30, pad_x=70, pad_y=24,
          rots=range(-12, 13, 4), accept=-1.0):
    """Greedy mosaic over a list of (image, weight) enrolment captures."""
    n = len(imgs)
    mo = Mosaic(pad_x, pad_y, rots)
    mw = float(np.mean([w.mean() for w in wgts]))
    min_w = min_w_frac * (W * H) * mw * mw

    # seed with the capture that registers best against a plain-centre placement
    # of every other one: cheap proxy = the medoid by pairwise weighted NCC
    if n == 1:
        seed = 0
    else:
        sc = np.zeros((n, n))
        tmp = Mosaic(0, 0, rots)
        for i in range(n):
            tmp.sw = wgts[i].copy(); tmp.swi = wgts[i] * imgs[i]
            for j in range(n):
                if i == j:
                    continue
                v, _, _, _ = tmp.register(imgs[j], wgts[j], min_w)
                sc[i, j] = v
        seed = int(np.argmax(sc.sum(axis=1)))

    mo.add(imgs[seed], wgts[seed], pad_y, pad_x, 0)
    remaining = [k for k in range(n) if k != seed]
    order = []
    while remaining:
        cands = [(mo.register(imgs[k], wgts[k], min_w), k) for k in remaining]
        cands.sort(key=lambda c: -c[0][0])
        (v, oy, ox, d), k = cands[0]
        if v < accept:
            break
        mo.add(imgs[k], wgts[k], oy, ox, d)
        order.append((k, v, oy, ox, d))
        remaining.remove(k)
    return mo, order


def score_probe(mo, img, wgt, min_w_frac=0.35, cap=1.0):
    A = mo.image()
    Aw = mo.weight(cap)
    mw = float(wgt.mean())
    min_w = min_w_frac * (W * H) * mw * mw
    best = -1.0
    for d in mo.rots:
        i, q = rotate_pair(img, wgt, d)
        sc, n = wncc_place(A, Aw, i, q, min_w)
        if np.isfinite(sc).any():
            v = float(np.nanmax(sc))
            if v > best:
                best = v
    return best
