"""
Ridge-orientation-field matching for the ELAN 0c6e 150x52 press sensor.

Core idea: describe each capture by a block-level ridge orientation field in the
DOUBLED-ANGLE domain (theta -> (cos 2theta, sin 2theta)), with a per-block
reliability weight (coherence x energy gate).  Two fields are compared with a
reliability-weighted mean of cos(2 dtheta) over the overlapping region, which is
exactly a normalised cross-correlation of the doubled-angle vector fields:

    S(dx,dy) = [ (Ax * Bx)(dx,dy) + (Ay * By)(dx,dy) ] / (Ra * Rb)(dx,dy)

where A = r_a * (cos 2t_a, sin 2t_a), B likewise, and '*' is plain correlation.
Because r = 0 outside the sensor / in unreliable regions, partial overlap costs
nothing beyond the reduction in effective sample count -- which is what we want
for a sensor whose window is smaller than the placement variation.

Rotation: the field is recomputed from the ROTATED image.  Rotating an image
rotates the ridge orientations with it, so resampling the field alone would be
wrong; recomputing sidesteps the "forgot to add the rotation angle" bug entirely
and is cheap at 150x52.  (Verified by test_rotation_consistency below.)

Everything here is integer-loop / box-filter / FFT-of-power-of-two friendly, so
it ports to C with libfprint's toolbox (no scipy, no OpenCV).
"""

import math
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# reuse the existing helpers verbatim (load_pgm, load_dataset, dprime, eer,
# far_at_frr, local_contrast_norm, bandpass, _rotate, ...)
exec(open(os.path.join(TOOLS, "matcher-lab.py")).read().split("def main(")[0])

W, H = 150, 52


# ------------------------------------------------------------- box filter --

def boxfilt(x, r):
    """Mean over a (2r+1)^2 window, edge-replicated.  O(N) via integral image;
    this is what the C port would use."""
    if r <= 0:
        return x.astype(np.float64)
    x = x.astype(np.float64)
    p = np.pad(x, ((r, r), (r, r)), mode="edge")
    c = np.cumsum(p, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1])), c], 0)
    y = c[2 * r + 1:, :] - c[:-(2 * r + 1), :]
    c = np.cumsum(y, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1)), c], 1)
    z = c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]
    return z / float((2 * r + 1) ** 2)


def boxsum_valid(x, r):
    """Same window but with ZERO padding (used for masks)."""
    if r <= 0:
        return x.astype(np.float64)
    x = x.astype(np.float64)
    p = np.pad(x, ((r, r), (r, r)), mode="constant")
    c = np.cumsum(p, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1])), c], 0)
    y = c[2 * r + 1:, :] - c[:-(2 * r + 1), :]
    c = np.cumsum(y, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1)), c], 1)
    return c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]


# --------------------------------------------------------------- rotation --

def rotate_valid(img, deg):
    """Rotate about the centre, bilinear, ZERO outside.  Returns (rotated,
    valid) where valid marks pixels whose source lay inside the frame.  Edge
    replication (as in matcher-lab's _rotate) would invent ridge structure in
    the corners, which an orientation field happily believes."""
    h, w = img.shape
    if deg == 0:
        return img.astype(np.float64), np.ones((h, w), bool)
    t = math.radians(deg)
    ct, st = math.cos(t), math.sin(t)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xr = ct * (xx - cx) + st * (yy - cy) + cx
    yr = -st * (xx - cx) + ct * (yy - cy) + cy
    ok = (xr >= 0) & (xr <= w - 1) & (yr >= 0) & (yr <= h - 1)
    xrc = np.clip(xr, 0, w - 1)
    yrc = np.clip(yr, 0, h - 1)
    x0 = np.floor(xrc).astype(int)
    y0 = np.floor(yrc).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xrc - x0
    fy = yrc - y0
    out = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
           img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return np.where(ok, out, 0.0), ok


# --------------------------------------------------------- the field itself --

class Field:
    """Doubled-angle orientation field with reliability weights.

    ax, ay : r * (cos 2theta, sin 2theta)   (theta = gradient angle; the ridge
             angle is theta+90deg, an irrelevant constant offset since both
             sides get it)
    r      : reliability in [0,1]
    v      : validity indicator (r above a floor) -- used only to count blocks
    """
    __slots__ = ("ax", "ay", "r", "v")

    def __init__(self, ax, ay, r, v):
        self.ax, self.ay, self.r, self.v = ax, ay, r, v


def make_field(img, block=8, lcn_sigma=6.0, energy_q=0.40, deg=0,
               pre=None, rel_floor=0.05, rel_pow=1.0):
    """Block orientation field of `img`, optionally rotated by `deg` first.

    `pre` may be a precomputed contrast-normalised image (so LCN is done once
    per capture rather than once per rotation).
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

    Vx = Gxx - Gyy                     # = E[cos 2theta] * energy
    Vy = 2.0 * Gxy                     # = E[sin 2theta] * energy
    mag = np.sqrt(Vx * Vx + Vy * Vy)
    en = Gxx + Gyy
    coh = mag / (en + 1e-12)           # 0..1, how consistent the flow is

    # energy gate: blocks with little ridge signal (no finger, smeared contact)
    # must not vote.  Quantile of the *rotated-valid* region only.
    inside = ok.copy()
    if deg != 0:
        inside = boxsum_valid(ok.astype(np.float64), r + 1) >= (2 * (r + 1) + 1) ** 2 - 0.5
    if inside.sum() < 50:
        inside = ok
    q = float(np.quantile(en[inside], energy_q)) if inside.any() else 1.0
    gate = en / (en + q + 1e-12)

    rel = coh * gate
    if rel_pow != 1.0:
        rel = rel ** rel_pow
    rel = np.where(inside, rel, 0.0)

    inv = 1.0 / (mag + 1e-12)
    ax = rel * Vx * inv
    ay = rel * Vy * inv
    v = (rel > rel_floor).astype(np.float64)
    return Field(ax, ay, rel, v)


# --------------------------------------------- correlation over all shifts --

def _corr_planes(max_dx, max_dy):
    ph, pw = H + 2 * max_dy + 1, W + 2 * max_dx + 1
    dy = np.arange(-max_dy, max_dy + 1)
    dx = np.arange(-max_dx, max_dx + 1)
    iy = (dy % ph)[:, None]
    ix = (dx % pw)[None, :]
    return ph, pw, iy, ix


def fft_pack(f, ph, pw, conj=False):
    """rfft2 of the four planes of a Field, ready for correlation."""
    F = [np.fft.rfft2(p, s=(ph, pw)) for p in (f.ax, f.ay, f.r, f.v)]
    if conj:
        F = [np.conj(x) for x in F]
    return F


def orient_score(FA, FB, ph, pw, iy, ix, min_blocks, block):
    """Best reliability-weighted mean cos(2 dtheta) over the shift window.

    FA is fft_pack(a), FB is fft_pack(b, conj=True).
    Returns (best_score, best_dx, best_dy) with score in [-1, 1].
    """
    num = np.fft.irfft2(FA[0] * FB[0] + FA[1] * FB[1], s=(ph, pw))
    den = np.fft.irfft2(FA[2] * FB[2], s=(ph, pw))
    cnt = np.fft.irfft2(FA[3] * FB[3], s=(ph, pw))

    n = num[iy, ix]
    d = den[iy, ix]
    c = cnt[iy, ix]
    # count in units of independent blocks
    ok = (c / float(block * block) >= min_blocks) & (d > 1e-9)
    if not ok.any():
        return -1.0, 0, 0
    s = np.where(ok, n / np.maximum(d, 1e-12), -np.inf)
    k = int(np.argmax(s))
    return float(s.flat[k]), int(k % s.shape[1]), int(k // s.shape[1])


def orient_score_centred(FA, FB, ph, pw, iy, ix, min_blocks, block):
    """Reliability-weighted CORRELATION COEFFICIENT of the doubled-angle fields.

    The plain weighted mean of cos(2 dtheta) (orient_score above) saturates:
    on a 150x52 patch the ridge flow is nearly uniform, so any two fingers agree
    to ~0.9 and the between-finger variation drowns in it.  That is precisely the
    failure of raw-intensity NCC, and the cure is the same -- subtract the mean
    over the OVERLAP before correlating, so only the *deviation* of the flow from
    the locally dominant direction counts.

        S = [ <w,a.b> - <w,a>.<w,b>/<w> ]
            / sqrt( (<w> - |<w,a>|^2/<w>) (<w> - |<w,b>|^2/<w>) )

    with w = r_a r_b.  Every term is a plain cross-correlation of two planes, so
    all shifts come out of seven FFT products (or seven direct loops in C).
    """
    Cab = np.fft.irfft2(FA[0] * FB[0] + FA[1] * FB[1], s=(ph, pw))[iy, ix]
    Sax = np.fft.irfft2(FA[0] * FB[2], s=(ph, pw))[iy, ix]
    Say = np.fft.irfft2(FA[1] * FB[2], s=(ph, pw))[iy, ix]
    Sbx = np.fft.irfft2(FA[2] * FB[0], s=(ph, pw))[iy, ix]
    Sby = np.fft.irfft2(FA[2] * FB[1], s=(ph, pw))[iy, ix]
    Sw = np.fft.irfft2(FA[2] * FB[2], s=(ph, pw))[iy, ix]
    cnt = np.fft.irfft2(FA[3] * FB[3], s=(ph, pw))[iy, ix]

    Swp = np.maximum(Sw, 1e-9)
    num = Cab - (Sax * Sbx + Say * Sby) / Swp
    va = Sw - (Sax * Sax + Say * Say) / Swp
    vb = Sw - (Sbx * Sbx + Sby * Sby) / Swp
    den = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
    ok = (cnt / float(block * block) >= min_blocks) & (den > 1e-9) & (Sw > 1e-6)
    if not ok.any():
        return -1.0, 0, 0
    s = np.where(ok, num / np.maximum(den, 1e-12), -np.inf)
    k = int(np.argmax(s))
    return float(s.flat[k]), int(k % s.shape[1]), int(k // s.shape[1])


# ------------------------------------------------------------ NCC baseline --

def ncc_all_shifts_vec(a, b, max_dx, max_dy, min_overlap, fa=None, fb=None):
    """Vectorised rewrite of fastncc.ncc_all_shifts (same convention, same
    result -- checked in selftest.py).  The python loop there is too slow for a
    2000-pair matrix."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape
    ph, pw = h + 2 * max_dy + 1, w + 2 * max_dx + 1
    if fa is None:
        fa = np.fft.rfft2(a, s=(ph, pw))
    if fb is None:
        fb = np.conj(np.fft.rfft2(b, s=(ph, pw)))
    corr = np.fft.irfft2(fa * fb, s=(ph, pw))

    ia = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ia2 = np.pad(a * a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ib = np.pad(b, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ib2 = np.pad(b * b, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    dy = np.arange(-max_dy, max_dy + 1)[:, None]
    dx = np.arange(-max_dx, max_dx + 1)[None, :]
    y0 = np.maximum(0, dy); y1 = np.minimum(h, h + dy)
    x0 = np.maximum(0, dx); x1 = np.minimum(w, w + dx)
    y0, y1 = np.broadcast_to(y0, (dy.size, dx.size)), np.broadcast_to(y1, (dy.size, dx.size))
    x0, x1 = np.broadcast_to(x0, (dy.size, dx.size)), np.broadcast_to(x1, (dy.size, dx.size))
    n = (y1 - y0) * (x1 - x0)

    def rect(ii, ry0, ry1, rx0, rx1):
        return ii[ry1, rx1] - ii[ry0, rx1] - ii[ry1, rx0] + ii[ry0, rx0]

    sa = rect(ia, y0, y1, x0, x1)
    sa2 = rect(ia2, y0, y1, x0, x1)
    sb = rect(ib, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
    sb2 = rect(ib2, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
    cross = corr[dy % ph, dx % pw]

    nn = np.maximum(n, 1)
    num = cross - sa * sb / nn
    va = sa2 - sa * sa / nn
    vb = sb2 - sb * sb / nn
    den = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    ok = (n >= min_overlap) & (den > 1e-9)
    if not ok.any():
        return -1.0
    return float(np.max(np.where(ok, num / np.maximum(den, 1e-12), -np.inf)))


# ---------------------------------------------- maps, for a JOINT search --
#
# Fusing two scores that were each maximised over their OWN best alignment is
# sloppy: for an impostor pair the two maxima usually sit at different shifts,
# and taking both inflates the impostor score.  Scoring both cues at the SAME
# shift and maximising the sum is strictly more informative and costs nothing
# extra once the correlation maps exist.

def orient_map(FA, FB, ph, pw, iy, ix, min_blocks, block):
    """Centred orientation correlation for every shift in the window."""
    Cab = np.fft.irfft2(FA[0] * FB[0] + FA[1] * FB[1], s=(ph, pw))[iy, ix]
    Sax = np.fft.irfft2(FA[0] * FB[2], s=(ph, pw))[iy, ix]
    Say = np.fft.irfft2(FA[1] * FB[2], s=(ph, pw))[iy, ix]
    Sbx = np.fft.irfft2(FA[2] * FB[0], s=(ph, pw))[iy, ix]
    Sby = np.fft.irfft2(FA[2] * FB[1], s=(ph, pw))[iy, ix]
    Sw = np.fft.irfft2(FA[2] * FB[2], s=(ph, pw))[iy, ix]
    cnt = np.fft.irfft2(FA[3] * FB[3], s=(ph, pw))[iy, ix]
    Swp = np.maximum(Sw, 1e-9)
    num = Cab - (Sax * Sbx + Say * Sby) / Swp
    va = Sw - (Sax * Sax + Say * Say) / Swp
    vb = Sw - (Sbx * Sbx + Sby * Sby) / Swp
    den = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
    ok = (cnt / float(block * block) >= min_blocks) & (den > 1e-9)
    return np.where(ok, num / np.maximum(den, 1e-12), -np.inf)


def ncc_map(a, b, max_dx, max_dy, min_overlap, fa=None, fb=None):
    """NCC for every shift in the window (same grid/order as orient_map)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape
    ph, pw = h + 2 * max_dy + 1, w + 2 * max_dx + 1
    if fa is None:
        fa = np.fft.rfft2(a, s=(ph, pw))
    if fb is None:
        fb = np.conj(np.fft.rfft2(b, s=(ph, pw)))
    corr = np.fft.irfft2(fa * fb, s=(ph, pw))
    ia = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ia2 = np.pad(a * a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ib = np.pad(b, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ib2 = np.pad(b * b, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    dy = np.arange(-max_dy, max_dy + 1)[:, None]
    dx = np.arange(-max_dx, max_dx + 1)[None, :]
    y0 = np.broadcast_to(np.maximum(0, dy), (dy.size, dx.size))
    y1 = np.broadcast_to(np.minimum(h, h + dy), (dy.size, dx.size))
    x0 = np.broadcast_to(np.maximum(0, dx), (dy.size, dx.size))
    x1 = np.broadcast_to(np.minimum(w, w + dx), (dy.size, dx.size))
    n = (y1 - y0) * (x1 - x0)

    def rect(ii, ry0, ry1, rx0, rx1):
        return ii[ry1, rx1] - ii[ry0, rx1] - ii[ry1, rx0] + ii[ry0, rx0]

    sa = rect(ia, y0, y1, x0, x1)
    sa2 = rect(ia2, y0, y1, x0, x1)
    sb = rect(ib, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
    sb2 = rect(ib2, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
    nn = np.maximum(n, 1)
    num = corr[dy % ph, dx % pw] - sa * sb / nn
    va = sa2 - sa * sa / nn
    vb = sb2 - sb * sb / nn
    den = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    ok = (n >= min_overlap) & (den > 1e-9)
    return np.where(ok, num / np.maximum(den, 1e-12), -np.inf)


# ------------------------------------------------------------- evaluation --

def _stats(gen, imp):
    g, i = np.asarray(gen, float), np.asarray(imp, float)
    return dprime(g, i), eer(g, i)[0], far_at_frr(g, i, 0.10)[0]


def eval_pooled(M, gen_idx, imp_idx, n_enroll=6, trials=16, seed=1234):
    """Scenario A.  Random template subsets, leave-one-out over the remaining
    genuine images.  Metrics are computed per trial and averaged; a fixed
    first-N template ordering is a selection effect."""
    rng = np.random.default_rng(seed)
    gen_idx = np.asarray(gen_idx)
    imp_idx = np.asarray(imp_idx)
    out = []
    for _ in range(trials):
        T = rng.choice(gen_idx, size=n_enroll, replace=False)
        Tset = set(T.tolist())
        probes = [p for p in gen_idx.tolist() if p not in Tset]
        gen = [M[T, p].max() for p in probes]
        imp = [M[T, q].max() for q in imp_idx.tolist()]
        out.append(_stats(gen, imp))
    a = np.array(out)
    return a.mean(0), a.std(0)


def eval_realistic(M, tmpl_idx, probe_idx, imp_idx):
    """Scenario B.  Enrol on the coverage set, verify with habitual presses."""
    T = np.asarray(tmpl_idx)
    gen = [M[T, p].max() for p in probe_idx]
    imp = [M[T, q].max() for q in imp_idx]
    return _stats(gen, imp), gen, imp
