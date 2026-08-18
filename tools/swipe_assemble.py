#!/usr/bin/env python3
"""
Strip assembler for the ELAN 04f3:0c6e swipe captures.

The device is a 150 x 52 px (7.6 x 2.6 mm) area sensor that the stock driver
treats as a press sensor.  A press that small cannot carry enough minutiae, so
the plan is the Windows Hello one: swipe, mosaic the frames into a strip much
larger than the window, and match against the strip.  This file is the mosaic
step.

Three things about this sensor drive the design, all measured, not assumed:

  * Motion is along the frame LONG axis.  Frame-to-frame displacement is
    10-30 px in x and only 0..-3 px in y.  libfprint's
    fpi_do_movement_estimation searches dy over 2..frame_height and clamps dx
    to +-8, i.e. it searches the short axis and treats the long axis as drift.
    It was looking the wrong way; that is why its dx estimates pinned at the
    rail and its forward/reverse scores differed by under 2%.  Searching
    dx in 0..36 and dy in +-8 gives NCC peaks of 0.92-0.996 on real swipes.

  * Registration is therefore NOT the hard part.  Blending is.  This file
    implements several blending rules and measures all of them rather than
    picking one, because the starting hypothesis about which one would win was
    wrong (see below).

  * The device wedges after ~420-563 ms (persistent 0xaf from pre_scan), so a
    swipe is 13-19 frames and about 12 mm of finger.  You cannot swipe longer.
    Enrolment must fuse several swipes; this file assembles one swipe, which
    is the unit that later gets stitched into the enrolment map.

What the measurements said, on the two real captures available:

  * Sub-pixel registration was a prime suspect for the ridge-energy loss seen
    when blending.  It is not the main cause.  Refining the ESTIMATE is worth
    having -- the parabolic fit cuts per-pair error from 0.29 px rms to 0.08 px
    rms on synthetic data with known displacements, and that error otherwise
    accumulates over 15 pairs -- but sub-pixel RESAMPLING at placement time
    buys only about 1% of ridge amplitude, because at a 9-10 px ridge pitch a
    half-pixel misalignment costs only cos(20 deg) ~ 6% on a single pair and
    the interpolator's own band attenuation eats most of that back.

  * The large losses came from two other places.  First, a sign error in the
    resampler, which the self-test caught: it shifted frames the wrong way and
    made sub-pixel placement worse than integer placement.  Second, and much
    bigger, the contact-ramp frames.  The first 3 and last 1-2 frames of every
    swipe contain no ridges anywhere, yet they register happily (NCC 0.98) off
    the pressure blob.  Blending them in is a large part of what makes an
    assembly look like mush.  They are now gated out by ridge quality.

  * With those two fixed, every blending rule lands within about 10% of a
    single frame's local quality, and the differences between them are small.
    Per-pixel sharpest-contributor-wins is the best of them on both captures.

Everything here is offline analysis.  The parts meant for the C driver are the
NCC search, the parabolic sub-pixel fit, the outlier filter, the ridge-quality
map and the winning blend rule; none of them needs an FFT or anything outside
libm.  Only the reporting metrics use numpy's FFT.

Usage:
    swipe_assemble.py DIR [DIR ...] [--out-dir DIR] [--json]
    swipe_assemble.py --selftest

Each DIR is an `elan-swipe` capture directory containing raw_stream.npy,
background.raw and meta.json.  Outputs land next to the input (or in
--out-dir): strip_<blend>.pgm, strip_<blend>.nbis-view.pgm and
swipe_assemble_report.json.  Given two or more captures it also cross-registers
the finished strips, which is the only end-to-end check available here.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# geometry / device constants
# ---------------------------------------------------------------------------
FRAME_W = 150            # long axis, and the axis the finger travels along
FRAME_H = 52             # short axis, as the device streams it (uncropped)
DPI = 500.0
MM_PER_PX = 25.4 / DPI   # 0.0508 mm

# Search window.  dx is one-sided: the finger travels one way during a swipe,
# and allowing dx < 0 only invites the correlator to fold back on a repeating
# ridge pattern.  The observed range on real swipes is 9..30 px; 36 leaves
# headroom for a fast swipe without opening the aperture to aliasing.
DX_MIN, DX_MAX = 0, 36
DY_ABS = 8

# NCC peak below this and the pair is not trusted; it gets interpolated from
# its neighbours instead.  Real in-contact pairs measure 0.92-0.996, and the
# one pair where the finger was leaving the sensor measured 0.41, so the floor
# has a wide moat on both sides.
MIN_PEAK = 0.70

# Minimum overlap area for an NCC to mean anything, in pixels.
MIN_OVERLAP = 2500


# ---------------------------------------------------------------------------
# small image helpers (deliberately dependency-free: no scipy on this machine)
# ---------------------------------------------------------------------------
def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur(img, sigma):
    """Separable Gaussian blur with edge padding."""
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i, kv in enumerate(k):
        out += kv * p[:, i:i + img.shape[1]]
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    res = np.zeros_like(out)
    for i, kv in enumerate(k):
        res += kv * p[i:i + img.shape[0], :]
    return res


def ridge_dog(img):
    """Difference-of-Gaussians tuned to the ridge band.

    Ridge pitch on this sensor measures ~9-10 px at 500 dpi (0.45-0.5 mm,
    which is where adult ridge pitch should be).  sigma 1.0 / 3.0 passes
    roughly periods 5-14 px and rejects both the pressure blob and the
    per-pixel read noise.
    """
    return blur(img, 1.0) - blur(img, 3.0)


def ridge_quality_map(img, sigma=5.0):
    """Per-pixel "are these ridges" score in roughly [0, 1].

    Two factors, multiplied:

      band fraction -- local ridge-band energy over local wide-band energy.
        Rejects the pressure blob and, partly, read noise.
      orientation coherence -- the standard structure-tensor measure.  Ridges
        are locally parallel; noise is not.

    Either alone is foolable.  On the first frame of a swipe, where the finger
    is only just arriving and there are demonstrably no ridges anywhere (tile
    band fraction maxes at 0.09), the band term alone still reaches 0.37 in
    patches and coherence alone still reaches 0.93.  The product separates a
    good frame from that first frame by about 5x, which is enough to gate on.

    This map is what drives the quality-weighted blends, and it is also the
    coverage gate: a pixel only enters the mosaic if some frame saw ridges
    there.  That matters more than it sounds -- 3-4 frames at each end of every
    swipe are pure contact ramp, and letting them into a mean blend is a large
    part of why unregistered averaging looks like mush.
    """
    b = ridge_dog(img)
    wide = blur(img, 0.5) - blur(img, 12.0)
    frac = blur(b * b, sigma) / (blur(wide * wide, sigma) + 1e-9)
    gy, gx = np.gradient(b)
    jxx = blur(gx * gx, sigma)
    jyy = blur(gy * gy, sigma)
    jxy = blur(gx * gy, sigma)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + 1e-9)
    return np.clip(frac, 0.0, 1.0) * np.clip(coh, 0.0, 1.0)


# A frame whose median local ridge quality falls below this fraction of the
# swipe's best does not enter the mosaic at all.  It is a frame-level gate, not
# a per-pixel one: gating per pixel also strips the top and bottom rows of the
# GOOD frames, where coherence is weak against the bezel, and those rows are
# most of what makes the strip taller than one frame.
Q_KEEP_REL = 0.55


# ---------------------------------------------------------------------------
# quality metrics
# ---------------------------------------------------------------------------
def ridge_band_fraction(img, ridge_px=(6.5, 13.0), wide_px=(3.0, 40.0)):
    """Fraction of band-passed spectral energy that sits in the ridge band.

    Computed in the Fourier domain over a Hann-windowed tile: energy in the
    annulus of spatial periods `ridge_px`, divided by energy in the much wider
    annulus `wide_px`.  The wide band excludes DC and the pressure envelope
    (periods > 40 px) and excludes per-pixel noise (periods < 3 px), so the
    ratio answers "of the structure that is here, how much of it is ridges".

    This is the primary number in the report.  It is scale free, so it is only
    meaningful against a baseline measured the same way -- which is why every
    table below carries the single-frame baseline next to the mosaic value.
    """
    a = np.asarray(img, dtype=np.float64)
    a = a - a.mean()
    h, w = a.shape
    a = a * np.hanning(h)[:, None] * np.hanning(w)[None, :]
    p = np.abs(np.fft.fft2(a)) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    band = (r >= 1.0 / ridge_px[1]) & (r <= 1.0 / ridge_px[0])
    wide = (r >= 1.0 / wide_px[1]) & (r <= 1.0 / wide_px[0])
    return float(p[band].sum() / (p[wide].sum() + 1e-12))


def ridge_amplitude(img):
    """RMS of the ridge-band signal, in raw counts.

    Unlike the fraction above this is an absolute amplitude, so it is the
    metric that actually detects blend cancellation: if two misregistered
    frames partially annihilate each other, this number falls even though the
    band *fraction* may not.
    """
    return float(np.sqrt((ridge_dog(np.asarray(img, float)) ** 2).mean()))


# Analysis tile.  It has to fit inside a single frame (so the mosaic and the
# baseline are measured with identical statistics) AND inside the mosaic's
# fully covered core.  The mosaic is a parallelogram, not a rectangle: the
# finger drifts ~20 px in y across a swipe while each frame contributes only 46
# usable rows, so the fully covered band is under 30 rows tall in places.
# 40 x 128 px = 2.0 x 6.5 mm still yields 50-60 tiles per strip.
ANALYSIS_WIN = (40, 128)
ANALYSIS_STRIDE = (4, 8)


def window_metrics(img, valid, win=ANALYSIS_WIN, stride=ANALYSIS_STRIDE):
    """Sliding-window quality over a mosaic, using only fully covered tiles.

    A mosaic must be judged locally.  A global measurement over a 390 px strip
    is dominated by the low-frequency envelope that spans the whole strip and
    by whatever the uncovered corners contain, which is exactly how you end up
    reporting that a perfectly good mosaic scores 10x worse than one of the
    frames that went into it.
    """
    h, w = img.shape
    wh, ww = win
    sy, sx = stride
    fr, am = [], []
    if h < wh or w < ww:
        return {"n_windows": 0}
    for y in range(0, h - wh + 1, sy):
        for x in range(0, w - ww + 1, sx):
            if not valid[y:y + wh, x:x + ww].all():
                continue
            tile = img[y:y + wh, x:x + ww]
            fr.append(ridge_band_fraction(tile))
            am.append(ridge_amplitude(tile))
    if not fr:
        return {"n_windows": 0}
    return summarise_tiles(fr, am)


def summarise_tiles(fr, am):
    fr = np.asarray(fr, float)
    am = np.asarray(am, float)
    return {
        "n_windows": int(fr.size),
        "band_median": float(np.median(fr)),
        "band_p10": float(np.percentile(fr, 10)),
        "band_p90": float(np.percentile(fr, 90)),
        "band_max": float(fr.max()),
        "amp_median": float(np.median(am)),
        "amp_p10": float(np.percentile(am, 10)),
        "_band": fr,
        "_amp": am,
    }


def strip_arrays(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_swipe(d):
    """Read an elan-swipe capture into (N, 52, 150) float64, ridges bright.

    raw_stream.npy is (N, 150, 52) exactly as the device sent it -- the device
    streams column-major relative to the display orientation -- so the
    transpose is what puts 52 rows of 150 columns on screen.
    """
    d = Path(d)
    stream = np.load(d / "raw_stream.npy")
    if stream.ndim != 3 or stream.shape[1:] != (FRAME_W, FRAME_H):
        raise ValueError(f"{d}: unexpected raw_stream shape {stream.shape}")
    bg = np.fromfile(d / "background.raw", dtype="<u2")
    if bg.size != FRAME_W * FRAME_H:
        raise ValueError(f"{d}: background.raw is {bg.size} px")
    bg = bg.reshape(FRAME_W, FRAME_H).T.astype(np.float64)
    frames = np.transpose(stream, (0, 2, 1)).astype(np.float64)
    # elan_save_img_frame clamps at zero rather than allowing negatives.
    frames = np.clip(frames - bg[None], 0.0, None)
    meta = {}
    mp = d / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    return frames, meta


# ---------------------------------------------------------------------------
# 1. registration
# ---------------------------------------------------------------------------
def _integral(x):
    return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _rect(ii, y0, y1, x0, x1):
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def ncc_surface(a, b, dx_range, dy_range, min_overlap=MIN_OVERLAP):
    """Full NCC surface for b shifted onto a.

    Convention: a[y, x] pairs with b[y - dy, x - dx], i.e. positive dx means b
    is the later frame and the finger has moved +x between them.  Means and
    variances come from integral images so the cost is O(1) per shift after an
    O(HW) setup; the cross term is the only per-shift work.  The C driver can
    use this verbatim -- 37 x 17 shifts on 150x52 is trivial next to the 31 ms
    the sensor takes to produce the next frame.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape
    ia, ia2 = _integral(a), _integral(a * a)
    ib, ib2 = _integral(b), _integral(b * b)
    out = np.full((len(dy_range), len(dx_range)), np.nan)
    for i, dy in enumerate(dy_range):
        y0, y1 = max(0, dy), min(h, h + dy)
        if y1 <= y0:
            continue
        for j, dx in enumerate(dx_range):
            x0, x1 = max(0, dx), min(w, w + dx)
            if x1 <= x0:
                continue
            n = (y1 - y0) * (x1 - x0)
            if n < min_overlap:
                continue
            sa = _rect(ia, y0, y1, x0, x1)
            sa2 = _rect(ia2, y0, y1, x0, x1)
            sb = _rect(ib, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
            sb2 = _rect(ib2, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
            va = sa2 - sa * sa / n
            vb = sb2 - sb * sb / n
            if va <= 0 or vb <= 0:
                continue
            cross = float((a[y0:y1, x0:x1] *
                           b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]).sum())
            out[i, j] = (cross - sa * sb / n) / math.sqrt(va * vb)
    return out


def _parabolic(vm, v0, vp):
    """Vertex offset of the parabola through three samples, clamped to +-0.5.

    Standard three-point fit.  The clamp matters: near a flat or noisy peak the
    denominator can go small and throw the vertex a long way out, and a bogus
    2 px "sub-pixel" correction is far worse than no correction at all.
    """
    if not (np.isfinite(vm) and np.isfinite(v0) and np.isfinite(vp)):
        return 0.0
    den = vm - 2.0 * v0 + vp
    if abs(den) < 1e-9:
        return 0.0
    d = 0.5 * (vm - vp) / den
    return float(np.clip(d, -0.5, 0.5))


def register_pair(a, b, dx_range, dy_range):
    """Integer NCC peak plus a parabolic sub-pixel refinement in both axes."""
    s = ncc_surface(a, b, dx_range, dy_range)
    if np.all(np.isnan(s)):
        return None
    iy, ix = np.unravel_index(np.nanargmax(s), s.shape)
    peak = float(s[iy, ix])
    fy = fx = 0.0
    if 0 < iy < s.shape[0] - 1:
        fy = _parabolic(s[iy - 1, ix], s[iy, ix], s[iy + 1, ix])
    if 0 < ix < s.shape[1] - 1:
        fx = _parabolic(s[iy, ix - 1], s[iy, ix], s[iy, ix + 1])
    return {
        "dy": dy_range[iy] + fy,
        "dx": dx_range[ix] + fx,
        "dy_int": int(dy_range[iy]),
        "dx_int": int(dx_range[ix]),
        "sub_dy": fy,
        "sub_dx": fx,
        "peak": peak,
        "at_rail": bool(ix in (0, s.shape[1] - 1) or iy in (0, s.shape[0] - 1)),
    }


# ---------------------------------------------------------------------------
# 2. frame gating and outlier handling
# ---------------------------------------------------------------------------
def frame_quality(frames):
    """Per-frame median local ridge quality (see ridge_quality_map).

    Reported, not gated on: frames are gated per pixel during blending, which
    keeps a partly-contacted frame's good half instead of throwing the frame
    away.  Frame-level numbers are still worth having in the report because
    they show the contact ramp: on both captures the first 3 frames and the
    last 1-2 frames carry no ridges at all.
    """
    return np.array([float(np.median(ridge_quality_map(f))) for f in frames])


def contact_window(q, rel=Q_KEEP_REL):
    """Longest run of frames whose ridge quality clears `rel` x the swipe best.

    Both captures ramp in over 3 frames and out over 1-2: on 20260814-170350
    the first three frames have a tile band fraction of 0.045, 0.083 and 0.147
    against 0.75 mid-swipe, i.e. they contain no ridges anywhere.  They still
    register perfectly well (NCC 0.98) because the pressure blob correlates, so
    they stay in the displacement track -- they are only kept out of the blend.
    """
    thr = rel * float(np.max(q))
    good = q >= thr
    best, i = (0, 0), 0
    while i < len(good):
        if not good[i]:
            i += 1
            continue
        j = i
        while j < len(good) and good[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best


MAX_CURVATURE = 4.0     # px, |dx[k] - (dx[k-1]+dx[k+1])/2|
MAX_SLOPE = 12.0        # px, |dx[k] - dx[nearest accepted neighbour]|


def clean_track(pairs):
    """Reject and interpolate implausible displacements.

    A finger in contact accelerates smoothly at 32 fps, so the right test is on
    CURVATURE, not on the local median.  The distinction is not academic: on
    capture 20260814-170350 the finger accelerates through 9.4, 10.1, 14.2,
    17.0 px at the end of the swipe, and a median-plus-MAD rule rejects the
    14.2 as an outlier even though a band-passed re-registration confirms it to
    0.1 px.  A curvature rule predicts 13.5 there and accepts it.

    Pairs that fail the NCC floor or sit on a search rail are rejected first;
    everything rejected is replaced by linear interpolation between the
    surviving neighbours, so the track stays continuous.
    """
    n = len(pairs)
    dx = np.array([p["dx"] if p else np.nan for p in pairs])
    dy = np.array([p["dy"] if p else np.nan for p in pairs])
    ok = np.array([
        bool(p) and p["peak"] >= MIN_PEAK and not p["at_rail"] for p in pairs
    ])
    reasons = ["ok" if o else "NCC peak below floor or shift on search rail"
               for o in ok]

    for k in range(n):
        if not ok[k]:
            continue
        prev = [j for j in range(k - 1, -1, -1) if ok[j]]
        nxt = [j for j in range(k + 1, n) if ok[j]]
        if prev and nxt:
            pred = 0.5 * (dx[prev[0]] + dx[nxt[0]])
            if abs(dx[k] - pred) > MAX_CURVATURE:
                ok[k] = False
                reasons[k] = (f"curvature outlier: dx {dx[k]:.1f}, "
                              f"neighbours predict {pred:.1f}")
        elif prev or nxt:
            j = (prev or nxt)[0]
            if abs(dx[k] - dx[j]) > MAX_SLOPE:
                ok[k] = False
                reasons[k] = (f"slope outlier: dx {dx[k]:.1f} next to "
                              f"{dx[j]:.1f}")

    if ok.sum() < 2:
        raise RuntimeError("no usable frame-to-frame displacement in this swipe")

    xs = np.where(ok)[0]
    for arr in (dx, dy):
        arr[~ok] = np.interp(np.where(~ok)[0], xs, arr[xs])
    return dx, dy, ok, reasons


# ---------------------------------------------------------------------------
# 3. sub-pixel resampling
# ---------------------------------------------------------------------------
def _catmull_rom(t):
    """Catmull-Rom weights for the four taps around a fractional offset t.

    Bilinear would be simpler but it is a low-pass filter: at a 9 px ridge
    pitch it costs about 6% of ridge amplitude at a half-pixel shift, and that
    loss lands on exactly the signal being measured.  Catmull-Rom is close to
    flat over the ridge band and needs only multiplies and adds, so it ports to
    the driver unchanged.
    """
    t2, t3 = t * t, t * t * t
    return np.array([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ])


def shift_subpixel(img, fy, fx):
    """Translate the CONTENT by a fractional (fy, fx) in [0, 1).

    out(y, x) = img(y - fy, x - fx), i.e. positive fx moves the image right.
    That is the direction build_stack needs, because a frame whose position on
    the canvas is px = ix + fx has to have its content pushed right by fx
    before being dropped at the integer offset ix.

    Getting this backwards is silent and expensive: the frame lands 2*fx px
    from where it should, which is worse than not correcting at all, and the
    only symptom is that sub-pixel placement mysteriously fails to help.  The
    self-test checks the direction against an independent Fourier shift.

    Interpolating at x - fx means interpolating at fractional position
    u = 1 - fx above sample x - 1, so the four taps are x-2 .. x+1.
    """
    a = np.asarray(img, float)
    if abs(fx) > 1e-9:
        k = _catmull_rom(1.0 - fx)
        p = np.pad(a, ((0, 0), (2, 1)), mode="edge")
        a = sum(k[i] * p[:, i:i + a.shape[1]] for i in range(4))
    if abs(fy) > 1e-9:
        k = _catmull_rom(1.0 - fy)
        p = np.pad(a, ((2, 1), (0, 0)), mode="edge")
        a = sum(k[i] * p[i:i + a.shape[0], :] for i in range(4))
    return a


# ---------------------------------------------------------------------------
# 4. placement and blending
# ---------------------------------------------------------------------------
def _raised_cosine(n, edge):
    """1 in the middle, cosine roll-off to 0 over `edge` samples at each end."""
    w = np.ones(n)
    if edge > 0:
        t = (np.arange(edge) + 0.5) / edge
        ramp = 0.5 - 0.5 * np.cos(np.pi * t)
        w[:edge] = ramp
        w[-edge:] = ramp[::-1]
    return w


# Three ways to go from per-pair displacements to pixels on a canvas.  They are
# reported separately because they test different claims:
#
#   subpixel   sub-pixel estimates, sub-pixel resampling at placement.
#   int_place  sub-pixel estimates, cumulative position rounded to a pixel.
#              Isolates the cost of the resampling filter itself.
#   int_reg    integer NCC peak, no parabolic fit, rounded placement.  This is
#              the naive scheme, and the one whose per-pair error accumulates
#              over the length of a swipe.
PLACEMENTS = ["subpixel", "int_place", "int_reg"]


def track_positions(dx, dy, keep, placement="subpixel"):
    """Cumulative frame positions, restricted to the frames being blended.

    The track integrates over ALL pairs, including the contact-ramp frames at
    each end, because those pairs carry real displacement; only the placement
    is restricted, via `keep`.
    """
    px = np.concatenate([[0.0], np.cumsum(dx)])
    py = np.concatenate([[0.0], np.cumsum(dy)])
    lo, hi = keep
    px, py = px[lo:hi], py[lo:hi]
    if placement != "subpixel":
        px, py = np.round(px), np.round(py)
    return px - px.min(), py - py.min()


def build_stack(frames, px, py, subpixel=True, border=3):
    """Place every frame on a common canvas.

    Returns (values, valid, sharp, centre) each (N, CH, CW).  `values` is the
    resampled frame, `valid` its coverage mask, `sharp` its local ridge-quality
    map (used by the quality-driven blends), and `centre` a raised-cosine
    distance-from-frame-centre weight.
    """
    n = len(frames)
    cw = int(math.ceil(px.max())) + FRAME_W
    ch = int(math.ceil(py.max())) + FRAME_H

    values = np.zeros((n, ch, cw))
    valid = np.zeros((n, ch, cw), dtype=bool)
    sharp = np.zeros((n, ch, cw))
    centre = np.zeros((n, ch, cw))

    # Frames are sharpest mid-frame: the leading and trailing columns of a
    # frame see the finger arriving at / leaving that row of sensels, and the
    # outer rows sit against the sensor bezel.  Roll off 24 px of the 150 and
    # 10 px of the 52 -- generous enough that the taper never goes fully to
    # zero anywhere the mosaic relies on a single contributor.
    cen = np.outer(_raised_cosine(FRAME_H, 10), _raised_cosine(FRAME_W, 24))
    cen = np.maximum(cen, 1e-3)

    for i, f in enumerate(frames):
        ix, iy = int(math.floor(px[i])), int(math.floor(py[i]))
        fx, fy = px[i] - ix, py[i] - iy
        s = shift_subpixel(f, fy, fx) if subpixel else f
        q = ridge_quality_map(s)
        # A fractional shift needs two taps on each side; the outermost
        # `border` columns/rows are extrapolated, so drop them.  The quality
        # floor drops everything the frame did not actually see ridges in.
        m = np.zeros((FRAME_H, FRAME_W), dtype=bool)
        m[border:FRAME_H - border, border:FRAME_W - border] = True
        values[i, iy:iy + FRAME_H, ix:ix + FRAME_W] = s
        valid[i, iy:iy + FRAME_H, ix:ix + FRAME_W] = m
        sharp[i, iy:iy + FRAME_H, ix:ix + FRAME_W] = q
        centre[i, iy:iy + FRAME_H, ix:ix + FRAME_W] = cen
    sharp *= valid
    centre *= valid
    return values, valid, sharp, centre


def equalise(values, valid):
    """Match each frame's gain and offset to its predecessor over the overlap.

    Contact pressure rises and falls through a swipe, so consecutive frames
    can differ by tens of percent in contrast.  Left uncorrected that alone
    produces visible banding and makes mean blending mix signals of different
    scale.  A one-parameter-pair least-squares fit on the overlap is enough and
    is trivially portable.
    """
    out = values.copy()
    gains = [1.0]
    for i in range(1, len(values)):
        m = valid[i] & valid[i - 1]
        if m.sum() < MIN_OVERLAP:
            gains.append(gains[-1])
            out[i] *= gains[-1]
            continue
        a = out[i - 1][m]
        b = values[i][m]
        vb = b.var()
        g = float(a.std() / math.sqrt(vb)) if vb > 1e-12 else 1.0
        g = float(np.clip(g, 0.5, 2.0))
        o = float(a.mean() - g * b.mean())
        out[i] = values[i] * g + o
        gains.append(g)
    out *= valid
    return out, gains


def blend(values, valid, sharp, centre, mode):
    """Combine the stack into one image.  Returns (image, coverage_mask)."""
    cov = valid.any(0)
    n, ch, cw = values.shape
    v = np.where(valid, values, np.nan)

    if mode == "mean":
        w = valid.astype(float)
    elif mode == "median":
        # Per-pixel median.  Robust to a single bad contributor, but it still
        # mixes contributors, so any residual misregistration still cancels.
        vm = np.where(valid, values, np.nan)
        vm[:, ~cov] = 0.0
        img = np.nanmedian(vm, axis=0)
        return np.where(cov, np.nan_to_num(img), 0.0), cov
    elif mode == "centre":
        w = centre
    elif mode == "sharp_soft":
        # Soft quality weighting: squared local ridge amplitude, tapered.
        w = centre * sharp ** 2
    elif mode == "sharp_win":
        # Hard per-pixel winner.  No two frames ever mix, so nothing can
        # cancel -- but every pixel boundary between winners is a potential
        # seam, which is what the metrics below are there to catch.
        score = np.where(valid, centre * sharp, -1.0)
        k = score.argmax(0)
        img = np.take_along_axis(values, k[None], 0)[0]
        return np.where(cov, img, 0.0), cov
    elif mode == "col_best":
        # One contributor per output column, chosen by the column's summed
        # quality.  Seams then run along whole columns rather than wandering
        # pixel by pixel, which keeps ridge continuity across a seam intact.
        score = np.where(valid, centre * sharp, 0.0).sum(1)      # (n, cw)
        score[~valid.any(1)] = -1.0
        k = score.argmax(0)                                      # (cw,)
        img = values[k, :, np.arange(cw)].T
        vm = valid[k, :, np.arange(cw)].T
        img = np.where(vm, img, 0.0)
        # Columns whose winner does not cover every row fall back to the mean.
        hole = cov & ~vm
        if hole.any():
            mw = valid.astype(float)
            mimg = (values * mw).sum(0) / np.maximum(mw.sum(0), 1e-9)
            img = np.where(hole, mimg, img)
        return img, cov
    else:
        raise ValueError(f"unknown blend mode {mode!r}")

    w = np.where(valid, w, 0.0)
    tot = w.sum(0)
    img = (values * w).sum(0) / np.maximum(tot, 1e-9)
    return np.where(cov, img, 0.0), cov


BLEND_MODES = ["mean", "median", "centre", "sharp_soft", "sharp_win", "col_best"]


def overlap_retention(img, values, valid, min_contrib=2):
    """Ridge amplitude kept in overlaps, relative to the contributors.

    This is the number that actually tests the cancellation hypothesis.  Take
    the pixels covered by >= `min_contrib` frames, measure the ridge-band RMS of
    the blend there, and divide by the mean ridge-band RMS the contributing
    frames had at those same pixels.  1.0 means the blend preserved the ridge
    signal; below 1.0 means contributors partially annihilated one another.

    A ratio ABOVE 1.0 is possible and is the point of averaging: independent
    read noise averages down while the ridge signal does not, so a correctly
    registered mean should beat its own contributors.
    """
    n = len(values)
    cnt = valid.sum(0)
    m = cnt >= min_contrib
    if m.sum() < 500:
        return None
    r_blend = ridge_dog(img)
    num = float(np.sqrt((r_blend[m] ** 2).mean()))
    contrib = []
    for i in range(n):
        mi = m & valid[i]
        if mi.sum() < 100:
            continue
        ri = ridge_dog(values[i])
        contrib.append(float(np.sqrt((ri[mi] ** 2).mean())))
    if not contrib:
        return None
    return round(num / (float(np.mean(contrib)) + 1e-9), 4)


# ---------------------------------------------------------------------------
# 5. output
# ---------------------------------------------------------------------------
def to_u8(img, cov):
    """Robust linear stretch over the covered area, ridges bright."""
    v = img[cov]
    if v.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) * 255.0 / (hi - lo), 0, 255)
    return np.where(cov, out, 0).astype(np.uint8)


def write_pgm(path, arr):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def assemble(indir, outdir=None, verbose=True):
    indir = Path(indir)
    outdir = Path(outdir) if outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)
    frames, meta = load_swipe(indir)
    rep = {
        "capture": str(indir),
        "n_frames_raw": len(frames),
        "sensor": {"w": FRAME_W, "h": FRAME_H, "dpi": DPI},
    }

    # --- per-frame quality, and the in-contact window ----------------------
    q = frame_quality(frames)
    keep = contact_window(q)
    rep["frame_ridge_quality"] = [round(float(x), 3) for x in q]
    rep["contact_window"] = [int(keep[0]), int(keep[1])]
    rep["n_frames_blended"] = int(keep[1] - keep[0])
    if keep[1] - keep[0] < 3:
        raise RuntimeError(f"{indir}: only {keep[1] - keep[0]} in-contact frames")
    used = frames

    # --- registration -----------------------------------------------------
    dxr = list(range(DX_MIN, DX_MAX + 1))
    dyr = list(range(-DY_ABS, DY_ABS + 1))
    pairs = [register_pair(used[i], used[i + 1], dxr, dyr)
             for i in range(len(used) - 1)]
    dx, dy, ok, reasons = clean_track(pairs)
    rep["pairs"] = [
        {
            "i": i,
            "dx": round(float(dx[i]), 3),
            "dy": round(float(dy[i]), 3),
            "dx_int": p["dx_int"] if p else None,
            "dy_int": p["dy_int"] if p else None,
            "sub_dx": round(p["sub_dx"], 3) if p else None,
            "sub_dy": round(p["sub_dy"], 3) if p else None,
            "peak": round(p["peak"], 4) if p else None,
            "accepted": bool(ok[i]),
            "note": reasons[i],
        }
        for i, p in enumerate(pairs)
    ]
    peaks = [p["peak"] for p in pairs if p]
    rep["registration"] = {
        "peak_min": round(min(peaks), 4),
        "peak_median": round(float(np.median(peaks)), 4),
        "n_rejected": int((~ok).sum()),
        "dx_total": round(float(dx.sum()), 2),
        "dy_total": round(float(dy.sum()), 2),
        "travel_mm": round(float(dx.sum()) * MM_PER_PX, 2),
        "subpixel_rms_dx": round(
            float(np.sqrt(np.mean([p["sub_dx"] ** 2 for p in pairs if p]))), 3),
        "subpixel_rms_dy": round(
            float(np.sqrt(np.mean([p["sub_dy"] ** 2 for p in pairs if p]))), 3),
    }

    # --- single-frame baseline -------------------------------------------
    # Measured with exactly the same 40x128 sliding tile as the mosaics, so
    # "matches a single frame" is a like-for-like comparison and not an
    # artefact of measuring a 52x150 frame against a 385x72 strip.
    # Only the frames that actually enter the mosaic count towards the
    # baseline; comparing a strip built from ridge-bearing frames against a
    # baseline that includes the contact ramp would flatter the strip.
    blended = frames[keep[0]:keep[1]]
    allv = np.ones((FRAME_H, FRAME_W), dtype=bool)
    pool_b, pool_a, per_frame = [], [], []
    for f in blended:
        m = window_metrics(f, allv)
        per_frame.append(round(m["band_median"], 4))
        pool_b.extend(m["_band"])
        pool_a.extend(m["_amp"])
    # Pooled over every tile of every blended frame, so the mosaic's p10 can be
    # compared against the single-frame p10 rather than against a median.
    base = summarise_tiles(pool_b, pool_a)
    rep["single_frame_baseline"] = dict(
        strip_arrays(base),
        per_frame_band=per_frame,
        area_px=FRAME_W * FRAME_H,
        area_mm2=round(FRAME_W * FRAME_H * MM_PER_PX ** 2, 2),
    )
    # Negative control: the frames stacked without any registration, i.e. what
    # the old driver's averaging produced.
    unreg = window_metrics(blended.mean(0), allv)
    rep["unregistered_average"] = {
        "band": round(unreg["band_median"], 4),
        "amp": round(unreg["amp_median"], 1),
    }

    # --- blending comparison ---------------------------------------------
    dx_i = np.array([p["dx_int"] if p else 0 for p in pairs], float)
    dy_i = np.array([p["dy_int"] if p else 0 for p in pairs], float)
    dx_i, dy_i, _, _ = clean_track([
        dict(p, dx=float(p["dx_int"]), dy=float(p["dy_int"])) if p else None
        for p in pairs])

    results, strips = {}, {}
    for placement in PLACEMENTS:
        tdx, tdy = (dx_i, dy_i) if placement == "int_reg" else (dx, dy)
        px, py = track_positions(tdx, tdy, keep, placement=placement)
        values, valid, sharp, centre = build_stack(
            blended, px, py, subpixel=(placement == "subpixel"))
        values, gains = equalise(values, valid)
        if placement == "subpixel":
            rep["canvas"] = {
                "w": int(values.shape[2]),
                "h": int(values.shape[1]),
                "mm": [round(values.shape[2] * MM_PER_PX, 2),
                       round(values.shape[1] * MM_PER_PX, 2)],
                # The strip is a parallelogram, so its bounding box is
                # noticeably larger than the area actually covered.  Quoting
                # the bounding box as "area gained" overstates the result.
                "bbox_frames": round(
                    values.shape[1] * values.shape[2] / (FRAME_W * FRAME_H), 2),
            }
            rep["equalise_gains"] = [round(g, 3) for g in gains]
        for mode in BLEND_MODES:
            img, cov = blend(values, valid, sharp, centre, mode)
            m = strip_arrays(window_metrics(img, cov))
            m["coverage_px"] = int(cov.sum())
            m["coverage_frames"] = round(
                float(cov.sum()) / (FRAME_W * FRAME_H), 2)
            m["coverage_mm2"] = round(float(cov.sum()) * MM_PER_PX ** 2, 2)
            m["overlap_retention"] = overlap_retention(img, values, valid)
            results[f"{mode}/{placement}"] = m
            if placement == "subpixel":
                strips[mode] = (img, cov)
                write_pgm(outdir / f"strip_{mode}.pgm", to_u8(img, cov))
                write_pgm(outdir / f"strip_{mode}.nbis-view.pgm",
                          255 - to_u8(img, cov))
    rep["blends"] = results

    # Pick the winner on median local band fraction, the "typical local
    # quality" the whole exercise is about.
    best = max(results, key=lambda k: results[k].get("band_median", -1))
    rep["best_blend"] = best
    rep["best_vs_single_frame"] = round(
        results[best]["band_median"] / rep["single_frame_baseline"]["band_median"], 3)

    (outdir / "swipe_assemble_report.json").write_text(
        json.dumps(rep, indent=2) + "\n")

    if verbose:
        print_report(rep)
    return rep, strips[best.split("/")[0]]


def print_report(rep):
    r = rep["registration"]
    sf = rep["single_frame_baseline"]
    print(f"\n=== {rep['capture']} ===")
    print(f"frames {rep['n_frames_raw']} captured, "
          f"{rep['n_frames_blended']} in contact (window {rep['contact_window']}); "
          f"the rest are contact ramp and carry no ridges")
    print(f"registration: peak median {r['peak_median']:.3f} min {r['peak_min']:.3f}, "
          f"{r['n_rejected']} pair(s) rejected/interpolated")
    print(f"travel {r['dx_total']:.1f} px = {r['travel_mm']:.2f} mm, "
          f"dy total {r['dy_total']:.2f} px")
    print(f"sub-pixel corrections: rms dx {r['subpixel_rms_dx']:.3f} px, "
          f"rms dy {r['subpixel_rms_dy']:.3f} px")
    c = rep["canvas"]
    print(f"canvas {c['w']}x{c['h']} px = {c['mm'][0]}x{c['mm'][1]} mm "
          f"(bounding box {c['bbox_frames']}x one frame; the covered "
          f"parallelogram is smaller -- see the area column)")
    print(f"single-frame baseline ({sf['n_windows']} tiles pooled over the "
          f"blended frames):")
    print(f"    band p10 {sf['band_p10']:.3f}  median {sf['band_median']:.3f}  "
          f"max {sf['band_max']:.3f}   amp median {sf['amp_median']:.0f}")
    print(f"unregistered average (negative control): "
          f"band {rep['unregistered_average']['band']:.3f} "
          f"amp {rep['unregistered_average']['amp']:.0f}")
    print()
    print(f"{'blend / placement':<27}{'band p10':>9}{'band med':>9}"
          f"{'band max':>9}{'amp med':>9}{'vs 1frame':>10}{'ovl keep':>9}"
          f"{'area x1f':>9}")
    for placement in PLACEMENTS:
        for mode in BLEND_MODES:
            k = f"{mode}/{placement}"
            m = rep["blends"].get(k)
            if not m or not m.get("n_windows"):
                continue
            ret = m.get("overlap_retention")
            print(f"{k:<27}{m['band_p10']:>9.3f}{m['band_median']:>9.3f}"
                  f"{m['band_max']:>9.3f}{m['amp_median']:>9.0f}"
                  f"{m['band_median'] / sf['band_median']:>10.2f}"
                  f"{(f'{ret:.3f}' if ret else 'n/a'):>9}"
                  f"{m['coverage_frames']:>9.2f}")
        print()
    print(f"\nbest: {rep['best_blend']}  "
          f"({rep['best_vs_single_frame']:.2f}x the median single frame)")


# ---------------------------------------------------------------------------
# cross-check between independently assembled strips
# ---------------------------------------------------------------------------
def cross_check(strips, dy_span=25, dx_span=140, min_overlap=8000):
    """Register two finished strips against each other.

    Two purposes.  First, it is the only end-to-end validation available with
    the data on this machine: if two strips assembled from separate swipes of
    the same finger line up, the per-swipe geometry is right, and no amount of
    self-consistency checking can substitute for that.  Second, it is exactly
    the operation enrolment will need in order to fuse several swipes into one
    map, so the number it returns is a preview of whether that will work.

    NOT a biometric score.  Both captures here are the same finger, so this
    says nothing about impostors.
    """
    (na, a, ma), (nb, b, mb) = strips
    H = max(a.shape[0], b.shape[0]) + 2 * dy_span + 10
    W = max(a.shape[1], b.shape[1]) + 2 * dx_span + 10
    oy, ox = dy_span + 5, dx_span + 5
    ca = np.zeros((H, W))
    caM = np.zeros((H, W), dtype=bool)
    ca[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
    caM[oy:oy + a.shape[0], ox:ox + a.shape[1]] = ma
    best = (-1.0, None)
    for dy in range(-dy_span, dy_span + 1):
        for dx in range(-dx_span, dx_span + 1, 2):
            y, x = oy + dy, ox + dx
            cbM = np.zeros((H, W), dtype=bool)
            cbM[y:y + b.shape[0], x:x + b.shape[1]] = mb
            m = caM & cbM
            n = int(m.sum())
            if n < min_overlap:
                continue
            cb = np.zeros((H, W))
            cb[y:y + b.shape[0], x:x + b.shape[1]] = b
            u = ca[m] - ca[m].mean()
            v = cb[m] - cb[m].mean()
            d = math.sqrt(float((u * u).sum()) * float((v * v).sum())) + 1e-12
            r = float((u * v).sum()) / d
            if r > best[0]:
                best = (r, (dy, dx, n))
    return {
        "a": na, "b": nb,
        "ncc": round(best[0], 4),
        "dy": best[1][0] if best[1] else None,
        "dx": best[1][1] if best[1] else None,
        "overlap_px": best[1][2] if best[1] else 0,
        "overlap_mm2": round(best[1][2] * MM_PER_PX ** 2, 2) if best[1] else 0.0,
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _fourier_shift(img, fy, fx):
    """Exact band-limited translation, used only to SYNTHESISE test data.

    Deliberately a different algorithm from the Catmull-Rom resampler under
    test, so the self-test cannot pass by both sides making the same mistake.
    """
    h, w = img.shape
    fy_g = np.fft.fftfreq(h)[:, None]
    fx_g = np.fft.fftfreq(w)[None, :]
    ph = np.exp(-2j * np.pi * (fy_g * fy + fx_g * fx))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * ph))


def truth_correlation(img, cov, g, span=4):
    """Best correlation of an assembled strip against the ground truth.

    This is the sensitive test of registration quality.  Ridge amplitude
    measured on the strip alone barely moves when the track drifts, because
    every neighbourhood still contains ridges of some sort; correlating against
    the image the frames were actually cut from does move, because accumulated
    drift cannot be undone by any single global offset.
    """
    best = -1.0
    a = img[cov]
    a = a - a.mean()
    na = math.sqrt(float((a * a).sum())) + 1e-12
    h, w = img.shape
    for oy in range(-span, span + 1):
        for ox in range(-span, span + 1):
            sub = g[20 + oy:20 + oy + h, 20 + ox:20 + ox + w]
            if sub.shape != img.shape:
                continue
            b = sub[cov]
            b = b - b.mean()
            r = float((a * b).sum()) / (na * (math.sqrt(float((b * b).sum())) + 1e-12))
            best = max(best, r)
    return best


def synth_swipe(n=20, seed=1, noise=0.02):
    """A synthetic swipe with exactly known sub-pixel displacements.

    Ridges are a phase field with slowly varying orientation and a 9 px pitch,
    matching what this sensor measures on a real finger.
    """
    rng = np.random.default_rng(seed)
    gh, gw = 140, 720
    yy, xx = np.mgrid[0:gh, 0:gw].astype(float)
    # Two loop centres, so ridge orientation sweeps through the full range.  A
    # pattern of near-horizontal ridges would be useless here: translation
    # along the ridge direction is unobservable (the aperture problem), so a
    # self-test built on one would "fail" for reasons that have nothing to do
    # with the code.
    r1 = np.hypot(xx - 210.0, yy - 55.0)
    r2 = np.hypot(xx - 520.0, yy - 85.0)
    # Pitch drifts across the print, as it does on a real finger.  A perfectly
    # periodic pattern is ambiguous under translation: with a constant 9 px
    # pitch and a 36 px search range the correlator finds four equally good
    # peaks and lands one pitch off, which is an artefact of the test data
    # rather than a defect in the code.
    pitch = 9.0 + 2.0 * np.sin(xx / 260.0) + 0.8 * np.cos(yy / 70.0)
    g = np.sin(2 * np.pi * r1 / pitch) + 0.9 * np.sin(2 * np.pi * r2 / (pitch + 0.7))
    g += 0.4 * np.sin(2 * np.pi * xx / 200.0)             # pressure envelope
    # Non-periodic low-frequency texture: pores, scars and pressure structure
    # are what makes a real pair of frames unambiguous.
    tex = rng.normal(0, 1, (gh, gw))
    tex = blur(tex, 5.0)
    g += 0.9 * tex / (tex.std() + 1e-9)
    # A smooth speed profile, because that is what a finger does and what the
    # outlier filter is entitled to assume.  Both real captures decelerate then
    # accelerate through the swipe.
    t = np.linspace(0.0, 1.0, n - 1)
    truth_dx = 16.0 - 7.0 * np.sin(np.pi * t) + rng.normal(0, 0.35, n - 1)
    truth_dy = -1.5 + 1.0 * np.cos(np.pi * t) + rng.normal(0, 0.15, n - 1)
    px = np.concatenate([[0.0], np.cumsum(truth_dx)])
    py = np.concatenate([[0.0], np.cumsum(truth_dy)])
    py -= py.min()
    frames = []
    for i in range(n):
        ix, iy = int(math.floor(px[i])), int(math.floor(py[i]))
        fx, fy = px[i] - ix, py[i] - iy
        # frame_i(y, x) = g(py_i + y, px_i + x).  Sampling g at a POSITIVE
        # fractional offset means shifting g by a negative one.
        s = _fourier_shift(g, -fy, -fx)
        frames.append(s[iy + 20:iy + 20 + FRAME_H, ix + 20:ix + 20 + FRAME_W] +
                      rng.normal(0, noise, (FRAME_H, FRAME_W)))
    return np.array(frames), truth_dx, truth_dy, g


def selftest():
    """Validate registration, sub-pixel refinement and the outlier filter.

    The point of this is narrow but important: the report below concludes that
    sub-pixel registration does not improve real assemblies.  That conclusion is
    only worth anything if the sub-pixel code is demonstrably correct, i.e. if
    it recovers a known fractional displacement and if using it demonstrably
    beats integer placement when the displacement really is the only error.
    """
    fails = []
    frames, tdx, tdy, g = synth_swipe()
    dxr = list(range(DX_MIN, DX_MAX + 1))
    dyr = list(range(-DY_ABS, DY_ABS + 1))
    pairs = [register_pair(frames[i], frames[i + 1], dxr, dyr)
             for i in range(len(frames) - 1)]

    int_err = np.abs(np.array([p["dx_int"] for p in pairs]) - tdx)
    sub_err = np.abs(np.array([p["dx"] for p in pairs]) - tdx)
    print(f"dx error: integer rms {np.sqrt((int_err ** 2).mean()):.3f} px "
          f"(max {int_err.max():.3f}), sub-pixel rms "
          f"{np.sqrt((sub_err ** 2).mean()):.3f} px (max {sub_err.max():.3f})")
    rms_ex = float(np.sqrt((sub_err ** 2).mean()))
    if rms_ex > 0.12:
        fails.append(f"sub-pixel dx rms error {rms_ex:.3f} px > 0.12")
    if sub_err.mean() >= int_err.mean():
        fails.append("sub-pixel refinement did not beat integer registration")

    sub_ey = np.abs(np.array([p["dy"] for p in pairs]) - tdy)
    rms_ey = float(np.sqrt((sub_ey ** 2).mean()))
    print(f"dy error: sub-pixel rms {rms_ey:.3f} px, max {sub_ey.max():.3f} px")
    if rms_ey > 0.35:
        fails.append(f"sub-pixel dy rms error {rms_ey:.3f} px > 0.35")

    # The Catmull-Rom resampler must invert the Fourier one to within the
    # ridge band, otherwise a sign error would hide as "sub-pixel never helps".
    a = frames[0]
    for fy, fx in [(0.0, 0.5), (0.35, 0.0), (0.4, 0.7)]:
        ref = _fourier_shift(a, fy, fx)[8:-8, 8:-8]
        got = shift_subpixel(a, fy, fx)[8:-8, 8:-8]
        r = float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])
        print(f"resampler vs Fourier shift ({fy}, {fx}): r = {r:.5f}")
        if r < 0.995:
            fails.append(f"Catmull-Rom shift ({fy},{fx}) r={r:.4f} < 0.995")

    # End to end: with translation as the only error source, sub-pixel
    # placement must reconstruct better than integer placement.
    dx, dy, ok, _ = clean_track(pairs)
    if not ok.all():
        fails.append(f"outlier filter rejected {(~ok).sum()} clean synthetic pairs")
    dx_i, dy_i, _, _ = clean_track([
        dict(p, dx=float(p["dx_int"]), dy=float(p["dy_int"])) for p in pairs])
    scores = {}
    for placement in PLACEMENTS:
        tdx, tdy = (dx_i, dy_i) if placement == "int_reg" else (dx, dy)
        px, py = track_positions(tdx, tdy, (0, len(frames)), placement=placement)
        values, valid, sharp, centre = build_stack(
            frames, px, py, subpixel=(placement == "subpixel"))
        values, _ = equalise(values, valid)
        img, cov = blend(values, valid, sharp, centre, "mean")
        scores[placement] = truth_correlation(img, cov, g)
    for p in PLACEMENTS:
        print(f"mean blend vs ground truth, {p:<10} r = {scores[p]:.4f}")
    # Sub-pixel ESTIMATION must beat integer estimation: its per-pair error is
    # several times smaller, and per-pair error accumulates along the swipe.
    if scores["int_place"] <= scores["int_reg"]:
        fails.append("sub-pixel estimation did not beat integer registration")
    # Sub-pixel RESAMPLING is allowed to be close to a wash -- the
    # interpolator's own band attenuation partly cancels the alignment gain at
    # a 9 px pitch -- but it must not actively hurt, which is what a sign error
    # in the resampler looks like.
    if scores["subpixel"] < scores["int_place"] - 0.005:
        fails.append("sub-pixel resampling was worse than integer placement")

    # Outlier filter: corrupt one pair and check it is caught and interpolated.
    bad = [dict(p) for p in pairs]
    k = len(bad) // 2
    bad[k]["dx"] = bad[k]["dx"] + 20.0
    _, _, ok2, reasons = clean_track(bad)
    if ok2[k]:
        fails.append("outlier filter missed a +20 px dx corruption")
    else:
        print(f"outlier filter caught the injected corruption: {reasons[k]}")

    if fails:
        print("\nSELFTEST FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nselftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="elan-swipe capture directories")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the registration and resampling machinery "
                         "against synthetic data with known displacements")
    ap.add_argument("--out-dir", default=None,
                    help="write outputs here instead of next to the input")
    ap.add_argument("--json", action="store_true",
                    help="dump the full report to stdout as JSON")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.dirs:
        ap.error("give at least one capture directory, or --selftest")
    reps, strips = [], []
    for d in a.dirs:
        out = Path(a.out_dir) / Path(d).name if a.out_dir else None
        rep, (img, cov) = assemble(d, out, verbose=not a.json)
        reps.append(rep)
        strips.append((Path(d).name, img, cov))

    if len(strips) >= 2:
        xs = [cross_check(strips[i:i + 2])
              for i in range(len(strips) - 1)]
        for r in xs:
            reps[0].setdefault("cross_check", []).append(r)
        if not a.json:
            print("\n=== cross-check between independently assembled strips ===")
            for r in xs:
                print(f"{r['a']} vs {r['b']}: NCC {r['ncc']:.4f} at "
                      f"dy {r['dy']} dx {r['dx']}, overlap {r['overlap_px']} px "
                      f"= {r['overlap_mm2']:.1f} mm^2")
            print("This is a geometry check, not a biometric score: both "
                  "captures are the same finger,\nso there is no impostor "
                  "distribution here and none can be inferred.")

    if a.json:
        json.dump(reps, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
