#!/usr/bin/env python3
"""
Multi-swipe finger map for the ELAN 04f3:0c6e.

swipe_assemble.py turns ONE swipe into a strip.  One swipe is bounded by the
device itself: capture wedges after 420-563 ms with a persistent 0xaf from
pre_scan, so a swipe is 13-19 frames, about 12 mm of travel, and the assembled
strip covers roughly 2.1x the area of the 7.6 x 2.6 mm sensor window.  That is
not a big enough finger map to matter.  This file is the enrolment step: it
registers several strips into ONE canvas, which is the only way past the
per-swipe ceiling.

Design, and the reasons for it:

  * Registration is done on the RIDGE BAND, not on raw intensity.  Raw strips
    correlate at 0.86 between the two real captures here, but a large part of
    that is the pressure envelope, which is smooth, broad and common to every
    capture of every finger.  Band-passing first is what makes the correlation
    peak sharp enough to have a peak-to-sidelobe ratio worth thresholding on.

  * The translation search is exhaustive, over EVERY shift with enough overlap,
    via a masked FFT normalised cross-correlation (Padfield's formulation).
    Strips from different swipes start at unrelated places on the finger, so
    there is no small search window to lean on.  The masked NCC is validated
    exact against a naive per-shift loop in --selftest.

  * Rotation is searched explicitly, over +-25 deg.  That range is not
    caution, it is measurement: the six real strips on this machine need
    -19.75 to +16.25 deg between them.  The finger does not land at the same
    angle twice, and a +-10 deg search -- which is what this file had first --
    silently returns rail-pinned nonsense for a third of the pairs.

  * Anisotropic scale along the swipe axis is SEARCHED, not assumed.  This file
    originally argued that it could be left at 1, on the grounds that a strip is
    built from displacements measured in pixels and is therefore metric by
    construction.  That argument is wrong, and `assess_scale` disproves it: two
    of the four real strip pairs prefer 3% of compression, and one of them gains
    0.121 NCC from being allowed it.  Three percent of a 330 px strip is 10 px,
    which was exactly the pose inconsistency that survived pose averaging while
    the model was rigid.  Adding the scale search took link NCCs from
    0.580/0.624/0.737 to 0.733/0.747/0.750, cut residual inconsistency from
    10.1 px to 4.3 px, and took the finished map from 0.905x a single frame to
    1.041x.  The finger is soft and the sensor drags on it.

  * A strip that does not belong is REJECTED.  This is the failure mode that
    killed the earlier press-mosaic attempt, where only 4-9 of 19 captures
    registered and the rest were forced in and smeared the canvas.  Acceptance
    needs all of: enough overlap, an NCC above threshold, a peak-to-sidelobe
    ratio above threshold, a rotation that is not on the search rail, and
    half-strip support -- each half of the strip, registered by itself, must
    still prefer the shift the whole strip chose.  The half-strip test is the
    one a forced registration cannot fake, because a false peak is one
    accidental coincidence and does not survive being cut in half.  It has to
    be scored on correlation VALUE rather than on position; see
    `split_half_check` for the ridge-aliasing trap that makes the obvious
    position-based version reject almost everything.

  * Thresholds are calibrated against a NULL DISTRIBUTION measured at run time:
    each strip is registered against phase-randomised, flipped and 180-rotated
    versions of the other strips, and against the 45 press captures.  Phase
    randomisation is the important one -- it preserves the power spectrum
    exactly, so it has the same ridge-band content and the same tile
    statistics, and only the correspondence is destroyed.  This is NOT an
    impostor distribution: every capture on this machine is one finger.  It
    bounds false acceptance against non-matching CONTENT, not against a
    different person.

  * Blending reuses the assembler's winner (per-pixel sharpest-contributor,
    `sharp_win`) and blends from FRAMES, not from finished strips: each frame is
    warped once, directly from sensor space into canvas space, by the
    composition of its within-swipe placement and its swipe's canvas pose.  The
    argument for that is that resampling a finished strip would be a second
    interpolation on top of the first, and the ridge band is what a second
    interpolation costs you.

    Both routes are built and measured, and on this data the argument does NOT
    win: the strip route scores 1.067x a single frame against the frame route's
    1.041x.  The reason is visible in the same report -- the frame route is far
    more sensitive to residual misregistration, because it picks a contributor
    per PIXEL, so neighbouring pixels can come from swipes that are still 4.3 px
    out of register, and the ridge breaks at that seam.  The strip route picks
    per strip, so a whole region comes from one swipe and stays internally
    coherent.  The frame route is kept as the default because it is the one that
    improves as registration improves (it went 0.905x -> 1.041x when the scale
    search cut the residual, while the strip route moved 0.994x -> 1.067x), but
    the honest statement today is that they are within 2.5% of each other and
    the second interpolation is not the dominant cost.  The dominant cost is
    registration.

Usage:
    swipe_map.py DIR [DIR ...] [--out-dir DIR] [--json]
    swipe_map.py --selftest
    swipe_map.py --selftest-only-fast

Each DIR is an elan-swipe capture directory.  A capture containing several
swipes (a stream where contact was broken and remade) is split automatically;
see `segment_capture`.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from swipe_assemble import (                                    # noqa: E402
    FRAME_W, FRAME_H, DPI, MM_PER_PX,
    DX_MIN, DX_MAX, DY_ABS, MIN_PEAK,
    ANALYSIS_WIN, ANALYSIS_STRIDE,
    BLEND_MODES,
    blur, ridge_dog, ridge_quality_map,
    window_metrics, summarise_tiles, strip_arrays,
    load_swipe, register_pair, frame_quality, contact_window, clean_track,
    track_positions, build_stack, equalise, blend, overlap_retention,
    shift_subpixel, _raised_cosine, _parabolic, _fourier_shift,
    to_u8, write_pgm,
)

# ---------------------------------------------------------------------------
# tunables, with the measurement each one rests on
# ---------------------------------------------------------------------------

# Rotation search.  The first version of this file searched +-10 deg, on the
# assumption that a finger lands at nearly the same angle every time.  It does
# not: the six real strips on this machine need -19.75 to +16.25 deg between
# them, and several pairs came back sitting exactly on a 10 deg rail.  A rail
# hit is not a small error, it is a wrong answer with a plausible-looking
# number attached, which is why `judge` now rejects on it.  +-25 deg at 1 deg
# costs 51 warps and 51 FFT sets, about 1.5 s per registration, and enrolment
# happens once.
ROT_MAX = 25.0
ROT_STEP = 1.0
ROT_FINE = 0.25

# Along-axis scale search.  See register_strip for the measurements: real pairs
# want up to 3% of compression along the swipe axis and one of them gains 0.121
# NCC from being allowed it.  +-6% at 1% is comfortably wider than anything
# measured, so a solution sitting on the rail means something is wrong rather
# than that the range was mean, and `judge` rejects on it.
SCALE_GRID = np.arange(0.94, 1.0601, 0.01)

# Minimum overlap for a registration to mean anything.  A strip covers about
# 17000 px; 5000 px is 12.9 mm^2, a little over half a sensor window.  Below
# that the normalisation is estimated from too little data and the NCC surface
# grows a fringe of high values at the extreme shifts.
#
# This is the floor for the SEARCH -- shifts with less overlap than this are not
# normalised and not considered.  It is no longer the binding acceptance test:
# 5000 px is 12.9 mm^2 and MIN_TRUSTED_OVERLAP_MM2 below rejects at 16 mm^2, so
# the mm^2 gate always fires first.  The two are kept separate because they do
# different jobs -- this one decides which shifts are even evaluated, that one
# decides whether the winning shift can be believed.
MIN_OVERLAP_MAP = 5000

# Acceptance thresholds, set from the null distribution measured in
# `null_distribution` on the six real strips.  The report prints the measured
# null maximum next to each threshold, so the margin is visible rather than
# asserted.  What 135 non-matching registrations gave, against the accepted
# genuine links from the same run:
#
#              null max   null p95   worst accepted genuine link
#   NCC          0.474      0.408          0.733
#   PSR         13.06       ~5             13.75
#
# Zero of the 135 nulls would be accepted.
#
# Read those two rows carefully, because they say opposite things.  NCC has a
# wide margin: 0.474 against 0.733.  PSR has almost none: 13.06 against 13.75.
# So on this data NCC is the discriminator and PSR is nearly useless as one --
# which is the REVERSE of what this file concluded before the along-axis scale
# search existed, when genuine NCCs were 0.58-0.74 and the NCC margin was the
# thin one.  Adding scale lifted the genuine correlations by about 0.13 and left
# the nulls where they were, and that is what flipped the ordering.
#
# PSR is kept in the acceptance rule anyway, because it fails independently of
# NCC and the cost of keeping it is one rejection of something already rejected.
# It should not be relied on alone, and the synthetic sweep shows why: the
# worst answer there, wrong by 15 px, scored PSR 16.9.
#
# Note also what a null max of 0.474 means: a strip turned end for end still
# correlates at 0.47 with another strip of the same finger.  Broad ridge-flow
# similarity is not identity, and any NCC threshold has to live above it.
ACCEPT_NCC = 0.55
ACCEPT_PSR = 8.0

# Half-strip support: the weaker half's NCC at the full solution, over the full
# NCC.  Measured, this turned out to be a poor discriminator on its own --
# nulls reach 0.96, because a null's "solution" is a broad weak peak that both
# halves agree is equally unremarkable.  It is kept as a floor against the one
# failure it does catch, a match carried entirely by one half, and it is
# reported so that the weakness is visible instead of implied.
ACCEPT_SPLIT_SUPPORT = 0.50

# Half-strip excess: how much better a half's own peak is than the full
# solution, counted only when that peak is further than SPLIT_SAME_PX away.
# This is the discriminator that works.  Genuine pairs measure 0.00-0.02;
# nulls and self-inconsistent pairs measure 0.07-0.29.
ACCEPT_SPLIT_EXCESS = 0.06

# Radius inside which a half's own peak counts as "the same place".  It is one
# and a bit ridge pitches on purpose: ridge pitch is 9-10 px here, so a half
# strip routinely locks onto the neighbouring ridge, and a 9 px disagreement is
# aliasing rather than dissent.  Real dissent measured 47-227 px.
SPLIT_SAME_PX = 12.0

# Overlap below which a registration should not be trusted even when it looks
# confident.  This is NOT a guess: `synthetic_split_check` sweeps overlap
# against a known ground truth, and on 20260814-170350 the corner error runs
# 15.43, 4.37, 2.57 px at 13.0, 12.9, 14.7 mm^2 and then drops to 1.77 px and
# below from 16.5 mm^2 upward.  The worst of those scored NCC 0.690 and PSR
# 16.9, so neither of the obvious confidence numbers would have caught it.
#
# The acceptance rule already rejects all three low-overlap cases, but only
# because the split-half test cannot be RUN on strips that small -- a real
# effect, and one that happens to land in the right place, but an accident
# rather than a designed margin.  This constant states the requirement
# explicitly so that it survives someone later making the split-half test work
# on smaller strips.  16 mm^2 is a little over two sensor windows' worth of
# overlap; at 12 mm of travel per swipe that is a generous but achievable ask.
MIN_TRUSTED_OVERLAP_MM2 = 16.0

# Exclusion radius around the peak when measuring the peak-to-sidelobe ratio.
# The correlation peak of a correctly registered strip is a few px wide
# (ridge pitch is 9-10 px), so 8 px excludes the peak and its shoulders
# without eating the sidelobe statistics.
PSR_EXCLUDE = 8

# Erode strip masks by this much before registering.  The outermost pixels of
# a strip are single-contributor and half-tapered, and they carry seam
# artefacts that correlate with nothing.
MASK_ERODE = 4


# ---------------------------------------------------------------------------
# 1. geometry: similarity transforms on (y, x) image coordinates
# ---------------------------------------------------------------------------
def similarity(theta_deg=0.0, sx=1.0, sy=1.0, ty=0.0, tx=0.0, cy=0.0, cx=0.0):
    """3x3 homogeneous matrix acting on column vectors [y, x, 1].

    canvas = R(theta) . diag(sy, sx) . (p - c) + c + t

    Rotation is in the usual image sense (x right, y down), so a positive
    theta turns the image clockwise on screen:

        Y' =  cos.Y + sin.X
        X' = -sin.Y + cos.X

    The sign convention is not obvious and is not something to get wrong
    quietly, so --selftest warps a real strip by a known angle and checks the
    estimator returns the negative of it (the pose maps the strip INTO the
    canvas, so it must undo the rotation that was applied).
    """
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c * sy, s * sx], [-s * sy, c * sx]], dtype=np.float64)
    A = np.eye(3)
    A[:2, :2] = R
    cvec = np.array([cy, cx], dtype=np.float64)
    A[:2, 2] = cvec - R @ cvec + np.array([ty, tx], dtype=np.float64)
    return A


def translation(ty, tx):
    A = np.eye(3)
    A[0, 2] = ty
    A[1, 2] = tx
    return A


def apply_to_points(A, pts):
    """pts is (N, 2) in (y, x); returns (N, 2) transformed."""
    pts = np.asarray(pts, dtype=np.float64)
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (A @ h.T).T[:, :2]


def transform_corners(A, shape):
    h, w = shape
    return apply_to_points(A, [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])


def decompose(A):
    """Recover (theta_deg, scale_y, scale_x) from a similarity/affine matrix."""
    m = A[:2, :2]
    sy = float(math.hypot(m[0, 0], m[1, 0]))
    sx = float(math.hypot(m[0, 1], m[1, 1]))
    theta = float(math.degrees(math.atan2(m[0, 1] / max(sx, 1e-12),
                                          m[1, 1] / max(sx, 1e-12))))
    return theta, sy, sx


# ---------------------------------------------------------------------------
# 2. resampling: one Catmull-Rom warp, never two
# ---------------------------------------------------------------------------
def _cr_weights(t):
    """Catmull-Rom weights for taps at floor-1, floor, floor+1, floor+2."""
    t2 = t * t
    t3 = t2 * t
    return (-0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2)


def sample_bicubic(img, ys, xs):
    """Catmull-Rom sample of `img` at float coordinates (ys, xs).

    Same kernel as the assembler's shift_subpixel, for the same reason:
    bilinear is a low-pass filter and at a 9 px ridge pitch it throws away
    several percent of exactly the band being measured.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    i0 = np.floor(ys).astype(np.int64)
    j0 = np.floor(xs).astype(np.int64)
    wy = _cr_weights(ys - i0)
    wx = _cr_weights(xs - j0)
    out = np.zeros(ys.shape, dtype=np.float64)
    for a in range(4):
        ia = np.clip(i0 - 1 + a, 0, h - 1)
        for b in range(4):
            jb = np.clip(j0 - 1 + b, 0, w - 1)
            out += wy[a] * wx[b] * img[ia, jb]
    return out


def warp(img, A, out_shape, border=0, extra=()):
    """Warp `img` into `out_shape` under the forward transform A.

    A maps source (y, x) -> destination (Y, X); the sampler needs the inverse.
    Returns (warped, valid) plus a warped copy of each array in `extra`, all
    sampled with the same coordinates so masks and quality maps stay in step
    with the image.

    `border` pixels are dropped from the source edge.  Two of those are the
    cubic kernel's outer taps, which would otherwise be extrapolated; the rest
    is the frame's own dead edge.
    """
    ih, iw = img.shape
    oh, ow = out_shape
    Ainv = np.linalg.inv(A)
    Y, X = np.mgrid[0:oh, 0:ow].astype(np.float64)
    ys = Ainv[0, 0] * Y + Ainv[0, 1] * X + Ainv[0, 2]
    xs = Ainv[1, 0] * Y + Ainv[1, 1] * X + Ainv[1, 2]
    lo = float(border)
    valid = ((ys >= lo) & (ys <= ih - 1 - lo) &
             (xs >= lo) & (xs <= iw - 1 - lo))
    ysc = np.clip(ys, 0, ih - 1)
    xsc = np.clip(xs, 0, iw - 1)
    out = sample_bicubic(img, ysc, xsc) * valid
    if not extra:
        return out, valid
    ex = [sample_bicubic(e, ysc, xsc) * valid for e in extra]
    return (out, valid, *ex)


def erode(mask, r):
    """Binary erosion by an (2r+1) square, via a summed-area table."""
    if r <= 0:
        return mask.copy()
    m = mask.astype(np.int32)
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = mask.shape
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    s = (ii[y1[:, None], x1[None, :]] - ii[y0[:, None], x1[None, :]] -
         ii[y1[:, None], x0[None, :]] + ii[y0[:, None], x0[None, :]])
    # Only pixels whose full window is inside the image AND fully set survive.
    inside = ((np.arange(h) >= r) & (np.arange(h) < h - r))[:, None] & \
             ((np.arange(w) >= r) & (np.arange(w) < w - r))[None, :]
    out = (s == area) & inside & mask
    return out


# ---------------------------------------------------------------------------
# 3. masked normalised cross-correlation over every translation
# ---------------------------------------------------------------------------
def _fft_pack(A, MA, shape):
    """Pre-transform the fixed side of the correlation.

    The canvas does not change while the angle grid is swept, so its three
    forward transforms are computed once and reused for every angle.  That is
    half the FFT work in the search.
    """
    A = np.where(MA, A, 0.0)
    m = MA.astype(np.float64)
    return (np.fft.rfft2(A, s=shape),
            np.fft.rfft2(A * A, s=shape),
            np.fft.rfft2(m, s=shape))


def masked_ncc(pack_a, shape_a, B, MB, min_overlap, shape=None):
    """NCC of B against A over EVERY integer shift, honouring both masks.

    Padfield's masked NCC: with the masks folded into the transforms, all six
    running sums (n, sum a, sum b, sum a^2, sum b^2, sum ab) come out of six
    correlations, so every shift is normalised over exactly its own overlap
    rather than over a fixed window.  That is what lets the search be
    exhaustive: strips from different swipes start at unrelated places on the
    finger and there is no small window to search inside.

    Convention matches the assembler: A[y, x] pairs with B[y - dy, x - dx], and
    the surface is indexed modulo the padded size, so a shift of -3 lives at
    index -3.

    Returns (surface, overlap_counts), both (P0, P1), NaN where the overlap is
    below `min_overlap` or a variance vanishes.
    """
    Ha, Wa = shape_a
    Hb, Wb = B.shape
    if shape is None:
        shape = (Ha + Hb, Wa + Wb)
    fA, fA2, fMA = pack_a
    B = np.where(MB, B, 0.0)
    mb = MB.astype(np.float64)
    fB = np.fft.rfft2(B, s=shape)
    fB2 = np.fft.rfft2(B * B, s=shape)
    fMB = np.fft.rfft2(mb, s=shape)

    def ix(fa, fb):
        return np.fft.irfft2(fa * np.conj(fb), s=shape)

    n = np.round(ix(fMA, fMB))
    sab = ix(fA, fB)
    sa = ix(fA, fMB)
    sb = ix(fMA, fB)
    sa2 = ix(fA2, fMB)
    sb2 = ix(fMA, fB2)

    ok = n >= min_overlap
    nz = np.where(ok, n, 1.0)
    num = sab - sa * sb / nz
    va = sa2 - sa * sa / nz
    vb = sb2 - sb * sb / nz
    den = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok & (den > 1e-9), num / np.maximum(den, 1e-30), np.nan)
    return r, n


def _wrap(i, p):
    """Signed shift from a modulo index."""
    return i - p if i > p // 2 else i


def surface_peak(surf, n):
    """Peak of an NCC surface, with its signed shift and a sidelobe rating.

    The peak-to-sidelobe ratio is (peak - median) / (1.4826 * MAD) over the
    valid shifts outside a PSR_EXCLUDE disc around the peak.  Median/MAD rather
    than mean/std because the surface is not Gaussian: correct registrations
    have a ridge of elevated values running along the swipe axis (a strip
    shifted by one ridge pitch still correlates somewhat), and that ridge would
    inflate a standard deviation and hide the very peak it surrounds.
    """
    if not np.isfinite(surf).any():
        return None
    k = int(np.nanargmax(surf))
    p0, p1 = surf.shape
    iy, ix = divmod(k, p1)
    peak = float(surf[iy, ix])
    dy, dx = _wrap(iy, p0), _wrap(ix, p1)

    yy = np.arange(p0)[:, None]
    xx = np.arange(p1)[None, :]
    dyy = np.minimum(np.abs(yy - iy), p0 - np.abs(yy - iy))
    dxx = np.minimum(np.abs(xx - ix), p1 - np.abs(xx - ix))
    far = (dyy > PSR_EXCLUDE) | (dxx > PSR_EXCLUDE)
    side = surf[far & np.isfinite(surf)]
    if side.size < 200:
        psr = 0.0
    else:
        med = float(np.median(side))
        mad = float(np.median(np.abs(side - med))) * 1.4826
        psr = float((peak - med) / max(mad, 1e-6))

    # Sub-pixel refinement of the shift, same parabolic fit as the assembler.
    fy = _parabolic(surf[(iy - 1) % p0, ix], peak, surf[(iy + 1) % p0, ix])
    fx = _parabolic(surf[iy, (ix - 1) % p1], peak, surf[iy, (ix + 1) % p1])
    return {
        "peak": peak,
        "dy": dy + fy,
        "dx": dx + fx,
        "dy_int": dy,
        "dx_int": dx,
        "psr": round(psr, 2),
        "overlap": int(n[iy, ix]),
    }


# ---------------------------------------------------------------------------
# 4. strips: one per swipe, carrying the frames that built them
# ---------------------------------------------------------------------------
# A capture is not necessarily one swipe.  20260814-165428 on this machine is a
# 60-frame stream containing FOUR swipes: the finger lifts, dx collapses to
# zero and the pair NCC goes NEGATIVE (-0.18, -0.24, -0.47) at each break,
# then a new swipe starts.  Those breaks are unmistakable -- every in-contact
# pair on that capture scores 0.79-0.99 -- so segmenting on them is safe, and
# it is the difference between having 2 strips to test the mapper on and
# having 6.
SEGMENT_BREAK_PEAK = 0.5     # pair NCC below this ends a swipe
SEGMENT_MIN_FRAMES = 6       # shorter runs cannot make a useful strip


def segment_capture(frames, dxr, dyr):
    """Split a capture into runs of frames that are one continuous swipe.

    Returns a list of (start, stop) index pairs.  A single-swipe capture comes
    back as one segment covering everything, so this is transparent for the
    normal case.
    """
    if len(frames) < SEGMENT_MIN_FRAMES:
        return []
    peaks = []
    for i in range(len(frames) - 1):
        p = register_pair(frames[i], frames[i + 1], dxr, dyr)
        peaks.append(p["peak"] if p else -1.0)
    segs, start = [], 0
    for i, pk in enumerate(peaks):
        if pk < SEGMENT_BREAK_PEAK:
            if i + 1 - start >= SEGMENT_MIN_FRAMES:
                segs.append((start, i + 1))
            start = i + 1
    if len(frames) - start >= SEGMENT_MIN_FRAMES:
        segs.append((start, len(frames)))
    return segs


class Strip:
    """One assembled swipe, plus everything needed to re-place its frames.

    The map does NOT blend finished strips.  It keeps each strip's frames and
    their within-swipe positions so that, once the swipe's canvas pose is
    known, every frame can be warped once, straight from sensor space to canvas
    space.  `img`/`cov` exist only to drive registration and to provide the
    double-resample comparison.
    """

    def __init__(self, name, frames, px, py, img, cov, report):
        self.name = name
        self.frames = frames          # (N, 52, 150), the in-contact frames
        self.px = px                  # within-strip positions, float px
        self.py = py
        self.img = img                # sharp_win blend of this swipe alone
        self.cov = cov
        self.report = report
        self.pose = None              # 3x3 strip -> canvas, once accepted
        self.gain = (1.0, 0.0)
        self.conf = None

    @property
    def band(self):
        return self.report.get("band_median", 0.0)

    def rank(self):
        """Seed/order key: covered area weighted by local ridge quality."""
        return self.report.get("coverage_px", 0) * max(self.band, 1e-6)


def assemble_one(frames, name, verbose=False):
    """Assemble one swipe's frames into a strip.

    Deliberately the same pipeline as swipe_assemble.assemble -- the same
    quality gate, the same NCC search, the same curvature outlier filter, the
    same sub-pixel placement and the same winning blend -- but it takes frames
    rather than a directory, and it returns the per-frame placements.  Anything
    that changed here would invalidate the assembler's measurements, so
    nothing does: the functions are imported, not reimplemented.
    """
    dxr = list(range(DX_MIN, DX_MAX + 1))
    dyr = list(range(-DY_ABS, DY_ABS + 1))
    q = frame_quality(frames)
    keep = contact_window(q)
    if keep[1] - keep[0] < 3:
        return None
    pairs = [register_pair(frames[i], frames[i + 1], dxr, dyr)
             for i in range(len(frames) - 1)]
    dx, dy, ok, _ = clean_track(pairs)
    px, py = track_positions(dx, dy, keep, placement="subpixel")
    kept = frames[keep[0]:keep[1]]
    values, valid, sharp, centre = build_stack(kept, px, py, subpixel=True)
    values, _ = equalise(values, valid)
    img, cov = blend(values, valid, sharp, centre, "sharp_win")
    peaks = [p["peak"] for p in pairs if p]
    m = strip_arrays(window_metrics(img, cov))
    rep = dict(m)
    rep.update({
        "name": name,
        "n_frames": int(keep[1] - keep[0]),
        "contact_window": [int(keep[0]), int(keep[1])],
        "peak_median": round(float(np.median(peaks)), 4) if peaks else None,
        "peak_min": round(float(min(peaks)), 4) if peaks else None,
        "n_pairs_rejected": int((~ok).sum()),
        "travel_mm": round(float(dx.sum()) * MM_PER_PX, 2),
        "size_px": [int(img.shape[1]), int(img.shape[0])],
        "coverage_px": int(cov.sum()),
        "coverage_mm2": round(float(cov.sum()) * MM_PER_PX ** 2, 2),
    })
    if verbose:
        print(f"  strip {name}: {rep['n_frames']} frames, "
              f"{rep['size_px'][0]}x{rep['size_px'][1]} px, "
              f"{rep['coverage_mm2']:.1f} mm^2 covered, "
              f"band median {rep.get('band_median', float('nan')):.3f}")
    return Strip(name, kept, px, py, img, cov, rep)


def strips_from_capture(path, verbose=True):
    """Load a capture directory and return every usable strip in it."""
    frames, _ = load_swipe(path)
    dxr = list(range(DX_MIN, DX_MAX + 1))
    dyr = list(range(-DY_ABS, DY_ABS + 1))
    segs = segment_capture(frames, dxr, dyr)
    name0 = Path(path).name
    out = []
    for k, (a, b) in enumerate(segs):
        nm = name0 if len(segs) == 1 else f"{name0}#{k}"
        try:
            s = assemble_one(frames[a:b], nm, verbose=verbose)
        except RuntimeError as e:
            if verbose:
                print(f"  strip {nm}: not assembled ({e})")
            continue
        if s is not None:
            out.append(s)
        elif verbose:
            print(f"  strip {nm}: too few in-contact frames")
    if verbose and len(segs) > 1:
        print(f"  {name0}: {len(frames)} frames contained {len(segs)} swipes "
              f"(split on pair NCC < {SEGMENT_BREAK_PEAK})")
    return out


# ---------------------------------------------------------------------------
# 5. registering a strip onto the canvas
# ---------------------------------------------------------------------------
def prep_band(img, cov, erode_r=MASK_ERODE):
    """Band-passed, masked image ready to correlate, plus its eroded mask.

    Correlating raw intensity is what the assembler's cross_check did, and it
    reported 0.857 between the two real strips.  Most of that is the pressure
    envelope: a smooth, broad, roughly centred blob that every capture of every
    finger has.  It inflates the peak AND flattens the surface around it, so
    the peak-to-sidelobe ratio -- the thing acceptance actually leans on --
    comes out meaningless.  Band-passing first drops the genuine peak somewhat
    and drops the sidelobes much further.
    """
    m = erode(cov, erode_r)
    d = ridge_dog(np.where(cov, img, 0.0))
    d = np.where(m, d, 0.0)
    if m.any():
        d = np.where(m, d - d[m].mean(), 0.0)
    return d, m


def warped_source(img, mask, theta, sx=1.0, sy=1.0):
    """Rotate/scale a strip about its centre into its own tight buffer.

    Returns (image, mask, W) where W maps ORIGINAL strip coordinates to buffer
    coordinates, so that once the search finds a buffer offset t the full pose
    is translation(t) @ W.  Keeping the composition explicit is what allows the
    frames to be warped once at the end, from sensor space straight to canvas
    space, rather than being dragged through the strip's own resampling twice.
    """
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    W0 = similarity(theta, sx=sx, sy=sy, cy=cy, cx=cx)
    cor = transform_corners(W0, img.shape)
    miny, minx = cor[:, 0].min(), cor[:, 1].min()
    maxy, maxx = cor[:, 0].max(), cor[:, 1].max()
    W = translation(-miny, -minx) @ W0
    out_shape = (int(math.ceil(maxy - miny)) + 1, int(math.ceil(maxx - minx)) + 1)
    wi, val, wm = warp(img, W, out_shape, border=1, extra=(mask.astype(float),))
    m = val & (wm > 0.99)
    return np.where(m, wi, 0.0), m, W


def _fft_shape(a_shape, b_shape):
    return (a_shape[0] + b_shape[0], a_shape[1] + b_shape[1])


def register_strip(canvas_img, canvas_cov, strip_img, strip_cov,
                   angles=None, min_overlap=MIN_OVERLAP_MAP,
                   refine=True, split_half=True, sx=1.0, scales=None):
    """Find the pose that puts `strip` on `canvas`, and rate the result.

    Search: an exhaustive translation search (masked FFT NCC) at every angle in
    `angles`, then, if `scales` is given, a sweep of anisotropic scale along the
    strip's long axis at the winning angle, then a re-refinement of the angle at
    the winning scale.  Returns None if no shift anywhere had enough overlap.

    Why scale is searched rather than assumed to be 1.  The argument for
    assuming it -- a strip is built from displacements measured in pixels, so it
    is metric by construction and swipe speed cannot stretch it -- sounds
    airtight and is contradicted by the data.  Measured on the real strips:

        20260814-165428#1 <- 20260814-170350   best scale 0.97, NCC 0.624 -> 0.745
        20260814-165428#2 <- 20260814-165428#1 best scale 0.97, NCC 0.580 -> 0.631
        20260814-170350   <- 20260814-165457   best scale 1.01, NCC 0.737 -> 0.749
        20260814-165428#2 <- 20260814-165428#3 best scale 1.00, no gain

    Two of four pairs want 3% of compression and one gains 0.12 NCC from it,
    which is not a rounding error.  Three percent of a 330 px strip is 10 px,
    and 10 px is precisely the residual pose inconsistency that survived pose
    averaging when the model was rigid.  The finger is soft and the sensor drags
    on it; the skin does not travel as one rigid body.

    The sweep is staged rather than a full 3-D grid: angle at scale 1, then
    scale at the winning angle, then angle again at the winning scale.  That is
    about 1.6x the cost of the rigid search instead of 13x, and it works because
    the angle and scale axes are close to separable over this range.
    """
    if angles is None:
        angles = np.arange(-ROT_MAX, ROT_MAX + 1e-9, ROT_STEP)
    A, MA = prep_band(canvas_img, canvas_cov)
    B, MB = prep_band(strip_img, strip_cov)
    if MA.sum() < min_overlap or MB.sum() < min_overlap:
        return None

    # The FFT size must cover the largest rotated and scaled buffer, so that one
    # packed transform of the canvas serves every angle and scale.
    rot = max(abs(float(a)) for a in angles) if len(angles) else 0.0
    # `scales or ()` would raise: scales is normally a numpy array, and a numpy
    # array of more than one element has no unambiguous truth value.
    scale_list = [] if scales is None else [float(s) for s in scales]
    sx_max = max([sx] + scale_list)
    th = math.radians(rot)
    bh, bw = B.shape
    pad_h = int(math.ceil(bh * math.cos(th) + bw * abs(math.sin(th)))) + 2
    pad_w = int(math.ceil(bw * math.cos(th) * max(sx_max, 1.0) +
                          bh * abs(math.sin(th)))) + 2
    shape = _fft_shape(MA.shape, (pad_h, pad_w))
    pack = _fft_pack(A, MA, shape)

    def evaluate(th_deg, s):
        bi, bm, W = warped_source(B, MB, float(th_deg), sx=float(s))
        surf, n = masked_ncc(pack, MA.shape, bi, bm, min_overlap, shape=shape)
        pk = surface_peak(surf, n)
        if pk is None:
            return None
        return dict(pk, theta=float(th_deg), sx=float(s), W=W, surf=surf, n=n,
                    bshape=bi.shape, bimg=bi, bmask=bm)

    best = None
    profile = []
    for th_deg in angles:
        cand = evaluate(th_deg, sx)
        profile.append((float(th_deg), cand["peak"] if cand else float("nan")))
        if cand and (best is None or cand["peak"] > best["peak"]):
            best = cand
    if best is None:
        return None

    scale_profile = []
    if scales is not None and len(scales) > 1:
        for s in scales:
            cand = (best if abs(float(s) - best["sx"]) < 1e-9
                    else evaluate(best["theta"], s))
            scale_profile.append((round(float(s), 4),
                                  round(cand["peak"], 4) if cand else None))
            if cand and cand["peak"] > best["peak"]:
                best = cand

    if refine and len(angles) > 2:
        # Refine the angle on a fine grid around the coarse winner, at the
        # winning scale.  A parabolic fit on the coarse profile is tempting and
        # wrong: the peak vs angle curve is not parabolic over a 1 deg grid on a
        # strip this long, because rotation moves the far end of the strip much
        # more than the near end.
        t0 = best["theta"]
        for th_deg in np.arange(t0 - ROT_STEP, t0 + ROT_STEP + 1e-9, ROT_FINE):
            if abs(th_deg - t0) < 1e-9:
                continue
            cand = evaluate(th_deg, best["sx"])
            if cand and cand["peak"] > best["peak"]:
                best = cand

    lim = max(abs(float(a)) for a in angles) if len(angles) else 0.0
    s_lo = min(scales) if scales is not None and len(scales) else None
    s_hi = max(scales) if scales is not None and len(scales) else None
    res = {
        "theta": round(best["theta"], 3),
        "at_rail": bool(len(angles) > 2 and
                        abs(abs(best["theta"]) - lim) <= ROT_STEP + 1e-9),
        "sx": round(best["sx"], 4),
        "scale_at_rail": bool(s_lo is not None and
                              (abs(best["sx"] - s_lo) < 1e-9 or
                               abs(best["sx"] - s_hi) < 1e-9)),
        "scale_profile": scale_profile,
        "dy": round(best["dy"], 3),
        "dx": round(best["dx"], 3),
        "ncc": round(best["peak"], 4),
        "psr": best["psr"],
        "overlap_px": best["overlap"],
        "overlap_mm2": round(best["overlap"] * MM_PER_PX ** 2, 2),
        "angle_profile": [(round(a, 2), round(p, 4) if np.isfinite(p) else None)
                          for a, p in profile],
        "pose": translation(best["dy"], best["dx"]) @ best["W"],
    }

    if split_half:
        res.update(split_half_check(pack, MA.shape, best, min_overlap, shape))
    return res


def split_half_check(pack, a_shape, best, min_overlap, shape):
    """Does each half of the strip, on its own, support the full solution?

    This is the check a forced registration fails.  A genuine match is
    supported by correspondence everywhere in the overlap, so either half of
    the strip, registered by itself, still likes the shift the whole strip
    chose.  A false peak is one accidental coincidence somewhere in the
    overlap: cut the strip in half and the half containing the coincidence
    keeps it while the other half prefers somewhere else entirely.

    The obvious version of this test -- do the halves' own peaks land within a
    few pixels of the full solution -- does not work on a fingerprint, and the
    first implementation here rejected almost everything because of it.  Ridges
    have a 9-10 px pitch, so a half-strip's correlation surface has near-equal
    peaks one whole pitch either side of the truth, and a half will happily sit
    on the neighbouring ridge.  Measured on the real strips, disagreements came
    out at 8.5, 9.1 and 17.3 px: one and two ridge pitches, not disagreement.

    So the test is scored on VALUE, not on position:

      support -- each half's NCC evaluated AT the full solution, relative to
        the full NCC.  A half that carries no correspondence scores near zero
        here no matter where its own peak is.
      excess -- how much better a half's own peak is than the full solution,
        counted only when that peak is far away.  Ridge aliasing produces a
        near-tie (a few hundredths), so a real preference for somewhere else
        stands out from it.
    """
    bm = best["bmask"]
    bi = best["bimg"]
    p0, p1 = shape
    fy, fx = best["dy_int"] % p0, best["dx_int"] % p1
    cols = np.where(bm.any(0))[0]
    if cols.size < 8:
        return {"split_support": None, "split_note": "strip too narrow to split"}
    mid = int(np.median(np.repeat(cols, bm[:, cols].sum(0))))
    halves, at_full = [], []
    for lo, hi in ((0, mid), (mid, bm.shape[1])):
        sub = np.zeros_like(bm)
        sub[:, lo:hi] = bm[:, lo:hi]
        if sub.sum() < min_overlap // 2:
            return {"split_support": None,
                    "split_note": "one half has too little area to test"}
        surf, n = masked_ncc(pack, a_shape, np.where(sub, bi, 0.0), sub,
                             max(min_overlap // 3, 1000), shape=shape)
        pk = surface_peak(surf, n)
        v = surf[fy, fx]
        if pk is None or not np.isfinite(v):
            return {"split_support": None,
                    "split_note": "one half found no shift with enough overlap"}
        halves.append(pk)
        at_full.append(float(v))
    full = best["peak"]
    dist = [math.hypot(p["dy"] - best["dy"], p["dx"] - best["dx"])
            for p in halves]
    excess = max((h["peak"] - v) if d > SPLIT_SAME_PX else 0.0
                 for h, v, d in zip(halves, at_full, dist))
    return {
        "split_support": round(float(min(at_full) / max(full, 1e-6)), 3),
        "split_excess": round(float(excess), 4),
        "split_px": round(float(max(dist)), 2),
        "split_halves": [{"dy": p["dy_int"], "dx": p["dx_int"],
                          "ncc": round(p["peak"], 3), "at_full": round(v, 3)}
                         for p, v in zip(halves, at_full)],
        "split_note": "each half's NCC at the full solution, over the full NCC",
    }


def judge(res, thresholds=None):
    """Accept or reject, with the reason spelled out.

    Four independent conditions, and a strip has to pass all of them.  They
    fail in different ways on purpose: overlap catches a strip that landed
    somewhere the canvas does not reach, NCC catches content that does not
    match, PSR catches a match that is no better than the rest of the surface,
    and split-half support catches a match carried by one lucky patch.  A rail
    hit on the rotation search is also a rejection: if the best angle is the
    largest angle searched, the true angle is outside the search and the
    reported one is meaningless.
    """
    t = thresholds or {}
    ncc_t = t.get("ncc", ACCEPT_NCC)
    psr_t = t.get("psr", ACCEPT_PSR)
    sup_t = t.get("support", ACCEPT_SPLIT_SUPPORT)
    exc_t = t.get("excess", ACCEPT_SPLIT_EXCESS)
    ov_t = t.get("overlap", MIN_OVERLAP_MAP)
    ov_mm2_t = t.get("overlap_mm2", MIN_TRUSTED_OVERLAP_MM2)
    if res is None:
        return False, "no shift anywhere had enough overlap with the canvas"
    why = []
    if res["overlap_px"] < ov_t:
        why.append(f"overlap {res['overlap_px']} px < {ov_t}")
    # Measured, not assumed: below this the answer is inaccurate while still
    # scoring well on NCC and PSR.  See MIN_TRUSTED_OVERLAP_MM2.
    elif res["overlap_mm2"] < ov_mm2_t:
        why.append(f"overlap {res['overlap_mm2']:.1f} mm2 < "
                   f"{ov_mm2_t:.1f} mm2, below which the sweep in "
                   f"synthetic_split_check measures 2.6-15.4 px error "
                   f"at NCC up to 0.90 and PSR up to 22")
    if res["ncc"] < ncc_t:
        why.append(f"NCC {res['ncc']:.3f} < {ncc_t}")
    if res["psr"] < psr_t:
        why.append(f"PSR {res['psr']:.1f} < {psr_t}")
    if res.get("at_rail"):
        why.append(f"best angle {res['theta']:.2f} deg is on the search rail")
    if res.get("scale_at_rail"):
        why.append(f"best along-axis scale {res['sx']:.3f} is on the search rail")
    sup = res.get("split_support")
    if sup is None:
        why.append("split-half test could not be run")
    else:
        if sup < sup_t:
            why.append(f"half-strip support {sup:.2f} < {sup_t}")
        if res.get("split_excess", 0.0) > exc_t:
            why.append(f"a half prefers a shift {res['split_px']:.0f} px away "
                       f"by {res['split_excess']:.3f} NCC > {exc_t}")
    if why:
        return False, "; ".join(why)
    return True, (f"NCC {res['ncc']:.3f}, PSR {res['psr']:.1f}, "
                  f"half-support {sup:.2f}, overlap {res['overlap_mm2']:.1f} mm^2")


# ---------------------------------------------------------------------------
# 6. calibration: what does a NON-match score?
# ---------------------------------------------------------------------------
def read_pgm(path):
    """Minimal binary PGM reader (the press dataset is P5, 150x52, 8-bit)."""
    data = Path(path).read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"{path}: not a binary PGM")
    fields, i = [], 2
    while len(fields) < 3:
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while j < len(data) and not data[j:j + 1].isspace():
            j += 1
        fields.append(int(data[i:j]))
        i = j
    w, h, mx = fields
    i += 1
    a = np.frombuffer(data[i:i + w * h], dtype=np.uint8).reshape(h, w)
    return a.astype(np.float64)


def phase_randomise(img, mask, rng):
    """Same power spectrum, no correspondence.

    The sharpest null available without a second finger.  Flips and rotations
    change the local ridge geometry as well as the correspondence, so a low
    score against them is partly the mapper noticing that the ridges no longer
    look like ridges.  Phase randomisation leaves the magnitude spectrum
    exactly intact -- same ridge pitch, same orientation energy, same band
    fraction -- and destroys only the thing a true match is supposed to detect.
    """
    a = np.where(mask, img, 0.0)
    F = np.fft.fft2(a)
    G = np.fft.fft2(rng.normal(0.0, 1.0, a.shape))
    out = np.real(np.fft.ifft2(np.abs(F) * G / (np.abs(G) + 1e-12)))
    if mask.any():
        out = out / (out[mask].std() + 1e-12) * (a[mask].std() + 1e-12)
    return np.where(mask, out, 0.0)


def null_distribution(strips, dataset_dir=None, rng=None, verbose=True,
                      max_press=45):
    """Score the mapper against content that must not register.

    Four families, all measured with the identical search the real strips get:

      phase   phase-randomised strips (same power spectrum, no correspondence)
      flipud  strips mirrored across the swipe axis
      rot180  strips turned end for end
      press   the 45 press captures from the old averaging driver

    What this is: a bound on how high a NON-corresponding image can score
    through this search.  What it is NOT: an impostor distribution.  Every
    capture on this machine is the same finger of the same person, so nothing
    here says what a different finger would score, and no false-accept rate can
    be inferred from it.  It sets the floor a threshold has to clear; a real
    FAR needs other fingers.
    """
    rng = rng or np.random.default_rng(20260814)
    rows = []
    for i, a in enumerate(strips):
        for j, b in enumerate(strips):
            if i == j:
                continue
            variants = [
                ("phase", phase_randomise(b.img, b.cov, rng), b.cov),
                ("flipud", np.flipud(b.img), np.flipud(b.cov)),
                ("rot180", b.img[::-1, ::-1], b.cov[::-1, ::-1]),
            ]
            for kind, bi, bc in variants:
                r = register_strip(a.img, a.cov, bi, bc,
                                   scales=SCALE_GRID)
                if r is None:
                    continue
                ok, why = judge(r)
                rows.append({"kind": kind, "a": a.name, "b": b.name,
                             "ncc": r["ncc"], "psr": r["psr"],
                             "split_support": r.get("split_support"),
                             "split_excess": r.get("split_excess"),
                             "overlap_px": r["overlap_px"],
                             "would_accept": bool(ok), "reason": why})
    if dataset_dir:
        press = sorted(Path(dataset_dir).rglob("*.pgm"))[:max_press]
        a = strips[0]
        for p in press:
            img = read_pgm(p)
            cov = np.ones(img.shape, dtype=bool)
            r = register_strip(a.img, a.cov, img, cov, scales=SCALE_GRID)
            if r is None:
                continue
            ok, why = judge(r)
            rows.append({"kind": "press", "a": a.name, "b": p.name,
                         "ncc": r["ncc"], "psr": r["psr"],
                         "split_support": r.get("split_support"),
                         "split_excess": r.get("split_excess"),
                         "overlap_px": r["overlap_px"],
                         "would_accept": bool(ok), "reason": why})
    if not rows:
        return {"n": 0}
    ncc = np.array([r["ncc"] for r in rows])
    psr = np.array([r["psr"] for r in rows])
    passed = [r for r in rows if r["would_accept"]]
    summary = {
        "n": len(rows),
        "by_kind": {},
        "ncc_max": round(float(ncc.max()), 4),
        "ncc_p95": round(float(np.percentile(ncc, 95)), 4),
        "ncc_median": round(float(np.median(ncc)), 4),
        "psr_max": round(float(psr.max()), 2),
        "psr_median": round(float(np.median(psr)), 2),
        "n_would_be_accepted": len(passed),
        "accepted_examples": passed[:5],
        "thresholds": {"ncc": ACCEPT_NCC, "psr": ACCEPT_PSR,
                       "split_support": ACCEPT_SPLIT_SUPPORT,
                       "split_excess": ACCEPT_SPLIT_EXCESS},
    }
    for kind in ("phase", "flipud", "rot180", "press"):
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        summary["by_kind"][kind] = {
            "n": len(sel),
            "ncc_max": round(max(r["ncc"] for r in sel), 4),
            "ncc_median": round(float(np.median([r["ncc"] for r in sel])), 4),
            "psr_max": round(max(r["psr"] for r in sel), 2),
        }
    if verbose:
        print(f"null distribution: {summary['n']} non-matching registrations, "
              f"NCC max {summary['ncc_max']:.3f} (median "
              f"{summary['ncc_median']:.3f}), PSR max {summary['psr_max']:.1f}")
        for k, v in summary["by_kind"].items():
            print(f"    {k:<7} n={v['n']:<3} NCC max {v['ncc_max']:.3f}  "
                  f"median {v['ncc_median']:.3f}  PSR max {v['psr_max']:.1f}")
    return summary


def assess_scale(a, b, theta, scales=None, min_overlap=MIN_OVERLAP_MAP):
    """Does along-axis scale matter?  Measured answer: yes.

    The prior argument was that it should not.  A strip is built from
    frame-to-frame displacements MEASURED in pixels, so it is metric by
    construction, and swipe speed changes only how many frames cover a given
    length of finger, not how long that length is.  Speed could then only enter
    through a systematic bias in the displacement estimate -- motion blur
    pulling the correlation peak backwards, say.

    That argument is clean, plausible, and contradicted by this function.
    Sweeping anisotropic scale along the long axis at the best angle, on the
    four real strip pairs:

        165428#2 <- 165428#1   best 0.97   NCC 0.580 -> 0.631   (+0.052)
        170350   <- 165457     best 1.01   NCC 0.737 -> 0.749   (+0.013)
        165428#1 <- 170350     best 0.97   NCC 0.624 -> 0.745   (+0.121)
        165428#2 <- 165428#3   best 1.00   NCC 0.691 -> 0.691   ( 0.000)

    Two pairs want 3% of compression and one gains 0.121 NCC, which is not a
    resolution limit -- it is the largest single improvement anything in this
    file produced.  The curves are smooth and singly-peaked, so this is a real
    optimum and not a rail artefact.

    The likely mechanism is not swipe speed but skin: the finger is soft, the
    sensor drags on it, and the skin stretches differently depending on how the
    finger is loaded.  A rigid model cannot reconcile two swipes that stretched
    the skin differently, and 3% over a 330 px strip is 10 px -- which is what
    the residual pose inconsistency was, before scale was searched.

    So scale is now SEARCHED in `register_strip`, and this function remains as
    the diagnostic that justifies it and as the way to check the range is wide
    enough.
    """
    if scales is None:
        scales = np.arange(0.94, 1.0601, 0.01)
    curve = []
    for s in scales:
        r = register_strip(a.img, a.cov, b.img, b.cov,
                           angles=[theta], refine=False, split_half=False,
                           min_overlap=min_overlap, sx=float(s))
        curve.append((round(float(s), 3), r["ncc"] if r else None))
    ok = [(s, v) for s, v in curve if v is not None]
    best = max(ok, key=lambda t: t[1]) if ok else (None, None)
    at1 = dict(ok).get(1.0)
    return {
        "curve": curve,
        "best_scale": best[0],
        "best_ncc": best[1],
        "ncc_at_scale_1": at1,
        "gain_over_scale_1": (round(best[1] - at1, 4)
                              if (at1 is not None and best[1] is not None)
                              else None),
    }


# ---------------------------------------------------------------------------
# 7. building the map
# ---------------------------------------------------------------------------
def frame_gain_offsets(frames, px, py):
    """Per-frame gain and offset within one swipe, in strip space.

    Mirrors swipe_assemble.equalise, which returns only the gains because it
    applies them itself.  The map needs the (gain, offset) pairs as numbers,
    because it applies them to the raw frames before warping them onto the
    canvas -- the whole point being that a frame is resampled once, so it can
    never be handed the strip's already-resampled pixels.
    """
    values, valid, _, _ = build_stack(frames, px, py, subpixel=True)
    out = np.zeros((len(frames), 2))
    out[0] = (1.0, 0.0)
    prev = values[0]
    for i in range(1, len(frames)):
        m = valid[i] & valid[i - 1]
        if m.sum() < 2500:
            out[i] = out[i - 1]
        else:
            a = prev[m]
            b = values[i][m]
            vb = b.var()
            g = float(a.std() / math.sqrt(vb)) if vb > 1e-12 else 1.0
            g = float(np.clip(g, 0.5, 2.0))
            o = float(a.mean() - g * b.mean())
            out[i] = (g, o)
        prev = values[i] * out[i, 0] + out[i, 1]
        prev *= valid[i]
    return out


def frame_pose(strip, i):
    """Sensor coordinates of frame i -> canvas coordinates.

    One matrix, so one resampling.  Left to right: place the frame inside its
    strip (a pure translation, possibly fractional), then place the strip on
    the canvas (rotation, optional scale, translation).
    """
    return strip.pose @ translation(strip.py[i], strip.px[i])


def map_bounds(strips, border=3):
    """Canvas bounding box over every frame of every accepted strip."""
    pts = []
    for s in strips:
        for i in range(len(s.frames)):
            F = frame_pose(s, i)
            pts.append(apply_to_points(F, [
                (border, border), (border, FRAME_W - 1 - border),
                (FRAME_H - 1 - border, border),
                (FRAME_H - 1 - border, FRAME_W - 1 - border)]))
    pts = np.concatenate(pts, axis=0)
    y0, x0 = math.floor(pts[:, 0].min()), math.floor(pts[:, 1].min())
    y1, x1 = math.ceil(pts[:, 0].max()), math.ceil(pts[:, 1].max())
    return int(y0), int(x0), int(y1 - y0) + 1, int(x1 - x0) + 1


def place_frames(strips, border=3, dtype=np.float32):
    """Warp every frame of every accepted strip onto one canvas.

    Returns (values, valid, sharp, centre, owner) where owner[i] is the index
    of the strip that contributed plane i.  Each plane is one frame, warped
    exactly once from sensor space, so the blends below see the same kind of
    input the single-swipe assembler saw -- which is what makes their numbers
    comparable to it.
    """
    y0, x0, ch, cw = map_bounds(strips, border)
    shift = translation(-y0, -x0)
    n = sum(len(s.frames) for s in strips)
    values = np.zeros((n, ch, cw), dtype=dtype)
    valid = np.zeros((n, ch, cw), dtype=bool)
    sharp = np.zeros((n, ch, cw), dtype=dtype)
    centre = np.zeros((n, ch, cw), dtype=dtype)
    owner = np.zeros(n, dtype=np.int32)

    cen = np.outer(_raised_cosine(FRAME_H, 10), _raised_cosine(FRAME_W, 24))
    cen = np.maximum(cen, 1e-3)

    k = 0
    for si, s in enumerate(strips):
        go = frame_gain_offsets(s.frames, s.px, s.py)
        G, O = s.gain
        for i, f in enumerate(s.frames):
            img = G * (go[i, 0] * f + go[i, 1]) + O
            q = ridge_quality_map(f)
            F = shift @ frame_pose(s, i)
            cor = transform_corners(F, (FRAME_H, FRAME_W))
            ay0 = max(0, int(math.floor(cor[:, 0].min())) - 1)
            ax0 = max(0, int(math.floor(cor[:, 1].min())) - 1)
            ay1 = min(ch, int(math.ceil(cor[:, 0].max())) + 2)
            ax1 = min(cw, int(math.ceil(cor[:, 1].max())) + 2)
            if ay1 <= ay0 or ax1 <= ax0:
                k += 1
                continue
            local = translation(-ay0, -ax0) @ F
            wi, val, wq, wc = warp(img, local, (ay1 - ay0, ax1 - ax0),
                                   border=border, extra=(q, cen))
            values[k, ay0:ay1, ax0:ax1] = wi
            valid[k, ay0:ay1, ax0:ax1] = val
            sharp[k, ay0:ay1, ax0:ax1] = np.maximum(wq, 0.0) * val
            centre[k, ay0:ay1, ax0:ax1] = np.maximum(wc, 1e-3) * val
            owner[k] = si
            k += 1
    return values, valid, sharp, centre, owner, (y0, x0)


def swipe_gain(canvas_img, canvas_cov, strip, pose):
    """Least-squares gain and offset matching a strip to the canvas.

    Contact pressure differs between swipes far more than it does between
    consecutive frames of one swipe, so without this the map gets visible
    banding at swipe boundaries and the quality-weighted blends start choosing
    contributors by brightness instead of by sharpness.
    """
    ch, cw = canvas_img.shape
    wi, val, wm = warp(strip.img, pose, (ch, cw), border=1,
                       extra=(strip.cov.astype(float),))
    m = canvas_cov & val & (wm > 0.99)
    if m.sum() < 1500:
        return (1.0, 0.0)
    a = canvas_img[m]
    b = wi[m]
    vb = b.var()
    g = float(a.std() / math.sqrt(vb)) if vb > 1e-12 else 1.0
    g = float(np.clip(g, 0.4, 2.5))
    o = float(a.mean() - g * b.mean())
    return (g, o)


def blend_map(strips, modes=("sharp_win",), border=3):
    """Place all frames and blend them, once per requested mode."""
    values, valid, sharp, centre, owner, origin = place_frames(strips, border)
    out = {}
    for mode in modes:
        img, cov = blend(values, valid, sharp, centre, mode)
        out[mode] = (np.asarray(img, np.float64), cov)
    # Attribution: which strip won each pixel, and which strips merely covered
    # it.  Both are needed -- "this swipe supplied 40% of the map" means
    # something different from "this swipe covered 60% of it", and only the
    # first is about what the map is made of.
    #
    # This is computed under the sharp_win rule (identical expression to
    # swipe_assemble.blend), ALWAYS, whatever `modes` was asked for.  So the
    # won_fraction column describes the sharp_win map specifically; for a
    # mixing blend like mean or median no single strip "wins" a pixel at all
    # and the column should be read as "would have won".
    score = np.where(valid, centre * sharp, -1.0)
    win = score.argmax(0)
    cov_any = valid.any(0)
    winner = np.where(cov_any, owner[win], -1)
    per_strip_cov = []
    for si in range(len(strips)):
        planes = np.where(owner == si)[0]
        per_strip_cov.append(valid[planes].any(0) if planes.size else
                             np.zeros_like(cov_any))
    return {
        "blends": out,
        "winner": winner,
        "cov": cov_any,
        "per_strip_cov": per_strip_cov,
        "origin": origin,
        "n_planes": len(values),
        "stack": (values, valid, sharp, centre, owner),
    }


def strip_route_map(strips, canvas_shape, origin):
    """The other way to build the map: warp each FINISHED strip and blend those.

    Kept only so the report can put a number on the cost of the second
    resampling, which is the reason the frame route exists.  Same blend rule,
    same poses, same quality maps -- the only difference is that the pixels
    have been through the assembler's interpolator before they meet this one.
    """
    y0, x0 = origin
    ch, cw = canvas_shape
    shift = translation(-y0, -x0)
    n = len(strips)
    values = np.zeros((n, ch, cw), dtype=np.float32)
    valid = np.zeros((n, ch, cw), dtype=bool)
    sharp = np.zeros((n, ch, cw), dtype=np.float32)
    centre = np.zeros((n, ch, cw), dtype=np.float32)
    for i, s in enumerate(strips):
        G, O = s.gain
        q = ridge_quality_map(np.where(s.cov, s.img, 0.0))
        cen = np.ones_like(s.img)
        wi, val, wq, wc, wm = warp(G * s.img + O, shift @ s.pose, (ch, cw),
                                   border=1,
                                   extra=(q, cen, s.cov.astype(float)))
        m = val & (wm > 0.99)
        values[i] = wi * m
        valid[i] = m
        sharp[i] = np.maximum(wq, 0.0) * m
        centre[i] = np.maximum(wc, 1e-3) * m
    return blend(values, valid, sharp, centre, "sharp_win")


# ---------------------------------------------------------------------------
# 8. the enrolment loop
# ---------------------------------------------------------------------------
def pose_distance(P, Q, shape):
    """How far apart two candidate poses are, in pixels at the strip's corners.

    Comparing matrices entry by entry would weight rotation and translation
    arbitrarily.  What matters is where the strip actually lands, so the
    distance is the largest displacement between the two poses over the
    strip's four corners.
    """
    a = transform_corners(P, shape)
    b = transform_corners(Q, shape)
    return float(np.hypot(*(a - b).T).max())


AGREE_PX = 6.0          # two poses this close are the same answer


def register_to_map(strip, accepted, thresholds=None):
    """Register a candidate against every accepted strip, and cross-check.

    Registering against the blended canvas was the first design and it is worse.
    Measured on the six real strips: 20260814-170350 registers against strip
    20260814-165457 at NCC 0.737 with PSR 17.1, but against a canvas that
    already contains 165457 it falls to 0.527 and gets rejected.  The canvas is
    a mosaic of contributors from different swipes with seams between them, and
    correlating against it dilutes the very correspondence being looked for.

    Registering pairwise also buys a check that no canvas can give: when two
    or more accepted strips each have enough overlap with the candidate, they
    each imply a pose, and those poses must agree.  Cycle consistency of that
    kind is the standard way to validate a mosaic, and here it comes for free.
    """
    cands = []
    for t in accepted:
        r = register_strip(t.img, t.cov, strip.img, strip.cov,
                           scales=SCALE_GRID)
        if r is None:
            continue
        ok, why = judge(r, thresholds)
        cands.append({"via": t.name, "target": t, "res": r,
                      "ok": bool(ok), "why": why,
                      "pose_ref": t.pose @ r["pose"]})
    if not cands:
        return None
    good = [c for c in cands if c["ok"]]
    if not good:
        best = max(cands, key=lambda c: c["res"]["psr"])
        return {"accepted": False, "best": best, "candidates": cands,
                "n_agree": 0, "n_tested": len(cands),
                "reason": f"best link (via {best['via']}): {best['why']}"}
    best = max(good, key=lambda c: c["res"]["psr"])
    agree, disagree = [], []
    for c in good:
        if c is best:
            continue
        d = pose_distance(best["pose_ref"], c["pose_ref"], strip.img.shape)
        (agree if d <= AGREE_PX else disagree).append((c["via"], round(d, 2)))
    return {
        "accepted": True,
        "best": best,
        "candidates": cands,
        "n_agree": len(agree),
        "n_tested": len(cands),
        "agree": agree,
        "disagree": disagree,
        "reason": (f"via {best['via']}: {best['why']}" +
                   (f"; {len(agree)} other strip(s) agree" if agree else "") +
                   (f"; {len(disagree)} disagree {disagree}" if disagree else "")),
    }


def build_finger_map(strips, thresholds=None, max_passes=3, verbose=True):
    """Register strips into one map, one at a time, rejecting what does not fit.

    Order matters and is chosen, not accidental: the seed is the strip with the
    most quality-weighted covered area, and candidates are tried best first, so
    the map is at its most informative when the weakest strips are judged
    against it.

    Several passes, because a strip can be rejected for genuinely having too
    little overlap with what has been accepted so far, and then register
    cleanly once a strip nearer to it has been added.  That is the honest way
    to handle a swipe from the far end of the finger: not by lowering the
    threshold, but by giving it something to overlap with.  Rejection is sticky
    only within a pass.
    """
    order = sorted(strips, key=lambda s: -s.rank())
    seed = order[0]
    seed.pose = np.eye(3)
    seed.gain = (1.0, 0.0)
    seed.conf = {"seed": True, "pass": 0}
    accepted = [seed]
    pending = list(order[1:])
    trials = []
    if verbose:
        print(f"seed: {seed.name} "
              f"({seed.report['coverage_mm2']:.1f} mm^2, band {seed.band:.3f})")

    for p in range(1, max_passes + 1):
        if not pending:
            break
        progressed = False
        still = []
        for s in pending:
            out = register_to_map(s, accepted, thresholds)
            rec = {"strip": s.name, "pass": p,
                   "accepted": bool(out and out["accepted"]),
                   "reason": out["reason"] if out else
                             "no accepted strip had enough overlap with it",
                   "n_tested": out["n_tested"] if out else 0,
                   "n_agree": out["n_agree"] if out else 0}
            if out:
                r = out["best"]["res"]
                rec.update({"via": out["best"]["via"],
                            "theta": r["theta"], "ncc": r["ncc"],
                            "psr": r["psr"], "overlap_px": r["overlap_px"],
                            "overlap_mm2": r["overlap_mm2"],
                            "split_support": r.get("split_support"),
                            "split_excess": r.get("split_excess"),
                            "split_px": r.get("split_px"),
                            "at_rail": r.get("at_rail"),
                            "links": [{"via": c["via"], "ncc": c["res"]["ncc"],
                                       "psr": c["res"]["psr"],
                                       "accepted": c["ok"]}
                                      for c in out["candidates"]]})
            trials.append(rec)
            if verbose:
                print(f"  pass {p}: {s.name:<22} "
                      f"{'ACCEPT' if rec['accepted'] else 'REJECT'}  "
                      f"{rec['reason']}")
            if not rec["accepted"]:
                still.append(s)
                continue
            b = out["best"]
            t = b["target"]
            s.pose = b["pose_ref"]
            g, o = swipe_gain(t.img, t.cov, s, b["res"]["pose"])
            Gt, Ot = t.gain
            s.gain = (Gt * g, Gt * o + Ot)
            s.conf = rec
            accepted.append(s)
            progressed = True
        pending = still
        if not progressed:
            break
    return accepted, pending, trials


def link_graph(accepted, thresholds=None, verbose=True):
    """Every accepted pairwise link between accepted strips.

    The enrolment loop only ever needs ONE good link per strip, but the map is
    only as good as the whole graph, so this recomputes all of them.  Cost is
    n(n-1)/2 registrations, which for a realistic enrolment of 5-8 swipes is
    10-28 searches at about 1.5 s each, once, at enrolment.
    """
    links = []
    for i in range(len(accepted)):
        for j in range(i + 1, len(accepted)):
            a, b = accepted[i], accepted[j]
            r = register_strip(a.img, a.cov, b.img, b.cov,
                               scales=SCALE_GRID)
            if r is None:
                continue
            ok, why = judge(r, thresholds)
            if not ok:
                continue
            links.append({"i": i, "j": j, "rel": r["pose"],
                          "ncc": r["ncc"], "psr": r["psr"],
                          "theta": r["theta"],
                          "overlap_px": r["overlap_px"], "why": why})
    if verbose:
        print(f"link graph: {len(links)} accepted links between "
              f"{len(accepted)} strips")
    return links


def link_inconsistency(accepted, links):
    """Corner distance between each link's prediction and the current poses."""
    out = []
    for L in links:
        i, j = L["i"], L["j"]
        pred = accepted[i].pose @ L["rel"]
        d = pose_distance(pred, accepted[j].pose, accepted[j].img.shape)
        out.append({"i": accepted[i].name, "j": accepted[j].name,
                    "psr": L["psr"], "ncc": L["ncc"], "px": round(d, 2)})
    return out


def _project_similarity(A):
    """Nearest shear-free transform to an averaged affine matrix.

    Averaging matrices entrywise produces a little shear, which is not a motion
    the finger can make and would accumulate over sweeps, so it is projected
    out.  What is NOT projected out is the anisotropic scale: an earlier version
    forced the average back to a pure rotation+translation, which silently threw
    away the along-axis scale the registration had just measured and put the 10
    px inconsistency straight back.  Rotation and both scales are kept;
    only shear is discarded.
    """
    th, sy, sx = decompose(A)
    out = similarity(th, sx=sx, sy=sy)
    out[:2, 2] = A[:2, 2]
    return out


def refine_poses(accepted, links, iters=60, verbose=True):
    """Make the pose graph globally consistent instead of merely chained.

    Greedy chaining takes one link per strip and inherits every error upstream
    of it.  Measured on the real strips accepted here, the links are
    individually excellent -- NCC 0.73-0.75, PSR 14-17, every acceptance test
    passed -- and still mutually inconsistent: strip 20260814-165457 lands 13.3
    px apart depending on which accepted strip you route it through.  That
    inconsistency IS the smeared-canvas failure mode, arriving through the front
    door with good confidence numbers.

    (Before the along-axis scale search was added, the same measurement read
    27.9 px and averaging only got it to 10.1 px.  Most of that was model error,
    not estimation noise, which is why searching scale helped far more than any
    amount of averaging could.)

    The fix is the standard one, pose averaging: hold the seed fixed and
    repeatedly replace every other strip's pose with the PSR-weighted average
    of what all its links predict, projected back onto a rigid transform.  It
    is coordinate descent on the sum of squared link residuals, it needs no
    derivatives, and on this data it converges in a few dozen sweeps.

    Reported honestly: this reduces inconsistency, it does not abolish it.  The
    residual is 4.3 px, about half a ridge pitch, and it is a real disagreement
    between real measurements.  It is also the reason the per-pixel frame blend
    does not yet beat the per-strip blend -- see the module docstring.  The
    things that shrink it further are a better deformation model, better strips,
    and more of them, in that order.

    Note that pose averaging is applied AFTER the anisotropic scale search, and
    `_project_similarity` deliberately preserves that scale.  An earlier version
    projected onto a pure rotation+translation and silently discarded it, which
    put the whole 10 px error straight back.
    """
    if len(accepted) < 3 or not links:
        return {"applied": False, "reason": "fewer than 3 strips or no links"}
    before = link_inconsistency(accepted, links)
    by_node = {k: [] for k in range(len(accepted))}
    for L in links:
        by_node[L["j"]].append((L["i"], L["rel"], L["psr"], False))
        by_node[L["i"]].append((L["j"], L["rel"], L["psr"], True))
    hist = []
    for it in range(iters):
        moved = 0.0
        for k in range(1, len(accepted)):
            preds, ws = [], []
            for (other, rel, psr, inverse) in by_node[k]:
                P = accepted[other].pose
                preds.append(P @ np.linalg.inv(rel) if inverse else P @ rel)
                ws.append(psr)
            if not preds:
                continue
            w = np.asarray(ws, float)
            w = w / w.sum()
            avg = sum(wi * P for wi, P in zip(w, preds))
            new = _project_similarity(avg)
            moved = max(moved, pose_distance(new, accepted[k].pose,
                                             accepted[k].img.shape))
            accepted[k].pose = new
        hist.append(round(moved, 3))
        if moved < 0.01:
            break
    after = link_inconsistency(accepted, links)
    res = {
        "applied": True,
        "iterations": len(hist),
        "max_step_history": hist[:10],
        "before": {"max_px": round(max(d["px"] for d in before), 2),
                   "median_px": round(float(np.median([d["px"] for d in before])), 2)},
        "after": {"max_px": round(max(d["px"] for d in after), 2),
                  "median_px": round(float(np.median([d["px"] for d in after])), 2)},
        "links_before": before,
        "links_after": after,
    }
    if verbose:
        print(f"pose averaging: link inconsistency max "
              f"{res['before']['max_px']:.1f} -> {res['after']['max_px']:.1f} px, "
              f"median {res['before']['median_px']:.1f} -> "
              f"{res['after']['median_px']:.1f} px "
              f"({res['iterations']} sweeps)")
    return res


def map_report(accepted, rejected, trials, modes=BLEND_MODES, verbose=True):
    """Assemble the final map and measure it."""
    mp = blend_map(accepted, modes=tuple(modes))
    values, valid, sharp, centre, owner = mp["stack"]
    cov = mp["cov"]
    ch, cw = cov.shape

    rep = {
        "n_strips_offered": len(accepted) + len(rejected),
        "n_strips_accepted": len(accepted),
        "n_strips_rejected": len(rejected),
        "rejection_rate": round(len(rejected) / max(len(accepted) + len(rejected), 1), 3),
        "rejected": [s.name for s in rejected],
        "trials": trials,
        "strips": [],
        "canvas": {
            "w": int(cw), "h": int(ch),
            "mm": [round(cw * MM_PER_PX, 2), round(ch * MM_PER_PX, 2)],
            "coverage_px": int(cov.sum()),
            "coverage_mm2": round(float(cov.sum()) * MM_PER_PX ** 2, 2),
            "coverage_frames": round(float(cov.sum()) / (FRAME_W * FRAME_H), 2),
            "bbox_frames": round(ch * cw / (FRAME_W * FRAME_H), 2),
            "n_frames": int(len(values)),
        },
    }

    # Per-strip contribution.  "won" is what the map is actually made of;
    # "covered" is what the strip could have supplied; "exclusive" is the area
    # that would be missing if the strip had been rejected, which is the number
    # that says whether a swipe was worth taking.
    for si, s in enumerate(accepted):
        c = mp["per_strip_cov"][si]
        others = np.zeros_like(c)
        for sj in range(len(accepted)):
            if sj != si:
                others |= mp["per_strip_cov"][sj]
        won = int((mp["winner"] == si).sum())
        theta, sy, sx = decompose(s.pose)
        rep["strips"].append({
            "name": s.name,
            "n_frames": len(s.frames),
            "pose": {"theta_deg": round(theta, 3),
                     "ty": round(float(s.pose[0, 2]), 2),
                     "tx": round(float(s.pose[1, 2]), 2),
                     "scale_y": round(sy, 4), "scale_x": round(sx, 4)},
            "gain": [round(s.gain[0], 3), round(s.gain[1], 1)],
            "confidence": {k: s.conf.get(k) for k in
                           ("ncc", "psr", "split_px", "overlap_mm2", "pass")}
            if s.conf else {"seed": True},
            "covered_px": int(c.sum()),
            "covered_mm2": round(float(c.sum()) * MM_PER_PX ** 2, 2),
            "exclusive_px": int((c & ~others).sum()),
            "exclusive_mm2": round(float((c & ~others).sum()) * MM_PER_PX ** 2, 2),
            "won_px": won,
            "won_fraction": round(won / max(int(cov.sum()), 1), 3),
            "strip_band_median": round(s.band, 4),
        })

    # Baseline: the same sliding tile over every frame that went into the map.
    allv = np.ones((FRAME_H, FRAME_W), dtype=bool)
    pool_b, pool_a = [], []
    for s in accepted:
        for f in s.frames:
            m = window_metrics(f, allv)
            if m.get("n_windows"):
                pool_b.extend(m["_band"])
                pool_a.extend(m["_amp"])
    base = summarise_tiles(pool_b, pool_a)
    rep["single_frame_baseline"] = strip_arrays(base)

    rep["blends"] = {}
    for mode, (img, c) in mp["blends"].items():
        m = strip_arrays(window_metrics(img, c))
        m["overlap_retention"] = overlap_retention(img, values, valid)
        m["vs_single_frame"] = (round(m["band_median"] / base["band_median"], 3)
                                if m.get("n_windows") else None)
        rep["blends"][mode] = m

    # The double-resample control.
    if len(accepted) > 1:
        simg, scov = strip_route_map(accepted, cov.shape, mp["origin"])
        m = strip_arrays(window_metrics(simg, scov))
        m["vs_single_frame"] = (round(m["band_median"] / base["band_median"], 3)
                                if m.get("n_windows") else None)
        rep["strip_route"] = m
        mp["blends"]["strip_route"] = (simg, scov)

    return rep, mp


# ---------------------------------------------------------------------------
# 9. validation with ground truth
# ---------------------------------------------------------------------------
def naive_masked_ncc(A, MA, B, MB, min_overlap, shifts):
    """Reference masked NCC, computed one shift at a time by brute force.

    Exists only to check `masked_ncc`.  The FFT version computes all six
    running sums through six correlations, which is fast and completely opaque;
    this one computes them by literally selecting the overlapping pixels, which
    is slow and obviously correct.  If they disagree, the fast one is wrong.
    """
    Ha, Wa = A.shape
    Hb, Wb = B.shape
    out = {}
    for (dy, dx) in shifts:
        ys = np.arange(Ha)
        xs = np.arange(Wa)
        by = ys - dy
        bx = xs - dx
        vy = (by >= 0) & (by < Hb)
        vx = (bx >= 0) & (bx < Wb)
        if not vy.any() or not vx.any():
            out[(dy, dx)] = np.nan
            continue
        a = A[np.ix_(ys[vy], xs[vx])]
        ma = MA[np.ix_(ys[vy], xs[vx])]
        b = B[np.ix_(by[vy], bx[vx])]
        mb = MB[np.ix_(by[vy], bx[vx])]
        m = ma & mb
        n = int(m.sum())
        if n < min_overlap:
            out[(dy, dx)] = np.nan
            continue
        av, bv = a[m], b[m]
        av = av - av.mean()
        bv = bv - bv.mean()
        den = math.sqrt(float((av * av).sum()) * float((bv * bv).sum()))
        out[(dy, dx)] = float((av * bv).sum() / den) if den > 1e-9 else np.nan
    return out


def _rand_field(shape, rng, pitch=9.0):
    """A ridge-like random field: band-limited noise at the sensor's pitch."""
    h, w = shape
    n = rng.normal(0.0, 1.0, (h, w))
    F = np.fft.fft2(n)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.hypot(fy, fx)
    band = np.exp(-((r - 1.0 / pitch) ** 2) / (2 * (0.25 / pitch) ** 2))
    return np.real(np.fft.ifft2(F * band))


def selftest(verbose=True):
    """Check the machinery against independent computations of the same thing.

    Every check here compares one implementation against a DIFFERENT one, not
    against a stored number.  A stored number only tells you the code still does
    what it did; an independent computation tells you what it does is right.
    """
    rng = np.random.default_rng(4242)
    checks = []

    def rec(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")

    # -- masked NCC against the brute-force loop ---------------------------
    A = _rand_field((60, 90), rng)
    B = _rand_field((40, 70), rng)
    MA = np.ones(A.shape, bool)
    MA[:5, :] = False                       # a non-trivial mask on both sides
    MB = np.ones(B.shape, bool)
    MB[:, :6] = False
    shape = _fft_shape(A.shape, B.shape)
    pack = _fft_pack(A, MA, shape)
    surf, n = masked_ncc(pack, A.shape, B, MB, 200, shape=shape)
    shifts = [(dy, dx) for dy in (-11, -3, 0, 7, 15) for dx in (-9, 0, 5, 21)]
    ref = naive_masked_ncc(A, MA, B, MB, 200, shifts)
    err = 0.0
    for (dy, dx), v in ref.items():
        got = surf[dy % shape[0], dx % shape[1]]
        if np.isnan(v) and np.isnan(got):
            continue
        err = max(err, abs(float(got) - float(v)))
    rec("masked NCC == brute force", err < 1e-9,
        f"max abs difference over {len(shifts)} shifts = {err:.3e}")

    # -- overlap counts against the brute-force loop -----------------------
    dy, dx = 7, 5
    ys = np.arange(A.shape[0]) - dy
    xs = np.arange(A.shape[1]) - dx
    vy = (ys >= 0) & (ys < B.shape[0])
    vx = (xs >= 0) & (xs < B.shape[1])
    want = int((MA[np.ix_(np.where(vy)[0], np.where(vx)[0])] &
                MB[np.ix_(ys[vy], xs[vx])]).sum())
    got = int(round(float(n[dy % shape[0], dx % shape[1]])))
    rec("overlap count == brute force", want == got, f"{got} vs {want} px")

    # -- a planted shift is recovered exactly ------------------------------
    big = _rand_field((80, 140), rng)
    ty, tx = -13, 22
    sub = big[20:60, 30:110]
    MAb = np.ones(big.shape, bool)
    MBb = np.ones(sub.shape, bool)
    pk = surface_peak(*masked_ncc(_fft_pack(big, MAb, _fft_shape(big.shape, sub.shape)),
                                  big.shape, sub, MBb, 500,
                                  shape=_fft_shape(big.shape, sub.shape)))
    rec("planted shift recovered", pk["dy_int"] == 20 and pk["dx_int"] == 30,
        f"found dy={pk['dy_int']} dx={pk['dx_int']}, want dy=20 dx=30, "
        f"NCC {pk['peak']:.4f} PSR {pk['psr']:.1f}")

    # -- erosion against a naive minimum filter ----------------------------
    # Density matters here.  At 75% true a 7x7 all-true window essentially
    # never occurs, so both sides return an empty mask and the check passes
    # while testing nothing -- which is exactly what the first version of this
    # test did.  95% true at r=2 leaves roughly a third of the pixels standing,
    # so the two implementations have something to disagree about.
    m = rng.random((40, 55)) > 0.05
    r = 2
    ref_e = np.zeros_like(m)
    for y in range(r, m.shape[0] - r):
        for x in range(r, m.shape[1] - r):
            ref_e[y, x] = m[y - r:y + r + 1, x - r:x + r + 1].all()
    got_e = erode(m, r)
    frac = float(ref_e.sum()) / ref_e.size
    rec("erode == naive minimum filter",
        np.array_equal(got_e, ref_e) and 0.05 < frac < 0.95,
        f"{int(got_e.sum())} px survive, naive gives {int(ref_e.sum())} "
        f"({frac:.0%} of the image, so the check is not vacuous)")

    # -- bicubic warp: integer translation is exact ------------------------
    img = _rand_field((50, 70), rng)
    w1, v1 = warp(img, translation(4, -6), img.shape, border=2)
    ref_t = np.zeros_like(img)
    ref_t[4:, :64] = img[:46, 6:]
    d = np.abs(w1 - ref_t)[v1].max()
    rec("integer-translation warp is exact", d < 1e-9, f"max error {d:.3e}")

    # -- rotate and rotate back recovers the image -------------------------
    cy, cx = (img.shape[0] - 1) / 2, (img.shape[1] - 1) / 2
    R = similarity(11.0, cy=cy, cx=cx)
    fwd, vf = warp(img, R, img.shape, border=2)
    back, vb = warp(fwd, np.linalg.inv(R), img.shape, border=2)
    mm = vb & erode(vf, 6)
    c = float(np.corrcoef(back[mm], img[mm])[0, 1]) if mm.sum() > 100 else 0.0
    rec("rotate then unrotate round-trips", c > 0.99,
        f"correlation {c:.5f} over {int(mm.sum())} px")

    # -- decompose inverts similarity --------------------------------------
    th, sxx = -13.5, 1.04
    dth, dsy, dsx = decompose(similarity(th, sx=sxx, sy=1.0, cy=3, cx=7))
    rec("decompose inverts similarity",
        abs(dth - th) < 1e-6 and abs(dsx - sxx) < 1e-6 and abs(dsy - 1.0) < 1e-6,
        f"theta {dth:.6f} (want {th}), sx {dsx:.6f} (want {sxx}), sy {dsy:.6f}")

    # -- rotation SIGN: the estimator must undo the applied rotation -------
    # This is the check the sign convention comment promises.  Rotate a
    # ridge-like field by +8 deg, register the rotated copy against the
    # original, and the recovered pose angle must be -8.
    base = _rand_field((70, 200), rng, pitch=9.0)
    cov0 = np.ones(base.shape, bool)
    Rb = similarity(8.0, cy=(base.shape[0] - 1) / 2, cx=(base.shape[1] - 1) / 2)
    rot, vrot = warp(base, Rb, base.shape, border=2)
    r = register_strip(base, erode(cov0, 6), rot, erode(vrot, 6),
                       angles=np.arange(-12, 12.01, 1.0), split_half=False,
                       min_overlap=3000)
    rec("rotation sign convention", r is not None and abs(r["theta"] + 8.0) <= 0.5,
        f"applied +8.00 deg, estimator returned {r['theta']:+.2f} deg "
        f"(want -8.00), NCC {r['ncc']:.3f}" if r else "registration failed")

    # -- a pure translation is recovered through the full search -----------
    # shifted[y, x] = base[y - 9, x + 17], i.e. the CONTENT moved by (+9, -17).
    # The pose maps strip coordinates back into canvas coordinates, so it has to
    # UNDO that motion: the expected answer is (-9, +17).  The first version of
    # this test asserted (+9, -17) and "failed" against correct code.
    shifted = np.zeros_like(base)
    moved_y, moved_x = 9, -17
    want_dy, want_dx = -moved_y, -moved_x
    shifted[9:, :183] = base[:61, 17:]
    vs = np.zeros(base.shape, bool)
    vs[9:, :183] = True
    r2 = register_strip(base, erode(cov0, 6), shifted, erode(vs, 6),
                        angles=np.arange(-4, 4.01, 1.0), split_half=False,
                        min_overlap=3000)
    ok2 = (r2 is not None and abs(r2["dy"] - want_dy) < 0.6 and
           abs(r2["dx"] - want_dx) < 0.6 and abs(r2["theta"]) < 0.6)
    rec("pure translation recovered through full search", ok2,
        f"content moved ({moved_y:+d},{moved_x:+d}) so want dy={want_dy:+d} "
        f"dx={want_dx:+d} theta=0; got dy={r2['dy']:+.2f} "
        f"dx={r2['dx']:+.2f} theta={r2['theta']:+.2f}" if r2 else "failed")

    # -- non-matching content is REJECTED ----------------------------------
    # The overlap threshold is lowered to match the search, so that the
    # rejection has to be earned on NCC and PSR -- on the CONTENT not matching
    # -- rather than won for free because the test buffer is small.
    other = _rand_field(base.shape, rng, pitch=9.0)
    r3 = register_strip(base, erode(cov0, 6), other, erode(cov0, 6),
                        angles=np.arange(-12, 12.01, 2.0), min_overlap=3000)
    ok3, why3 = judge(r3, {"overlap": 3000})
    rec("unrelated ridge field is rejected", not ok3,
        f"NCC {r3['ncc']:.3f} PSR {r3['psr']:.1f} -> {why3}" if r3 else "no result")

    # -- pose_distance is a real distance ----------------------------------
    P = similarity(3.0, ty=2, tx=-5, cy=20, cx=60)
    rec("pose_distance zero iff identical",
        pose_distance(P, P, (52, 150)) < 1e-9 and
        pose_distance(P, np.eye(3), (52, 150)) > 1.0,
        f"self {pose_distance(P, P, (52, 150)):.2e}, "
        f"vs identity {pose_distance(P, np.eye(3), (52, 150)):.2f} px")

    n_fail = sum(1 for c in checks if not c["pass"])
    if verbose:
        print(f"selftest: {len(checks) - n_fail}/{len(checks)} checks passed")
    return {"checks": checks, "n": len(checks), "n_failed": n_fail}


def _mean_offset(sub_strip, full_strip, first_full_index):
    """Constant offset between a sub-strip's frame placements and the full one's.

    If re-assembling a subset were perfectly repeatable, every shared frame
    would sit at the same place in the sub-strip as in the full strip, give or
    take one constant.  It is not perfectly repeatable -- the sub-strip
    re-estimates its own track and re-picks its own origin -- so this returns
    the mean offset AND the spread around it.  The spread is the ground truth's
    own error bar, and any claim about the mapper's accuracy is only meaningful
    against it.
    """
    n = len(sub_strip.frames)
    idx = np.arange(n) + first_full_index
    if idx.max() >= len(full_strip.px):
        keep = idx < len(full_strip.px)
        idx, n = idx[keep], int(keep.sum())
    dy = np.asarray(sub_strip.py[:n]) - np.asarray(full_strip.py[idx])
    dx = np.asarray(sub_strip.px[:n]) - np.asarray(full_strip.px[idx])
    return (float(dy.mean()), float(dx.mean()),
            float(dy.std()), float(dx.std()), int(n))


def synthetic_split_check(capture, overlap_frames=tuple(range(8)), verbose=True):
    """Ground-truth validation built from ONE real swipe and nothing else.

    Split one swipe's frames into two consecutive runs, assemble each into its
    own strip through the ordinary pipeline, and register those two strips with
    the ordinary mapper.  The answer is known in advance, because both strips
    are built from frames whose positions along the finger were measured once,
    in the full swipe, before the split.  So the true relation between the two
    strips is a pure translation whose value is known and whose rotation is
    zero.

    The split point is fixed and the two runs are allowed to SHARE
    `overlap_frames` frames, so sweeping that parameter sweeps the overlap area
    between the two strips while holding everything else constant.  That turns
    the check into a measurement of the thing enrolment actually needs to know:
    how much overlap two swipes must have before the mapper's answer can be
    trusted.

      overlap_frames = 0  the halves share NO frames.  The strips still overlap
        on the finger, because consecutive frames overlap by about 140 of 150
        columns, but every pixel in the overlap comes from a different exposure
        -- different noise, different pressure, different blur.  This is the
        honest end of the sweep and the one that resembles two real swipes.

      overlap_frames > 0  part of the overlap is literally the same pixels, so
        the problem gets progressively easier and the error should fall to zero.

    Measured on 20260814-170350, the result is the reason the acceptance rule
    has more than one test in it:

        overlap    corner error     NCC    PSR    accepted
        13.0 mm2      15.43 px    0.690   16.9      no
        12.9 mm2       4.37 px    0.889   22.4      no
        14.7 mm2       2.57 px    0.896   21.2      no
        16.5 mm2       1.77 px    0.915   21.1     YES
        22.1 mm2       0.09 px    0.951   20.4     YES

    NCC and PSR are near-useless as accuracy predictors here: the WORST answer
    in the sweep, wrong by 15.4 px (0.78 mm, nearly two ridge pitches, easily
    enough to smear a canvas), scored NCC 0.690 and PSR 16.9 -- comfortably
    above both thresholds.  On NCC and PSR alone it would have been accepted.
    What rejected it was the split-half requirement, which on strips this small
    cannot be run at all, and "cannot be run" is treated as a rejection.  That
    turns out to separate the sweep exactly: every case where the split-half
    test could run was accurate to under 1.8 px, and every case where it could
    not was wrong by 2.6 px or more.

    That separation is real but it is INCIDENTAL -- it works because a strip
    with too little overlap to split is also a strip with too little overlap to
    register accurately.  It should not be relied on as if it were a designed
    margin; see `MIN_TRUSTED_OVERLAP_MM2`, which states it explicitly.

    What this does NOT establish: both strips come from one swipe of one
    finger, so the finger never lifted, never landed at a new angle, and never
    changed pressure between them.  Rotation is zero by construction, so a zero
    rotation error here is not evidence that the rotation search works on real
    swipes -- that claim rests on the synthetic rotation check in --selftest and
    on the spread of angles measured between the real strips.
    """
    dxr = list(range(DX_MIN, DX_MAX + 1))
    dyr = list(range(-DY_ABS, DY_ABS + 1))
    frames, _ = load_swipe(capture)
    segs = segment_capture(frames, dxr, dyr)
    if not segs:
        return {"error": "capture has no usable swipe segment"}
    a0, b0 = max(segs, key=lambda s: s[1] - s[0])
    full = assemble_one(frames[a0:b0], "full", verbose=False)
    if full is None:
        return {"error": "full segment did not assemble"}
    kept = full.frames
    nk = len(kept)
    out = {"capture": str(capture), "segment": [int(a0), int(b0)],
           "n_frames_in_contact": int(nk), "splits": []}
    if verbose:
        print(f"synthetic split of {Path(capture).name} "
              f"segment {a0}:{b0} ({nk} in-contact frames), "
              f"split at frame {nk // 2}")
        print(f"  {'ov':>2}  {'Afr':>3} {'Bfr':>3}  {'ovlap':>8}  "
              f"{'err_px':>7} {'err_mm':>7}  {'dtheta':>6}  "
              f"{'NCC':>5} {'PSR':>5}  {'verdict':<7}  negative controls")

    for ov in overlap_frames:
        mid = nk // 2
        ia, ib = (0, mid + ov), (mid, nk)
        if ia[1] - ia[0] < 4 or ib[1] - ib[0] < 4:
            continue
        A = assemble_one(kept[ia[0]:ia[1]], f"halfA(ov={ov})", verbose=False)
        B = assemble_one(kept[ib[0]:ib[1]], f"halfB(ov={ov})", verbose=False)
        if A is None or B is None:
            out["splits"].append({"overlap_frames": ov,
                                  "error": "a half did not assemble"})
            continue
        ka = ia[0] + A.report["contact_window"][0]
        kb = ib[0] + B.report["contact_window"][0]
        ay, ax, asy, asx, na = _mean_offset(A, full, ka)
        by, bx, bsy, bsx, nb = _mean_offset(B, full, kb)
        true_ty, true_tx = ay - by, ax - bx
        gt_err = math.hypot(asy + bsy, asx + bsx)

        r = register_strip(A.img, A.cov, B.img, B.cov, scales=SCALE_GRID)
        ok, why = judge(r)
        rec = {
            "overlap_frames": ov,
            "shares_frames": bool(ov > 0),
            "half_a_frames": int(len(A.frames)),
            "half_b_frames": int(len(B.frames)),
            "truth": {"ty": round(true_ty, 3), "tx": round(true_tx, 3),
                      "theta_deg": 0.0},
            "truth_uncertainty_px": round(gt_err, 3),
            "truth_spread": {"half_a": [round(asy, 3), round(asx, 3)],
                             "half_b": [round(bsy, 3), round(bsx, 3)]},
            "accepted": bool(ok),
            "judge": why,
        }
        if r is not None:
            Ptrue = translation(true_ty, true_tx)
            rec.update({
                "recovered": {"ty": r["dy"], "tx": r["dx"],
                              "theta_deg": r["theta"], "sx": r["sx"]},
                # True scale is exactly 1 here: both halves are the same skin
                # in the same state.  Any systematic departure from 1 would be
                # estimator bias, and would mean the 3-6% measured between
                # different swipes was an artefact rather than deformation.
                "error_sx": round(r["sx"] - 1.0, 4),
                "error_ty": round(r["dy"] - true_ty, 3),
                "error_tx": round(r["dx"] - true_tx, 3),
                "error_theta_deg": round(r["theta"], 3),
                "corner_error_px": round(
                    pose_distance(r["pose"], Ptrue, B.img.shape), 3),
                "corner_error_mm": round(
                    pose_distance(r["pose"], Ptrue, B.img.shape) * MM_PER_PX, 4),
                "ncc": r["ncc"], "psr": r["psr"],
                "overlap_mm2": r["overlap_mm2"],
                "split_support": r.get("split_support"),
                "split_excess": r.get("split_excess"),
            })
        # Negative controls run through the identical search.
        neg = []
        for kind, bi, bc in (
                ("rot180", B.img[::-1, ::-1], B.cov[::-1, ::-1]),
                ("flipud", np.flipud(B.img), np.flipud(B.cov)),
                ("phase", phase_randomise(B.img, B.cov,
                                          np.random.default_rng(7)), B.cov)):
            rn = register_strip(A.img, A.cov, bi, bc, scales=SCALE_GRID)
            if rn is None:
                neg.append({"kind": kind, "result": "no overlap anywhere"})
                continue
            okn, whyn = judge(rn)
            neg.append({"kind": kind, "accepted": bool(okn), "ncc": rn["ncc"],
                        "psr": rn["psr"], "reason": whyn})
        rec["negative_controls"] = neg
        rec["negative_controls_all_rejected"] = all(
            not c.get("accepted", False) for c in neg)
        out["splits"].append(rec)

        if verbose:
            if r is not None:
                print(f"  {ov:>2}  {len(A.frames):>3} {len(B.frames):>3}  "
                      f"{r['overlap_mm2']:>8.1f}  "
                      f"{rec['corner_error_px']:>7.2f} {rec['corner_error_mm']:>7.3f}  "
                      f"{r['theta']:>+6.2f}  {r['ncc']:>5.3f} {r['psr']:>5.1f}  "
                      f"{'ACCEPT' if ok else 'reject'}   "
                      + ("ok" if rec["negative_controls_all_rejected"]
                         else "NEGATIVE CONTROL ACCEPTED"))
            else:
                print(f"  {ov:>2}  registration returned nothing")

    # What the sweep is for: does any INACCURATE registration get accepted?
    done = [s for s in out["splits"] if "corner_error_px" in s]
    acc = [s for s in done if s["accepted"]]
    rej = [s for s in done if not s["accepted"]]
    out["summary"] = {
        "n_splits": len(done),
        "worst_error_px_accepted": (round(max(s["corner_error_px"] for s in acc), 3)
                                    if acc else None),
        "best_error_px_rejected": (round(min(s["corner_error_px"] for s in rej), 3)
                                   if rej else None),
        "min_overlap_mm2_accepted": (round(min(s["overlap_mm2"] for s in acc), 2)
                                     if acc else None),
        "any_negative_control_accepted": any(
            not s["negative_controls_all_rejected"] for s in done),
        # Scale bias: the mean recovered sx over splits whose true scale is 1.
        "mean_sx_error": (round(float(np.mean([s["error_sx"] for s in acc])), 4)
                          if acc else None),
        "max_abs_sx_error": (round(float(np.max(np.abs(
            [s["error_sx"] for s in acc]))), 4) if acc else None),
        "ground_truth_uncertainty_px": (
            round(max(s["truth_uncertainty_px"] for s in done), 4) if done else None),
    }
    # The separation statement: accepted answers should all be better than
    # every rejected one.  If this is False the acceptance rule is not doing
    # its job and the thresholds are wrong, whatever the averages look like.
    out["summary"]["accept_reject_separated"] = bool(
        acc and rej and
        out["summary"]["worst_error_px_accepted"] <
        out["summary"]["best_error_px_rejected"])
    if verbose:
        s = out["summary"]
        print()
        print(f"  ground truth is exact to {s['ground_truth_uncertainty_px']:.3f} px "
              f"(spread of the shared-frame offsets)")
        if s["accept_reject_separated"]:
            print(f"  SEPARATED: every accepted registration is accurate to "
                  f"{s['worst_error_px_accepted']:.2f} px or better; the best "
                  f"REJECTED one is still {s['best_error_px_rejected']:.2f} px out")
        else:
            print("  NOT SEPARATED: an accepted registration is worse than a "
                  "rejected one -- the acceptance rule is not discriminating")
        print(f"  negative controls: "
              f"{'all rejected' if not s['any_negative_control_accepted'] else 'ONE WAS ACCEPTED'}")
        if s["max_abs_sx_error"] is not None:
            print(f"  along-axis scale bias: mean {s['mean_sx_error']:+.4f}, "
                  f"worst {s['max_abs_sx_error']:.4f} -- true scale is 1 here, "
                  f"so this bounds the estimator's bias and tells you whether "
                  f"the 3-6% between different swipes is real")
    return out


# ---------------------------------------------------------------------------
# 10. output and CLI
# ---------------------------------------------------------------------------
DEFAULT_OUT = Path.home() / ".local/share/elan-fp/finger-map"


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def attribution_pgm(winner, n_strips):
    """Grey level per contributing swipe, so the mosaic's seams are visible."""
    out = np.zeros(winner.shape, np.uint8)
    for si in range(n_strips):
        out[winner == si] = int(40 + 200 * si / max(n_strips - 1, 1))
    return out


def write_map(outdir, mp, rep, accepted):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for mode, (img, cov) in mp["blends"].items():
        u8 = to_u8(img, cov)
        write_pgm(outdir / f"map_{mode}.pgm", u8)
        write_pgm(outdir / f"map_{mode}.nbis-view.pgm", 255 - u8)
        written.append(f"map_{mode}.pgm")
    write_pgm(outdir / "map_attribution.pgm",
              attribution_pgm(mp["winner"], len(accepted)))
    written.append("map_attribution.pgm")
    (outdir / "swipe_map_report.json").write_text(
        json.dumps(rep, indent=2, default=_jsonable))
    written.append("swipe_map_report.json")
    return written


def print_summary(rep):
    c = rep["canvas"]
    print()
    print(f"MAP: {c['w']}x{c['h']} px = {c['mm'][0]:.1f} x {c['mm'][1]:.1f} mm")
    print(f"     covered {c['coverage_mm2']:.1f} mm^2 "
          f"= {c['coverage_frames']:.2f} sensor windows "
          f"(bounding box would claim {c['bbox_frames']:.2f})")
    print(f"     {rep['n_strips_accepted']} of {rep['n_strips_offered']} strips "
          f"accepted, rejection rate {rep['rejection_rate']:.0%}")
    if rep["rejected"]:
        print(f"     rejected: {', '.join(rep['rejected'])}")
    print()
    print(f"  {'swipe':<24} {'won':>7} {'covered':>9} {'exclusive':>10} "
          f"{'NCC':>6} {'PSR':>6}")
    for s in rep["strips"]:
        cf = s["confidence"]
        ncc = cf.get("ncc")
        psr = cf.get("psr")
        print(f"  {s['name']:<24} {s['won_fraction']:>6.1%} "
              f"{s['covered_mm2']:>8.1f}mm2 {s['exclusive_mm2']:>9.1f}mm2 "
              f"{(f'{ncc:.3f}' if ncc is not None else 'seed'):>6} "
              f"{(f'{psr:.1f}' if psr is not None else '-'):>6}")
    base = rep["single_frame_baseline"].get("band_median")
    print()
    print(f"  blend                band_median   vs single frame "
          f"(baseline {base:.3f})")
    for mode, m in sorted(rep["blends"].items(),
                          key=lambda kv: -(kv[1].get("band_median") or 0)):
        bm = m.get("band_median")
        if bm is None:
            continue
        ret = m.get("overlap_retention")
        print(f"  {mode:<20} {bm:>10.3f}   {m.get('vs_single_frame'):>6}"
              f"   overlap retention {ret if ret is not None else '-'}")
    if "strip_route" in rep:
        sr = rep["strip_route"]
        print(f"  {'(strip route)':<20} {sr.get('band_median'):>10.3f}   "
              f"{sr.get('vs_single_frame'):>6}   <- second resampling, control")


def main():
    ap = argparse.ArgumentParser(
        description="Register several swipes into one finger map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("dirs", nargs="*", help="elan-swipe capture directories")
    ap.add_argument("--out-dir", default=None,
                    help=f"where to write the map (default {DEFAULT_OUT}); "
                         "capture dirs are often root-owned, so the default "
                         "is deliberately elsewhere")
    ap.add_argument("--selftest", action="store_true",
                    help="check the machinery against independent computations")
    ap.add_argument("--synthetic", metavar="DIR", default=None,
                    help="ground-truth check: split ONE swipe into two halves, "
                         "assemble each, and see if the mapper recovers the "
                         "known offset")
    ap.add_argument("--nulls", dest="nulls", action="store_true", default=True,
                    help="measure the non-match score distribution (default)")
    ap.add_argument("--no-nulls", dest="nulls", action="store_false")
    ap.add_argument("--dataset", default=str(Path.home() /
                                             ".local/share/elan-fp/dataset"),
                    help="press captures, used only as a null control")
    ap.add_argument("--scale", action="store_true",
                    help="sweep along-axis scale to test whether it matters")
    ap.add_argument("--modes", default="sharp_win",
                    help="comma-separated blend modes, or 'all'")
    ap.add_argument("--json", action="store_true",
                    help="print the full report as JSON on stdout")
    args = ap.parse_args()

    report = {}
    if args.selftest:
        print("selftest")
        report["selftest"] = selftest()
        if not args.dirs and not args.synthetic:
            if args.json:
                print(json.dumps(report, indent=2, default=_jsonable))
            return 0 if report["selftest"]["n_failed"] == 0 else 1

    if args.synthetic:
        print()
        report["synthetic"] = synthetic_split_check(args.synthetic)
        if not args.dirs:
            if args.json:
                print(json.dumps(report, indent=2, default=_jsonable))
            return 0

    if not args.dirs:
        ap.error("give at least one capture directory, or --selftest/--synthetic")

    modes = (BLEND_MODES if args.modes == "all"
             else [m.strip() for m in args.modes.split(",") if m.strip()])

    print("assembling strips")
    strips = []
    for d in args.dirs:
        strips.extend(strips_from_capture(d))
    if not strips:
        print("no usable strips", file=sys.stderr)
        return 1
    print(f"{len(strips)} strip(s) from {len(args.dirs)} capture(s)")

    if len(strips) < 2:
        print("only one strip: nothing to register", file=sys.stderr)
        return 1

    if args.nulls:
        print()
        ds = args.dataset if Path(args.dataset).is_dir() else None
        report["null_distribution"] = null_distribution(strips, dataset_dir=ds)

    print()
    print("building the map")
    accepted, rejected, trials = build_finger_map(strips)

    links = link_graph(accepted)
    report["pose_refinement"] = refine_poses(accepted, links)

    if args.scale and len(accepted) >= 2:
        print()
        a, b = accepted[0], accepted[1]
        th = decompose(b.pose)[0]
        report["scale_assessment"] = assess_scale(a, b, th)
        sa = report["scale_assessment"]
        print(f"along-axis scale: best {sa['best_scale']} "
              f"(NCC {sa['best_ncc']:.4f} vs {sa['ncc_at_scale_1']:.4f} at 1.00, "
              f"gain {sa['gain_over_scale_1']})")

    rep, mp = map_report(accepted, rejected, trials, modes=modes)
    report["map"] = rep
    print_summary(rep)

    outdir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    written = write_map(outdir, mp, report, accepted)
    print()
    print(f"wrote {len(written)} file(s) to {outdir}")

    if args.json:
        print(json.dumps(report, indent=2, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
