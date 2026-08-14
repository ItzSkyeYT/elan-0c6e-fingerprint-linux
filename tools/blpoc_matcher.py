"""
blpoc_matcher.py -- Band-Limited Phase-Only Correlation fingerprint matcher
for very small press-sensor images (ELAN 04f3:0c6e, 150 x 52).

Pure Python 3 standard library only.  No numpy, no scipy, no PIL.
This file doubles as the specification for the C port that will live inside
the libfprint driver, so every step is written out explicitly and every
tunable is collected at the top.

==============================================================================
1.  WHY NOT RAW NORMALISED CROSS-CORRELATION
==============================================================================
The current elanpress matcher correlates RAW pixel intensities.  On a press
sensor the raw image is dominated by contact pressure, contact area and skin
moisture -- session properties, not identity properties.  Two different
fingers pressed with the same force produce very similar low-frequency
intensity blobs, and that is exactly what NCC rewards.  Measured d' on this
hardware was 0.56 and 0.12.

Phase-Only Correlation discards the amplitude spectrum entirely and keeps only
the phase.  Amplitude is where pressure/gain/moisture live; phase is where
ridge geometry lives.  Band-limiting additionally throws away

  * everything below the ridge band (illumination gradients, the pressure
    blob, background-subtraction drift), and
  * everything above it (ADC noise, quantisation, single-pixel speckle),

so that what remains is, close to by construction, ridge structure only.

==============================================================================
2.  WHY WHOLE-IMAGE BLPOC IS NOT ENOUGH, MEASURED ON THIS SENSOR
==============================================================================
Whole-image BLPOC was implemented first and evaluated on 26 captures from this
sensor (12 of one finger, 14 of another; 157 genuine and 168 impostor pairs --
but see the health warning in section 6, those captures were derived renders,
not raw driver output).  It behaved perfectly on synthetic transforms of one
of those captures --

    score(a, a)                        = 1.000
    score(a, a shifted by (7, 2) px)   = 0.999
    score(a, a + Gaussian noise sd 8)  = 0.924

-- and yet gave

    genuine  0.192 +- 0.018      impostor 0.192 +- 0.017      d' = 0.01

on real repeated presses.  The algorithm was not broken; the model was.  Two
presses of the same finger differ by ELASTIC SKIN DISTORTION, not by a rigid
transform.  A few pixels of non-rigid stretch across a 150 px wide image is
already a large fraction of a ridge period (measured at 8-10 px on this
sensor, see section 3), so the ridge phase drifts by half a period from one
end of the image to the other and the global correlation peak collapses to the
noise floor.  Every published whole-image POC result is on an image with 3x to
10x this area, where the same absolute distortion is a much smaller fraction
of the field of view.

The fix is the step that the literature reports as the difference between
d' ~ 2.45 and d' ~ 6.9 on 220x110 finger-knuckle images: correlate LOCALLY
CORRESPONDED BLOCKS and then impose a GLOBAL AFFINE CONSISTENCY constraint on
the block displacements.  Concretely:

    * each block finds its own best displacement within a small window, which
      absorbs elastic distortion; but
    * the score is not read at each block's own peak.  A global affine model
      is robustly fitted to the block displacement field, and the score is the
      correlation value read back at the MODEL-PREDICTED displacement.

That second half is what stops the first half from being a cheat.  Given a
free +-4 px search, an impostor block can often slide onto the nearest ridge
of the other finger and score well -- but it will do so at a displacement that
disagrees with its neighbours, the affine model will not follow it, and the
value read back at the model position is then near zero.  A genuine pair
produces a smooth, low-order displacement field that the model does follow.
The score therefore measures "do these two prints agree everywhere under ONE
plausible deformation", which is a statement about identity, not about
whether two textures with the same ridge pitch can be locally aligned.

==============================================================================
3.  PIPELINE
==============================================================================
  prepare(image)                          once per stored/probe image
    * local contrast normalisation over a 17x17 box (integral images) --
      removes the slowly varying pressure/moisture field, equalises a dry
      press against a wet one
    * foreground mask from local standard deviation; background is zeroed

  score(A, B)
    STAGE 1  global registration.  For each candidate rotation theta, rotate A
             by -theta/2 and B by +theta/2 (splitting the rotation keeps the
             matcher exactly symmetric in its arguments), Hann-window,
             zero-pad to a power of two, FFT, band-limited normalised cross
             power spectrum, inverse FFT.  Keep the top few local maxima over
             all (theta, dx, dy) as CANDIDATE alignments -- not just the best
             one, because on this sensor the global peak is barely above the
             noise floor and frequently is not the true alignment.
    STAGE 2  for each candidate: crop both images to their common region
             (correlating pixels only one image ever saw is pure noise; the
             IEICE 2004 paper credits common-region extraction with most of a
             3.5x EER improvement), then whole-common-region BLPOC.
    STAGE 3  for each candidate: block-wise BLPOC with a robust affine
             consistency constraint, as described in section 2.
    Fuse stages 2 and 3, apply an overlap-coverage penalty, and take the best
    candidate.

==============================================================================
4.  MEASURED RIDGE SPECTRUM OF THIS SENSOR
==============================================================================
Radial power spectrum averaged over 12 real captures, after local contrast
normalisation (percentage of total power per 0.02 cyc/px bin):

    0.00-0.02  DC             4.7%
    0.02-0.04  25-50 px       8.7%     <- pressure blob / drift
    0.04-0.06  17-25 px      12.6%
    0.06-0.08  12-17 px      12.5%
    0.08-0.10  10-12 px      11.5%
    0.10-0.12   8-10 px      25.2%     <- RIDGE FUNDAMENTAL
    0.12-0.14   7- 8 px      15.1%
    0.14-0.16   6- 7 px       5.7%
    0.16-0.18   6 px          2.2%
    0.18-0.50   <6 px         1.7%     <- noise

So the ridge fundamental is 0.10-0.13 cyc/px (period 8-10 px), consistent with
a ~500 dpi sensor.  The default band below is deliberately a little wider than
that to tolerate dry/wet pitch change and stretch, but not much wider: every
retained frequency bin that carries no ridge energy contributes a unit-length
random phasor to the correlation and dilutes the peak.  This is the single
most influential pair of numbers in the file.

==============================================================================
5.  EXPECTATIONS
==============================================================================
Published BLPOC EERs: 1.7% (256x384 pressure sensor), 1.15% (384x256), 3.06%
(FVC2002 DB1, 388x374), and 1.68% vs 6.35% for two independent implementations
on the SAME 220x110 data -- a 4x spread from implementation detail alone.  No
published result uses an image as small as 150x52.  Treat all of them as upper
bounds and re-sweep BAND_LO/BAND_HI, BLOCK_SIZE and BLOCK_SEARCH on real data
before shipping.

==============================================================================
6.  VALIDATION STATUS -- READ THIS BEFORE TRUSTING ANY NUMBER
==============================================================================
Verified here:
  * the FFT (round-trip error 3e-15), the sign convention (shift recovery
    exact to 1 px), score(a,a) == 1.0 exactly, exact symmetry, range [0,1];
  * on synthetic ridge fields carrying a large identical "pressure blob"
    (the nuisance that defeats raw NCC): genuine 0.455 +- 0.015 vs impostor
    0.114 +- 0.074, d' = 6.4, no overlap;
  * on a REAL capture from this sensor: score(a, a shifted 7 px) = 0.999,
    score(a, a + noise sd 8) = 0.924, score(a, a stretched 4%) = 0.66.

NOT verified: genuine-versus-impostor separation on real repeated presses.
The only real captures available in the workspace were 2.5x-upscaled PNG
renders (12 of "right index", 14 of "right middle"), and on them NOTHING
separated the two classes -- not this matcher (d' = 0.01), not raw NCC
(d' = 0.30, with the impostor mean ABOVE the genuine mean), not an
independent ridge-orientation-field correlation (d' = 0.00), and not a free
32x32 patch search that bypasses global registration entirely (best patch
peak: genuine 0.595 +- 0.037, impostor 0.581 +- 0.030 -- i.e. any patch of
any finger matches somewhere in any other image just as well as a genuine
one).  Raw NCC on those renders also fails to reproduce the reported hardware
ground truth (genuine 0.54 there, 0.20 here), so they are evidently not the
raw driver images.

Therefore the first thing to do with this module is to re-run the evaluation
on RAW captures, and to check that the genuine set really consists of
overlapping presses of one finger.  If a free patch search on raw data again
shows genuine and impostor local peaks at the same level, the problem is not
the matcher: it means the captures do not overlap, and the fix has to be in
enrolment/capture (more stages, larger stored area, rejecting low-overlap
presses) rather than in the comparison function.
==============================================================================
"""

from __future__ import annotations

import math

# =============================================================================
# TUNABLES
# =============================================================================

# --- ridge band, in cycles per pixel -----------------------------------------
# See section 4.  A ridge period of p pixels is a radial frequency of 1/p.
BAND_LO = 0.06          # 1/17 px
BAND_HI = 0.20          # 1/5  px

# --- stage 1 global search ----------------------------------------------------
ROTATIONS = (-8.0, -4.0, 0.0, 4.0, 8.0)   # degrees; MUST be symmetric about 0
MAX_SHIFT_X = 45        # px; also fixes the horizontal zero padding
MAX_SHIFT_Y = 10        # px; also fixes the vertical zero padding
N_CANDIDATES = 6        # alignments carried into stages 2 and 3
CANDIDATE_NMS = 8       # px; minimum separation between candidate alignments

# --- local contrast normalisation --------------------------------------------
NORM_RADIUS = 8         # 17x17 box: about two ridge periods.  Big enough to
                        # contain ridge+valley, small enough to track the
                        # pressure field.
NORM_CLIP = 3.0         # clip to +-3 sigma (kills ADC spikes)
MASK_SD_FRACTION = 0.35 # foreground if local sd >= this * median local sd
FLOOR_SD_FRACTION = 0.20# never divide by a local sd below this * median

# --- stage 3 block-wise BLPOC with affine consistency -------------------------
BLOCK_SIZE = 32         # power of two; ~3-4 ridge periods across a block
BLOCK_STEP = 8          # dense grid: the affine fit needs many samples
BLOCK_SEARCH = 4        # px, free per-block search radius.  The key trade-off
                        # in the whole file: large enough to absorb elastic
                        # distortion, but kept UNDER HALF A RIDGE PERIOD
                        # (measured 8-10 px, so under 4-5 px) because beyond
                        # that a block can slide onto the neighbouring ridge
                        # of any finger and impostor scores collapse onto
                        # genuine ones.  The affine constraint below is what
                        # makes even this much freedom safe.
BLOCK_MIN_MASK = 0.75   # fraction of block pixels that must be foreground
BLOCK_MIN_SD = 0.30     # block must carry this much normalised contrast
MIN_VALID_BLOCKS = 6    # below this, stage 3 is not usable
AFFINE_ROUNDS = 3       # iteratively reweighted least squares rounds
AFFINE_SIGMA = 2.0      # px; Cauchy scale for the robust reweighting
AFFINE_REG = 0.05       # ridge regularisation pulling the linear part of the
                        # displacement field towards zero (no stretch/rotation)

# --- fusion -------------------------------------------------------------------
W_GLOBAL = 0.25         # weight of stage 2 (whole common region)
W_BLOCK = 0.75          # weight of stage 3 (affine-consistent block average)
MIN_COVERAGE = 0.50     # the common region must cover at least this fraction
                        # of the smaller foreground area, otherwise the score
                        # is scaled down linearly.  Without this a tiny,
                        # accidentally-agreeing overlap could pass.

# --- numerics -----------------------------------------------------------------
EPS = 1e-12
SNAP_EPS = 1e-9         # snap to exactly 0.0 / 1.0 within this


# =============================================================================
# RADIX-2 COMPLEX FFT
#   Iterative Cooley-Tukey, decimation in time.  Ports directly to C with
#   typedef struct { double re, im; } cpx;
# =============================================================================

_TWIDDLE_CACHE: dict = {}
_BITREV_CACHE: dict = {}


def _twiddles(n: int, inverse: bool) -> list[complex]:
    """exp(sign * 2*pi*i*k/n) for k in [0, n/2); sign = +1 for the inverse."""
    key = (n, inverse)
    tw = _TWIDDLE_CACHE.get(key)
    if tw is None:
        sign = 1.0 if inverse else -1.0
        tw = [complex(math.cos(sign * 2.0 * math.pi * k / n),
                      math.sin(sign * 2.0 * math.pi * k / n))
              for k in range(n // 2)]
        _TWIDDLE_CACHE[key] = tw
    return tw


def _bitrev(n: int) -> list[int]:
    """Bit-reversal permutation table for a power-of-two length n."""
    tab = _BITREV_CACHE.get(n)
    if tab is None:
        bits = n.bit_length() - 1
        tab = [0] * n
        for i in range(n):
            r = 0
            v = i
            for _ in range(bits):
                r = (r << 1) | (v & 1)
                v >>= 1
            tab[i] = r
        _BITREV_CACHE[n] = tab
    return tab


def fft1(a: list[complex], inverse: bool = False) -> None:
    """In-place UNSCALED 1-D DFT of a power-of-two length list."""
    n = len(a)
    if n <= 1:
        return
    rev = _bitrev(n)
    for i in range(n):
        j = rev[i]
        if j > i:
            a[i], a[j] = a[j], a[i]
    tw = _twiddles(n, inverse)
    m = 2
    while m <= n:
        half = m >> 1
        step = n // m
        for start in range(0, n, m):
            k = 0
            for p in range(start, start + half):
                q = p + half
                u = a[p]
                v = a[q] * tw[k]
                a[p] = u + v
                a[q] = u - v
                k += step
        m <<= 1


def fft2(data: list[complex], w: int, h: int, inverse: bool = False) -> None:
    """
    In-place 2-D DFT of a row-major w*h complex array (w, h powers of two).
    Forward is unscaled; inverse divides by w*h.

    All-zero rows and columns are skipped.  That is not merely an
    optimisation: when inverting a band-limited spectrum more than half the
    rows and columns are identically zero, so it roughly quarters the work.
    """
    for y in range(h):
        base = y * w
        seg = data[base:base + w]
        nonzero = False
        for v in seg:
            if v:
                nonzero = True
                break
        if not nonzero:
            continue
        fft1(seg, inverse)
        data[base:base + w] = seg

    col = [0j] * h
    for x in range(w):
        nonzero = False
        for y in range(h):
            v = data[y * w + x]
            col[y] = v
            if v:
                nonzero = True
        if not nonzero:
            continue
        fft1(col, inverse)
        for y in range(h):
            data[y * w + x] = col[y]

    if inverse:
        s = 1.0 / (w * h)
        for i in range(w * h):
            data[i] *= s


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pow2_floor(n: int) -> int:
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p


# =============================================================================
# PREPROCESSING
# =============================================================================

def local_normalise(img: list[float], w: int, h: int,
                    radius: int = NORM_RADIUS):
    """
    Local mean/standard-deviation normalisation plus a foreground mask.

        out[i] = clip((img[i] - mean_box(i)) / max(sd_box(i), floor), +-CLIP)

    and 0 wherever the mask is 0.  Box statistics come from integral images so
    the cost is O(w*h) independent of the radius.  In C use int64 for the sum
    and double for the sum of squares.

    Returns (normalised, mask) with mask entries 0 or 1.
    """
    n = w * h
    iw = w + 1
    s1 = [0.0] * (iw * (h + 1))
    s2 = [0.0] * (iw * (h + 1))
    for y in range(h):
        rs1 = 0.0
        rs2 = 0.0
        row = y * w
        o = (y + 1) * iw
        p = y * iw
        for x in range(w):
            v = img[row + x]
            rs1 += v
            rs2 += v * v
            s1[o + x + 1] = s1[p + x + 1] + rs1
            s2[o + x + 1] = s2[p + x + 1] + rs2

    sd = [0.0] * n
    mean = [0.0] * n
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        a = y0 * iw
        b = y1 * iw
        for x in range(w):
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            cnt = (x1 - x0) * (y1 - y0)
            t1 = s1[b + x1] - s1[b + x0] - s1[a + x1] + s1[a + x0]
            t2 = s2[b + x1] - s2[b + x0] - s2[a + x1] + s2[a + x0]
            m = t1 / cnt
            var = t2 / cnt - m * m
            if var < 0.0:
                var = 0.0
            i = y * w + x
            mean[i] = m
            sd[i] = math.sqrt(var)

    med = sorted(sd)[n // 2]
    floor_sd = max(med * FLOOR_SD_FRACTION, EPS)
    mask_sd = med * MASK_SD_FRACTION

    out = [0.0] * n
    mask = [0] * n
    for i in range(n):
        if sd[i] >= mask_sd and sd[i] > EPS:
            mask[i] = 1
            v = (img[i] - mean[i]) / max(sd[i], floor_sd)
            if v > NORM_CLIP:
                v = NORM_CLIP
            elif v < -NORM_CLIP:
                v = -NORM_CLIP
            out[i] = v
    return out, mask


_HANN_CACHE: dict = {}


def _hann(n: int) -> list[float]:
    """Symmetric Hann window; both endpoints are exactly zero."""
    win = _HANN_CACHE.get(n)
    if win is None:
        win = [1.0] if n == 1 else [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1))
                                   for i in range(n)]
        _HANN_CACHE[n] = win
    return win


def rotate(img: list[float], mask: list[int], w: int, h: int, degrees: float):
    """
    Bilinear rotation about the image centre.  Pixels whose source falls
    outside the image, or whose four source neighbours are not all foreground,
    become background (value 0, mask 0).

    An angle of exactly 0 returns copies -- no resampling, no blur.  That is
    what makes score(a, a) come out exactly 1.
    """
    if abs(degrees) < 1e-12:
        return list(img), list(mask)
    rad = math.radians(degrees)
    ca = math.cos(rad)
    sa = math.sin(rad)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    out = [0.0] * (w * h)
    omask = [0] * (w * h)
    for y in range(h):
        dy = y - cy
        for x in range(w):
            dx = x - cx
            sx = ca * dx + sa * dy + cx        # inverse map: dest -> source
            sy = -sa * dx + ca * dy + cy
            x0 = int(math.floor(sx))
            y0 = int(math.floor(sy))
            if x0 < 0 or y0 < 0 or x0 + 1 >= w or y0 + 1 >= h:
                continue
            i00 = y0 * w + x0
            i10 = i00 + 1
            i01 = i00 + w
            i11 = i01 + 1
            if not (mask[i00] and mask[i10] and mask[i01] and mask[i11]):
                continue
            fx = sx - x0
            fy = sy - y0
            j = y * w + x
            out[j] = ((1.0 - fx) * (1.0 - fy) * img[i00] +
                      fx * (1.0 - fy) * img[i10] +
                      (1.0 - fx) * fy * img[i01] +
                      fx * fy * img[i11])
            omask[j] = 1
    return out, omask


def window_and_pad(img: list[float], w: int, h: int,
                   W: int, H: int) -> list[complex]:
    """
    Apply a separable Hann window and centre the result in a W x H zero-padded
    complex array.

    The window is not optional.  The DFT treats the image as periodic; without
    a window the wrap-around discontinuity at the border injects a strong
    cross-shaped artefact into the phase spectrum which is nearly identical
    for every image and therefore inflates impostor scores.
    """
    wx = _hann(w)
    wy = _hann(h)
    out = [0j] * (W * H)
    ox = (W - w) // 2
    oy = (H - h) // 2
    for y in range(h):
        gy = wy[y]
        src = y * w
        dst = (y + oy) * W + ox
        for x in range(w):
            v = img[src + x] * gy * wx[x]
            if v:
                out[dst + x] = complex(v, 0.0)
    return out


# =============================================================================
# BAND-LIMITED PHASE-ONLY CORRELATION
# =============================================================================

_BAND_CACHE: dict = {}


def band_indices(W: int, H: int,
                 lo: float = None, hi: float = None) -> list[int]:
    """
    Flat indices of the annulus  lo <= sqrt(fx^2 + fy^2) <= hi  in cycles per
    pixel, where fx = u/W and fy = v/H are SIGNED frequencies.

    An annulus is used rather than the rectangular K1 x K2 box of the classic
    BLPOC papers because the padded array here is strongly anisotropic
    (256 x 64 for a 150 x 52 image): a rectangle measured in bins would keep a
    completely different frequency range horizontally and vertically.  The set
    is symmetric under (u,v) -> (-u,-v), so the inverse transform of the phase
    spectrum is real.  DC is excluded automatically because lo > 0.
    """
    if lo is None:
        lo = BAND_LO
    if hi is None:
        hi = BAND_HI
    key = (W, H, lo, hi)
    idx = _BAND_CACHE.get(key)
    if idx is not None:
        return idx
    idx = []
    lo2 = lo * lo
    hi2 = hi * hi
    for v in range(H):
        fy = (v if v * 2 < H else v - H) / H
        fy2 = fy * fy
        base = v * W
        for u in range(W):
            fx = (u if u * 2 < W else u - W) / W
            r2 = fx * fx + fy2
            if lo2 <= r2 <= hi2:
                idx.append(base + u)
    _BAND_CACHE[key] = idx
    return idx


def poc_surface(FA: list[complex], FB: list[complex], W: int, H: int,
                band: list[int]) -> list[float]:
    """
    Band-limited phase-only correlation surface.

        R(u,v) = conj(FA) * FB / |conj(FA) * FB|      inside the band
                 0                                    outside
        r(x,y) = IDFT{R} * (W*H / |band|)

    The scale factor makes r(0,0) exactly 1 when FA == FB, so every value of
    the surface is directly a similarity in [-1, 1].

    Sign convention, verified by the self-test: if b[x] = a[x - d] then the
    peak sits at x = +d, i.e. A[x] corresponds to B[x + d].
    """
    S = [0j] * (W * H)
    for i in band:
        A = FA[i]
        B = FB[i]
        ma = abs(A)
        mb = abs(B)
        if ma < EPS or mb < EPS:
            # No energy here in one of the images: there is no phase to
            # compare, so contribute nothing.  (Calling "both empty" a match
            # would let two blank images score 1.0.)
            continue
        S[i] = (A.conjugate() * B) / (ma * mb)
    fft2(S, W, H, inverse=True)
    scale = (W * H) / float(len(band))
    return [c.real * scale for c in S]


def peak_search(surface: list[float], W: int, H: int,
                max_dx: int, max_dy: int):
    """Largest value of the correlation surface within +-(max_dx, max_dy)."""
    best = -2.0
    bx = by = 0
    for dy in range(-max_dy, max_dy + 1):
        base = (dy % H) * W
        for dx in range(-max_dx, max_dx + 1):
            v = surface[base + (dx % W)]
            if v > best:
                best = v
                bx = dx
                by = dy
    return best, bx, by


def surface_at(surface: list[float], W: int, H: int,
               x: float, y: float) -> float:
    """Bilinear read of the correlation surface at a fractional displacement."""
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    r0 = (y0 % H) * W
    r1 = ((y0 + 1) % H) * W
    c0 = x0 % W
    c1 = (x0 + 1) % W
    return ((1 - fx) * (1 - fy) * surface[r0 + c0] +
            fx * (1 - fy) * surface[r0 + c1] +
            (1 - fx) * fy * surface[r1 + c0] +
            fx * fy * surface[r1 + c1])


# =============================================================================
# TEMPLATE PREPARATION
# =============================================================================

class Template:
    """A preprocessed image, ready to be matched.  Cheap to keep in RAM."""

    __slots__ = ("w", "h", "norm", "mask", "area")

    def __init__(self, w, h, norm, mask):
        self.w = w
        self.h = h
        self.norm = norm
        self.mask = mask
        self.area = sum(mask)


def prepare(img, w: int = 150, h: int = 52) -> Template:
    """Per-image preprocessing.  Do this once per stored image."""
    if len(img) != w * h:
        raise ValueError("expected %d pixels, got %d" % (w * h, len(img)))
    norm, mask = local_normalise([float(v) for v in img], w, h)
    return Template(w, h, norm, mask)


# =============================================================================
# STAGE 1 -- GLOBAL REGISTRATION CANDIDATES
# =============================================================================

def _local_maxima(surface, W, H, max_dx, max_dy, count, nms):
    """
    The `count` strongest local maxima of the correlation surface inside the
    allowed shift box, separated by at least `nms` pixels.
    """
    pts = []
    for dy in range(-max_dy, max_dy + 1):
        base = (dy % H) * W
        for dx in range(-max_dx, max_dx + 1):
            pts.append((surface[base + (dx % W)], dx, dy))
    pts.sort(reverse=True)
    out = []
    for v, dx, dy in pts:
        ok = True
        for _, ox, oy in out:
            if abs(dx - ox) < nms and abs(dy - oy) < nms:
                ok = False
                break
        if ok:
            out.append((v, dx, dy))
            if len(out) >= count:
                break
    return out


def _global_candidates(ta: Template, tb: Template):
    """
    Search rotation x translation with whole-image BLPOC and return a list of
    candidate alignments, best first:

        [(peak, theta, dx, dy, rotA, maskA, rotB, maskB), ...]

    The candidate rotation theta is SPLIT between the two images -- A is
    rotated by -theta/2 and B by +theta/2 -- so that both suffer the same
    resampling blur, and so that swapping the arguments maps theta onto -theta
    which is also in the list.  That is what makes score() exactly symmetric.

    Several candidates are kept because on a 150x52 press image the global
    peak sits close to the noise floor (measured: 0.14-0.23 for genuine pairs)
    and is often NOT the true alignment.  Stage 3 is selective enough to be
    trusted to pick the right one out of a handful.
    """
    w, h = ta.w, ta.h
    W = _next_pow2(w + 2 * MAX_SHIFT_X)
    H = _next_pow2(h + 2 * MAX_SHIFT_Y)
    max_dx = min(MAX_SHIFT_X, (W - w) // 2)
    max_dy = min(MAX_SHIFT_Y, (H - h) // 2)
    band = band_indices(W, H)

    per_rot = max(2, N_CANDIDATES // 2)
    pool = []
    for theta in ROTATIONS:
        ra, ma = rotate(ta.norm, ta.mask, w, h, -0.5 * theta)
        rb, mb = rotate(tb.norm, tb.mask, w, h, +0.5 * theta)
        FA = window_and_pad(ra, w, h, W, H)
        FB = window_and_pad(rb, w, h, W, H)
        fft2(FA, W, H, False)
        fft2(FB, W, H, False)
        surf = poc_surface(FA, FB, W, H, band)
        for peak, dx, dy in _local_maxima(surf, W, H, max_dx, max_dy,
                                          per_rot, CANDIDATE_NMS):
            pool.append((peak, theta, dx, dy, ra, ma, rb, mb))
    pool.sort(key=lambda t: -t[0])
    return pool[:N_CANDIDATES]


def _common_region(ra, ma, rb, mb, w, h, dx, dy):
    """
    Crop both images to the rectangle where they overlap after the
    registration shift, and build the joint foreground mask.

    A[x, y] corresponds to B[x + dx, y + dy], so the valid x range in A
    coordinates is  max(0, -dx) <= x < min(w, w - dx).
    """
    x0 = max(0, -dx)
    x1 = min(w, w - dx)
    y0 = max(0, -dy)
    y1 = min(h, h - dy)
    cw = x1 - x0
    ch = y1 - y0
    if cw <= 0 or ch <= 0:
        return 0, 0, [], [], []
    ca = [0.0] * (cw * ch)
    cb = [0.0] * (cw * ch)
    cm = [0] * (cw * ch)
    for y in range(ch):
        sa = (y0 + y) * w + x0
        sb = (y0 + y + dy) * w + (x0 + dx)
        d = y * cw
        for x in range(cw):
            if ma[sa + x] and mb[sb + x]:
                cm[d + x] = 1
                ca[d + x] = ra[sa + x]
                cb[d + x] = rb[sb + x]
    return cw, ch, ca, cb, cm


# =============================================================================
# STAGE 2 -- WHOLE-COMMON-REGION BLPOC
# =============================================================================

def _global_stage2(ca, cb, cw, ch):
    """Whole-common-region BLPOC with only a couple of pixels of slack."""
    W = _next_pow2(cw)
    H = _next_pow2(ch)
    if W < 8 or H < 8:
        return 0.0
    band = band_indices(W, H)
    if not band:
        return 0.0
    FA = window_and_pad(ca, cw, ch, W, H)
    FB = window_and_pad(cb, cw, ch, W, H)
    fft2(FA, W, H, False)
    fft2(FB, W, H, False)
    surf = poc_surface(FA, FB, W, H, band)
    peak, _, _ = peak_search(surf, W, H, 2, 2)
    return peak


# =============================================================================
# STAGE 3 -- BLOCK-WISE BLPOC WITH ROBUST AFFINE CONSISTENCY
# =============================================================================

def _solve3(m, rhs):
    """Solve a 3x3 symmetric positive definite system by Gaussian elimination."""
    a = [row[:] + [rhs[i]] for i, row in enumerate(m)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-12:
            return None
        a[c], a[p] = a[p], a[c]
        piv = a[c][c]
        for r in range(3):
            if r == c:
                continue
            f = a[r][c] / piv
            if f:
                for k in range(c, 4):
                    a[r][k] -= f * a[c][k]
    return [a[i][3] / a[i][i] for i in range(3)]


def _fit_affine_axis(X, Y, U, Wt, reg):
    """
    Weighted, ridge-regularised least squares fit of  u = a*X + b*Y + c.

    The regularisation pulls a and b (the linear, i.e. rotation/stretch part)
    towards zero but leaves c (a uniform residual translation) free.  A
    genuine pair needs only a small linear term; letting it grow unchecked
    would turn the model into a free-form warp that can also explain an
    impostor.
    """
    sxx = syy = sxy = sx = sy = sw = 0.0
    sxu = syu = su = 0.0
    for i in range(len(X)):
        wt = Wt[i]
        if wt <= 0.0:
            continue
        x = X[i]
        y = Y[i]
        u = U[i]
        sxx += wt * x * x
        syy += wt * y * y
        sxy += wt * x * y
        sx += wt * x
        sy += wt * y
        sw += wt
        sxu += wt * x * u
        syu += wt * y * u
        su += wt * u
    if sw <= 0.0:
        return 0.0, 0.0, 0.0
    lam_x = reg * sxx + EPS
    lam_y = reg * syy + EPS
    m = [[sxx + lam_x, sxy, sx],
         [sxy, syy + lam_y, sy],
         [sx, sy, sw]]
    sol = _solve3(m, [sxu, syu, su])
    if sol is None:
        return 0.0, 0.0, su / sw
    return sol[0], sol[1], sol[2]


def _block_stage3(ca, cb, cm, cw, ch):
    """
    Block-wise BLPOC with a global affine consistency constraint.

    1. tile the common region with overlapping blocks
    2. per block, compute the full BLPOC surface and take its free peak within
       +-BLOCK_SEARCH -> a displacement estimate (u, v) with a confidence
    3. robustly fit  u = a*X + b*Y + c,  v = d*X + e*Y + f  to the
       displacement field (IRLS with a Cauchy weight), where (X, Y) is the
       block centre relative to the centre of the common region
    4. score = mean over blocks of the correlation value read back at the
       MODEL-PREDICTED displacement, not at each block's own free peak

    Step 4 is the whole point; see section 2 of the module docstring.

    Returns a dict with the affine-consistent score plus diagnostics, or None
    if there were not enough usable blocks.
    """
    bx = min(BLOCK_SIZE, _pow2_floor(cw))
    by = min(BLOCK_SIZE, _pow2_floor(ch))
    if bx < 16 or by < 16:
        return None
    band = band_indices(bx, by)
    if not band:
        return None

    def positions(total, size, step):
        pos = list(range(0, total - size + 1, step))
        if not pos:
            return [0]
        if pos[-1] + size < total:
            pos.append(total - size)
        return pos

    xs = positions(cw, bx, BLOCK_STEP)
    ys = positions(ch, by, BLOCK_STEP)
    need = BLOCK_MIN_MASK * bx * by
    n = bx * by

    X = []
    Y = []
    U = []
    V = []
    P = []
    surfaces = []
    ba = [0.0] * n
    bb = [0.0] * n
    for oy in ys:
        for ox in xs:
            cnt = 0
            sa = sb = sa2 = sb2 = 0.0
            for y in range(by):
                s = (oy + y) * cw + ox
                d = y * bx
                for x in range(bx):
                    va = ca[s + x]
                    vb = cb[s + x]
                    ba[d + x] = va
                    bb[d + x] = vb
                    cnt += cm[s + x]
                    sa += va
                    sb += vb
                    sa2 += va * va
                    sb2 += vb * vb
            if cnt < need:
                continue
            vara = sa2 / n - (sa / n) ** 2
            varb = sb2 / n - (sb / n) ** 2
            if vara <= 0.0 or varb <= 0.0:
                continue
            if math.sqrt(vara) < BLOCK_MIN_SD or math.sqrt(varb) < BLOCK_MIN_SD:
                continue
            FA = window_and_pad(ba, bx, by, bx, by)
            FB = window_and_pad(bb, bx, by, bx, by)
            fft2(FA, bx, by, False)
            fft2(FB, bx, by, False)
            surf = poc_surface(FA, FB, bx, by, band)
            peak, du, dv = peak_search(surf, bx, by, BLOCK_SEARCH, BLOCK_SEARCH)
            X.append(ox + bx * 0.5 - cw * 0.5)
            Y.append(oy + by * 0.5 - ch * 0.5)
            U.append(float(du))
            V.append(float(dv))
            P.append(peak)
            surfaces.append(surf)

    nb = len(P)
    if nb < MIN_VALID_BLOCKS:
        return None

    # --- robust affine fit of the displacement field -------------------------
    wt = [max(p, 0.0) ** 2 for p in P]
    coef_u = coef_v = None
    for _ in range(AFFINE_ROUNDS):
        au, bu, cu = _fit_affine_axis(X, Y, U, wt, AFFINE_REG)
        av, bv, cv = _fit_affine_axis(X, Y, V, wt, AFFINE_REG)
        coef_u = (au, bu, cu)
        coef_v = (av, bv, cv)
        base = [max(p, 0.0) ** 2 for p in P]
        for i in range(nb):
            ru = au * X[i] + bu * Y[i] + cu - U[i]
            rv = av * X[i] + bv * Y[i] + cv - V[i]
            r2 = (ru * ru + rv * rv) / (AFFINE_SIGMA * AFFINE_SIGMA)
            wt[i] = base[i] / (1.0 + r2)          # Cauchy / Lorentzian

    # --- read the correlation back at the model-predicted displacement -------
    au, bu, cu = coef_u
    av, bv, cv = coef_v
    total = 0.0
    inliers = 0
    for i in range(nb):
        pu = au * X[i] + bu * Y[i] + cu
        pv = av * X[i] + bv * Y[i] + cv
        total += surface_at(surfaces[i], bx, by, pu, pv)
        ru = pu - U[i]
        rv = pv - V[i]
        if ru * ru + rv * rv <= AFFINE_SIGMA * AFFINE_SIGMA:
            inliers += 1
    return {
        "score": total / nb,
        "free": sum(P) / nb,                  # ablation: no consistency check
        "blocks": nb,
        "inlier_fraction": inliers / nb,
        "affine_u": coef_u,
        "affine_v": coef_v,
    }


# =============================================================================
# THE MATCHER
# =============================================================================

def _fuse(g2, blk, coverage):
    """Combine stage 2 and stage 3 and apply the overlap-coverage penalty."""
    raw = g2 if blk is None else (W_GLOBAL * g2 + W_BLOCK * blk)
    if raw < 0.0:
        raw = 0.0
    elif raw > 1.0:
        raw = 1.0
    if coverage < MIN_COVERAGE:
        raw *= coverage / MIN_COVERAGE
    if raw > 1.0 - SNAP_EPS:
        raw = 1.0
    elif raw < SNAP_EPS:
        raw = 0.0
    return raw


def score_components(ta: Template, tb: Template) -> dict:
    """
    Full computation, returning every intermediate value for the best
    candidate alignment.  Tuning aid: one evaluation pass then yields the
    numbers for any candidate fusion rule.
    """
    if ta.w != tb.w or ta.h != tb.h:
        raise ValueError("image sizes differ")
    best = {"score": 0.0, "stage1": 0.0, "stage2": 0.0, "block": None,
            "free": 0.0, "inlier_fraction": 0.0, "coverage": 0.0,
            "theta": 0.0, "dx": 0, "dy": 0, "blocks": 0, "candidates": 0}
    if ta.area == 0 or tb.area == 0:
        return best                     # no foreground: refuse, do not accept

    w, h = ta.w, ta.h
    cands = _global_candidates(ta, tb)
    best["candidates"] = len(cands)
    for peak1, theta, dx, dy, ra, ma, rb, mb in cands:
        cw, ch, ca, cb, cm = _common_region(ra, ma, rb, mb, w, h, dx, dy)
        if cw <= 0 or ch <= 0:
            continue
        overlap = sum(cm)
        if overlap == 0:
            continue
        denom = min(ta.area, tb.area)
        coverage = overlap / denom if denom else 0.0
        g2 = _global_stage2(ca, cb, cw, ch)
        b3 = _block_stage3(ca, cb, cm, cw, ch)
        s = _fuse(g2, None if b3 is None else b3["score"], coverage)
        if s >= best["score"]:
            best.update({
                "score": s, "stage1": peak1, "stage2": g2,
                "block": None if b3 is None else b3["score"],
                "free": 0.0 if b3 is None else b3["free"],
                "inlier_fraction": 0.0 if b3 is None else b3["inlier_fraction"],
                "blocks": 0 if b3 is None else b3["blocks"],
                "coverage": coverage, "theta": theta, "dx": dx, "dy": dy,
            })
    return best


def score_prepared(ta: Template, tb: Template) -> float:
    """Match two already-prepared templates.  Returns a similarity in [0, 1]."""
    return score_components(ta, tb)["score"]


def score(a, b, w: int = 150, h: int = 52) -> float:
    """
    Similarity of two fingerprint images, in [0, 1], higher = more similar.

    `a` and `b` are flat row-major lists of ints (any range; the module
    normalises).  For repeated matching against a gallery, call prepare() once
    per stored image and then score_prepared().
    """
    return score_prepared(prepare(a, w, h), prepare(b, w, h))


def score_multi(probe, gallery, w: int = 150, h: int = 52) -> float:
    """
    Match a probe against several enrolled images of one finger.

    libfprint enrolment stores 8 stages.  The maximum is used rather than the
    mean because the stages deliberately sample different parts of the
    fingertip, so most of them legitimately do not overlap the probe.
    """
    tp = prepare(probe, w, h)
    best = 0.0
    for g in gallery:
        s = score_prepared(tp, prepare(g, w, h))
        if s > best:
            best = s
    return best


# =============================================================================
# SELF-TEST
# =============================================================================

def _synthetic(w, h, angle_deg=25.0, period=9.0, phase=0.0,
               curve=0.0006, blob=True, noise=0.0, seed=1,
               shift_x=0.0, shift_y=0.0):
    """
    A crude but adequate stand-in for a press capture: curved parallel ridges
    of a given orientation/pitch/phase, plus (optionally) a large smooth
    "contact pressure" blob that is IDENTICAL for every finger.

    The blob is the point of the test.  It is what raw normalised cross
    correlation latches onto, so a matcher that is not fooled by it must score
    a different ridge field clearly lower even though the blob matches
    perfectly.
    """
    import random
    rnd = random.Random(seed)
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    out = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            u = (x - shift_x) - cx
            v = (y - shift_y) - cy
            t = ca * u + sa * v
            s = -sa * u + ca * v
            t += curve * s * s * 12.0                 # gentle curvature
            val = 128.0 + 70.0 * math.sin(2.0 * math.pi * t / period + phase)
            if blob:
                r2 = ((x - cx) / (w * 0.45)) ** 2 + ((y - cy) / (h * 0.7)) ** 2
                val += 60.0 * math.exp(-r2)
            if noise:
                val += rnd.gauss(0.0, noise)
            out[y * w + x] = max(0, min(255, int(round(val))))
    return out


def _warp(img, w, h, kx=0.0, ky=0.0, sx=0.0, sy=0.0):
    """
    Elastic-ish warp: displacement grows quadratically from the centre, which
    is roughly how skin stretches under a differently-angled press.  Used to
    check that the affine consistency stage tolerates genuine distortion.
    """
    out = [0] * (w * h)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    for y in range(h):
        for x in range(w):
            ux = x - cx
            uy = y - cy
            fx = x - (sx + kx * ux)
            fy = y - (sy + ky * uy)
            x0 = int(math.floor(fx))
            y0 = int(math.floor(fy))
            if x0 < 0 or y0 < 0 or x0 + 1 >= w or y0 + 1 >= h:
                continue
            tx = fx - x0
            ty = fy - y0
            i = y0 * w + x0
            out[y * w + x] = int(round(
                (1 - tx) * (1 - ty) * img[i] + tx * (1 - ty) * img[i + 1] +
                (1 - tx) * ty * img[i + w] + tx * ty * img[i + w + 1]))
    return out


def _selftest():
    import time
    W, H = 150, 52

    print("-- FFT sanity ------------------------------------------------")
    a = [complex(math.sin(i * 0.3) + 0.5 * math.cos(i * 1.1), 0.0)
         for i in range(64)]
    b = list(a)
    fft1(b, False)
    fft1(b, True)
    b = [v / 64.0 for v in b]                  # fft1 is unscaled both ways
    err = max(abs(b[i] - a[i]) for i in range(64))
    assert err < 1e-9, err
    print("   1-D round trip max error %.2e" % err)

    plane = [complex((i * 37 % 11) - 5, 0.0) for i in range(32 * 16)]
    ref = list(plane)
    fft2(plane, 32, 16, False)
    fft2(plane, 32, 16, True)
    err = max(abs(plane[i] - ref[i]) for i in range(32 * 16))
    assert err < 1e-9, err
    print("   2-D round trip max error %.2e" % err)

    print("-- shift recovery (fixes the sign convention) ----------------")
    base = _synthetic(W, H, angle_deg=15.0, period=9.0, blob=False)
    for (sx, sy) in ((7, 0), (-11, 0), (0, 3), (5, -2)):
        moved = _synthetic(W, H, angle_deg=15.0, period=9.0, blob=False,
                           shift_x=sx, shift_y=sy)   # b[x] = a[x - s]
        cands = _global_candidates(prepare(base, W, H), prepare(moved, W, H))
        peak, theta, dx, dy = cands[0][:4]
        assert abs(dx - sx) <= 1 and abs(dy - sy) <= 1, \
            "expected (%d,%d) got (%d,%d)" % (sx, sy, dx, dy)
        print("   shift (%+3d,%+3d) recovered as (%+3d,%+3d) peak %.3f"
              % (sx, sy, dx, dy, peak))

    print("-- invariants ------------------------------------------------")
    A = _synthetic(W, H, angle_deg=25.0, period=9.0, phase=0.0, seed=1)
    t0 = time.time()
    s_aa = score(A, A, W, H)
    dt = time.time() - t0
    print("   score(a, a)   = %.12f      (%.2f s per comparison)" % (s_aa, dt))
    assert s_aa == 1.0, s_aa

    B = _synthetic(W, H, angle_deg=-40.0, period=11.0, phase=1.9, seed=2)
    s_ab = score(A, B, W, H)
    s_ba = score(B, A, W, H)
    print("   score(a, b)   = %.12f" % s_ab)
    print("   score(b, a)   = %.12f   (symmetry)" % s_ba)
    assert abs(s_ab - s_ba) < 1e-9, (s_ab, s_ba)
    assert 0.0 <= s_ab <= 1.0

    print("-- tolerance to genuine variation ----------------------------")
    for label, img in (
            ("shift (7, 2)", _warp(A, W, H, sx=7, sy=2)),
            ("stretch 4%", _warp(A, W, H, kx=0.04, ky=0.02)),
            ("stretch + shift", _warp(A, W, H, kx=0.04, ky=0.02, sx=5, sy=1)),
    ):
        s = score(A, img, W, H)
        print("   %-18s %.4f" % (label, s))
        assert s > 0.5, (label, s)

    print("-- separation on synthetic data ------------------------------")
    genuine = []
    for k, (sx, sy, rot, ns) in enumerate(
            ((4, 1, 0.0, 6.0), (-6, -2, 3.0, 8.0), (9, 2, -4.0, 5.0))):
        g = _synthetic(W, H, angle_deg=25.0 + rot, period=9.0, phase=0.0,
                       seed=10 + k, noise=ns, shift_x=sx, shift_y=sy)
        genuine.append(score(A, g, W, H))
    impostor = []
    for k, (ang, per, ph) in enumerate(
            ((-40.0, 11.0, 1.9), (60.0, 8.0, 0.4), (5.0, 12.0, 2.7))):
        b = _synthetic(W, H, angle_deg=ang, period=per, phase=ph,
                       seed=30 + k, noise=6.0)
        impostor.append(score(A, b, W, H))
    for v in genuine + impostor:
        assert 0.0 <= v <= 1.0, v

    def stats(v):
        m = sum(v) / len(v)
        return m, math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))

    mg, sg = stats(genuine)
    mi, si = stats(impostor)
    d = math.sqrt(2.0) * abs(mg - mi) / math.sqrt(sg * sg + si * si + 1e-12)
    print("   genuine  %s  mean %.3f sd %.3f"
          % (["%.3f" % v for v in genuine], mg, sg))
    print("   impostor %s  mean %.3f sd %.3f"
          % (["%.3f" % v for v in impostor], mi, si))
    print("   d' = %.2f  (synthetic, 3+3 samples -- indicative only)" % d)
    assert min(genuine) > max(impostor), "distributions overlap"

    print("\nall self-tests passed")


if __name__ == "__main__":
    _selftest()
