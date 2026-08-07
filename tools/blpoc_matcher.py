"""
blpoc_matcher.py -- Band-Limited Phase-Only Correlation (BLPOC) fingerprint
matcher for very small press-sensor images (ELAN 04f3:0c6e, 150 x 52).

Pure Python 3 standard library only.  No numpy, no scipy, no PIL.
This file doubles as the specification for the C port that will live inside
the libfprint driver, so every step is written out explicitly and the
tunables are collected at the top.

------------------------------------------------------------------------------
WHY THIS INSTEAD OF RAW NORMALISED CROSS-CORRELATION
------------------------------------------------------------------------------
The current elanpress matcher correlates RAW pixel intensities.  On a press
sensor the raw intensity image is dominated by contact pressure, contact area
and skin moisture -- all of which are *session* properties, not *identity*
properties.  Two different fingers pressed with the same force produce very
similar low-frequency intensity blobs, which is exactly what NCC rewards.
Measured d' on this hardware was 0.56 and 0.12.

Phase-Only Correlation throws the amplitude spectrum away entirely and keeps
only the phase.  Amplitude is where pressure/gain/moisture live; phase is
where ridge geometry lives.  Band-limiting additionally discards

  * everything below the ridge band  (residual illumination gradients, the
    pressure blob, background-subtraction drift), and
  * everything above the ridge band  (sensor noise, quantisation, the
    single-pixel speckle that a 14-bit ADC on a 52-row array produces).

What is left is, almost by construction, the ridge structure and nothing else.

------------------------------------------------------------------------------
PIPELINE
------------------------------------------------------------------------------
  prepare(image)                       (done once per stored/probe image)
    1. local contrast normalisation over a 17x17 box (integral images), which
       removes the slowly varying pressure/moisture field and equalises
       contrast between a dry press and a wet press
    2. foreground mask from local standard deviation; background is zeroed

  score(A, B)
    3. STAGE 1 -- global registration.
       For each candidate rotation theta, rotate A by -theta/2 and B by
       +theta/2 (splitting the rotation between the two images keeps the whole
       matcher exactly symmetric in its arguments), Hann-window, zero-pad to
       a power of two, forward FFT, form the band-limited normalised cross
       power spectrum, inverse FFT.  The highest correlation peak over all
       (theta, dx, dy) gives the registration.
    4. Common-region extraction.  Crop both images to the region that is
       foreground in both after registration.  This is the step the IEICE 2004
       paper credits with most of its 3.5x EER improvement: correlating over
       pixels that only one of the two images actually saw is pure noise.
    5. STAGE 2 -- global BLPOC on the common region only.
    6. STAGE 3 -- block-wise BLPOC.  Tile the common region with overlapping
       blocks, run BLPOC on each block pair with a *tiny* residual search
       window, and average the block peaks.  On PolyU FKP (220x110) whole-
       image BLPOC gives d' ~ 2.45 while block-wise correlation-surface
       averaging gives d' ~ 4.4-6.9.  Whole-image BLPOC alone is NOT expected
       to reach d' >= 3 on an image this small; the block stage is what buys
       the margin, because it demands that the ridge structure agree
       *everywhere at one single alignment* rather than on average.
    7. Fuse stage 2 and stage 3 and apply an overlap-coverage penalty.

------------------------------------------------------------------------------
NUMBERS TO KEEP IN MIND WHEN TUNING
------------------------------------------------------------------------------
  * Published BLPOC EERs: 1.7% (256x384 pressure sensor), 1.15% (384x256),
    3.06% (FVC2002 DB1, 388x374), 1.68%-6.35% on 220x110 FKP ROIs -- the same
    algorithm, two implementations, a 4x spread.  Windowing, band choice,
    common-region extraction and peak scoring are not polish, they are most of
    the performance.  Expect to have to sweep BAND_LO/BAND_HI on real data.
  * No published result uses an image as small as 150x52.  Treat the numbers
    above as an upper bound.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

# =============================================================================
# TUNABLES
# =============================================================================

# --- ridge band, in cycles per pixel -----------------------------------------
# A ridge period of p pixels is a radial frequency of 1/p.  The ELAN 0c6e is
# ~500 dpi, so a 0.4-0.6 mm ridge pitch is roughly 8-12 px, i.e. 0.08-0.12
# cyc/px.  The band is deliberately wider than that to tolerate stretch and
# dry/wet pitch variation, but it must exclude DC (pressure blob) and the top
# octave (sensor noise).
# SWEEP THESE FIRST on real data; they are the single most influential pair.
BAND_LO = 0.05          # 1/20 px  -- everything slower than this is nuisance
BAND_HI = 0.25          # 1/4  px  -- everything faster than this is noise

# --- stage 1 global search ----------------------------------------------------
ROTATIONS = (-8.0, -4.0, 0.0, 4.0, 8.0)   # degrees; must be symmetric about 0
MAX_SHIFT_X = 45        # px; also fixes the horizontal zero padding
MAX_SHIFT_Y = 10        # px; also fixes the vertical zero padding

# --- local contrast normalisation --------------------------------------------
NORM_RADIUS = 8         # 17x17 box.  ~2 ridge periods: big enough to contain
                        # ridge+valley, small enough to track the pressure field
NORM_CLIP = 3.0         # clip normalised values to +-3 sigma (kills ADC spikes)
MASK_SD_FRACTION = 0.35 # foreground if local sd >= this * median local sd
FLOOR_SD_FRACTION = 0.20# do not divide by a local sd below this * median

# --- stage 3 block-wise BLPOC -------------------------------------------------
BLOCK_SIZE = 32         # must be a power of two; ~3-4 ridge periods per block
BLOCK_STEP = 16         # 50% overlap
BLOCK_SEARCH = 2        # px.  CRITICAL: must be well under half a ridge period,
                        # otherwise every block can be re-aligned onto the
                        # nearest ridge of *any* finger and impostor scores
                        # collapse onto genuine ones.
BLOCK_MIN_MASK = 0.75   # fraction of block pixels that must be foreground
BLOCK_MIN_SD = 0.30     # block must have at least this much normalised contrast
MIN_VALID_BLOCKS = 3    # below this, fall back to the global score alone

# --- fusion -------------------------------------------------------------------
W_GLOBAL = 0.35         # weight of stage 2 (whole common region)
W_BLOCK = 0.65          # weight of stage 3 (block average)
MIN_COVERAGE = 0.50     # common region must cover at least this fraction of
                        # the smaller foreground area, else the score is scaled
                        # down linearly.  Prevents a tiny high-scoring overlap
                        # from passing.

# --- numerics -----------------------------------------------------------------
EPS = 1e-12
SNAP_EPS = 1e-9         # snap scores within this of 0 or 1 to exactly 0 or 1


# =============================================================================
# 1.  RADIX-2 COMPLEX FFT
#     Iterative Cooley-Tukey, decimation in time.  ~60 lines, ports directly
#     to C with a  typedef struct { double re, im; } cpx;
# =============================================================================

_TWIDDLE_CACHE: dict = {}
_BITREV_CACHE: dict = {}


def _twiddles(n: int, inverse: bool) -> list[complex]:
    """exp(sign * 2*pi*i*k/n) for k in [0, n/2), sign = +1 for inverse."""
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
    """Bit-reversal permutation table for length n (n a power of two)."""
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
    """In-place unscaled 1-D DFT of a power-of-two length list."""
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
    The forward transform is unscaled; the inverse divides by w*h.

    All-zero rows and columns are skipped.  This is not just an optimisation:
    for the inverse transform of a band-limited spectrum more than half the
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
# 2.  PREPROCESSING
# =============================================================================

def local_normalise(img: list[float], w: int, h: int,
                    radius: int = NORM_RADIUS):
    """
    Local mean/standard-deviation normalisation plus a foreground mask.

    out[i] = clip((img[i] - mean_box(i)) / max(sd_box(i), floor), +-NORM_CLIP)
    and 0 wherever the mask is 0.

    Box statistics come from integral images so the cost is O(w*h) regardless
    of the radius.  In C use int64 for the sum and double for the sum of
    squares (14-bit input squared * 7800 px still fits comfortably).

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
        y0 = y - radius
        if y0 < 0:
            y0 = 0
        y1 = y + radius + 1
        if y1 > h:
            y1 = h
        a = y0 * iw
        b = y1 * iw
        for x in range(w):
            x0 = x - radius
            if x0 < 0:
                x0 = 0
            x1 = x + radius + 1
            if x1 > w:
                x1 = w
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

    order = sorted(sd)
    med = order[n // 2]
    floor_sd = med * FLOOR_SD_FRACTION
    if floor_sd < EPS:
        floor_sd = EPS
    mask_sd = med * MASK_SD_FRACTION

    out = [0.0] * n
    mask = [0] * n
    for i in range(n):
        if sd[i] >= mask_sd and sd[i] > EPS:
            mask[i] = 1
            d = sd[i]
            if d < floor_sd:
                d = floor_sd
            v = (img[i] - mean[i]) / d
            if v > NORM_CLIP:
                v = NORM_CLIP
            elif v < -NORM_CLIP:
                v = -NORM_CLIP
            out[i] = v
    return out, mask


_HANN_CACHE: dict = {}


def _hann(n: int) -> list[float]:
    """Periodic-free (symmetric) Hann window; endpoints are exactly zero."""
    win = _HANN_CACHE.get(n)
    if win is None:
        if n == 1:
            win = [1.0]
        else:
            win = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1))
                   for i in range(n)]
        _HANN_CACHE[n] = win
    return win


def rotate(img: list[float], mask: list[int], w: int, h: int, degrees: float):
    """
    Bilinear rotation about the image centre.  Pixels whose source falls
    outside the image, or whose four source neighbours are not all foreground,
    become background (value 0, mask 0).

    An angle of exactly 0 returns copies -- no resampling, no blur.  This is
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
            # inverse map: destination -> source
            sx = ca * dx + sa * dy + cx
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
            v = ((1.0 - fx) * (1.0 - fy) * img[i00] +
                 fx * (1.0 - fy) * img[i10] +
                 (1.0 - fx) * fy * img[i01] +
                 fx * fy * img[i11])
            j = y * w + x
            out[j] = v
            omask[j] = 1
    return out, omask


def window_and_pad(img: list[float], w: int, h: int,
                   W: int, H: int) -> list[complex]:
    """
    Apply a separable Hann window and centre the result in a W x H zero-padded
    complex array.

    The window is not optional.  The DFT treats the image as periodic; without
    it the wrap-around discontinuity at the image border injects a strong
    cross-shaped artefact into the phase spectrum, which is identical for every
    image and therefore inflates impostor scores.
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
# 3.  BAND-LIMITED PHASE-ONLY CORRELATION
# =============================================================================

_BAND_CACHE: dict = {}


def band_indices(W: int, H: int,
                 lo: float = BAND_LO, hi: float = BAND_HI) -> list[int]:
    """
    Flat indices of the annulus  lo <= sqrt(fx^2 + fy^2) <= hi  in cycles per
    pixel, where fx = u/W and fy = v/H with u, v taken as signed frequencies.

    An annulus (rather than the rectangular K1 x K2 box of the classic BLPOC
    papers) is used because the padded array is strongly anisotropic here
    (256 x 128 for a 150 x 52 image) and a rectangular box in *bin* units
    would keep a completely different frequency range horizontally and
    vertically.  The set is symmetric under (u,v) -> (-u,-v), so the inverse
    transform of the phase spectrum is real.

    DC is excluded automatically because lo > 0.
    """
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

    The scale factor makes r(0,0) == 1 exactly when FA == FB, so every peak
    value is directly a similarity in [-1, 1].

    Sign convention (verified by the self-test): if b[x] = a[x - d] then the
    peak sits at x = d, i.e.  A[x] corresponds to B[x + d].
    """
    S = [0j] * (W * H)
    for i in band:
        A = FA[i]
        B = FB[i]
        ma = abs(A)
        mb = abs(B)
        if ma < EPS or mb < EPS:
            # One (or both) images carry no energy at this frequency.  There
            # is no phase to compare; contribute nothing.  (When BOTH are
            # empty we could call it agreement, but that would let two blank
            # images match perfectly, so we do not.)
            continue
        S[i] = (A.conjugate() * B) / (ma * mb)
    fft2(S, W, H, inverse=True)
    scale = (W * H) / float(len(band))
    return [c.real * scale for c in S]


def peak_search(surface: list[float], W: int, H: int,
                max_dx: int, max_dy: int):
    """Largest value of the correlation surface within +-(max_dx, max_dy)."""
    best = -2.0
    bx = 0
    by = 0
    for dy in range(-max_dy, max_dy + 1):
        base = (dy % H) * W
        for dx in range(-max_dx, max_dx + 1):
            v = surface[base + (dx % W)]
            if v > best:
                best = v
                bx = dx
                by = dy
    return best, bx, by


# =============================================================================
# 4.  TEMPLATE PREPARATION
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
    """Run the per-image preprocessing.  Do this once per stored image."""
    if len(img) != w * h:
        raise ValueError("expected %d pixels, got %d" % (w * h, len(img)))
    f = [float(v) for v in img]
    norm, mask = local_normalise(f, w, h)
    return Template(w, h, norm, mask)


# =============================================================================
# 5.  THE MATCHER
# =============================================================================

def _global_align(ta: Template, tb: Template):
    """
    Stage 1.  Search rotation x translation with whole-image BLPOC.

    The candidate rotation theta is split between the two images -- A is
    rotated by -theta/2 and B by +theta/2 -- so that both suffer exactly the
    same amount of resampling blur and so that swapping the arguments maps the
    candidate theta onto -theta, which is also in the list.  That makes the
    whole matcher exactly symmetric.

    Returns (peak, theta, dx, dy, rotA, maskA, rotB, maskB).
    """
    w, h = ta.w, ta.h
    W = _next_pow2(w + 2 * MAX_SHIFT_X)
    H = _next_pow2(h + 2 * MAX_SHIFT_Y)
    max_dx = min(MAX_SHIFT_X, (W - w) // 2)
    max_dy = min(MAX_SHIFT_Y, (H - h) // 2)
    band = band_indices(W, H)

    best = None
    for theta in ROTATIONS:
        ra, ma = rotate(ta.norm, ta.mask, w, h, -0.5 * theta)
        rb, mb = rotate(tb.norm, tb.mask, w, h, +0.5 * theta)
        FA = window_and_pad(ra, w, h, W, H)
        FB = window_and_pad(rb, w, h, W, H)
        fft2(FA, W, H, False)
        fft2(FB, W, H, False)
        surf = poc_surface(FA, FB, W, H, band)
        peak, dx, dy = peak_search(surf, W, H, max_dx, max_dy)
        if best is None or peak > best[0]:
            best = (peak, theta, dx, dy, ra, ma, rb, mb)
    return best


def _common_region(ra, ma, rb, mb, w, h, dx, dy):
    """
    Stage 1b.  Crop both images to the rectangle where they overlap after the
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


def _global_stage2(ca, cb, cw, ch):
    """Stage 2.  Whole-common-region BLPOC, zero residual shift allowed."""
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
    peak, _, _ = peak_search(surf, W, H, BLOCK_SEARCH, BLOCK_SEARCH)
    return peak


def _block_stage3(ca, cb, cm, cw, ch):
    """
    Stage 3.  Block-wise BLPOC over the common region.

    Each block pair is correlated independently and the peak is taken within a
    +-BLOCK_SEARCH window (2 px).  The average over blocks is the score.

    Why this separates so much better than the global peak: the global peak
    can be dragged up by a few strongly correlated regions, and on a 150x52
    image "a few regions" is most of the image.  The block average instead
    asks whether the ridge phase agrees in *every* part of the print at one
    single alignment.  A wrong finger can accidentally agree somewhere; it
    almost never agrees everywhere.
    """
    bx = min(BLOCK_SIZE, _pow2_floor(cw))
    by = min(BLOCK_SIZE, _pow2_floor(ch))
    if bx < 8 or by < 8:
        return None
    band = band_indices(bx, by)
    if not band:
        return None

    def positions(total, size, step):
        pos = []
        p = 0
        while p + size <= total:
            pos.append(p)
            p += step
        if pos and pos[-1] + size < total:
            pos.append(total - size)
        if not pos:
            pos = [0]
        return pos

    xs = positions(cw, bx, BLOCK_STEP)
    ys = positions(ch, by, BLOCK_STEP)

    need = BLOCK_MIN_MASK * bx * by
    peaks = []
    ba = [0.0] * (bx * by)
    bb = [0.0] * (bx * by)
    for oy in ys:
        for ox in xs:
            cnt = 0
            sa = sb = 0.0
            sa2 = sb2 = 0.0
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
            n = bx * by
            vara = sa2 / n - (sa / n) ** 2
            varb = sb2 / n - (sb / n) ** 2
            if vara <= 0 or varb <= 0:
                continue
            if math.sqrt(vara) < BLOCK_MIN_SD or math.sqrt(varb) < BLOCK_MIN_SD:
                continue
            FA = window_and_pad(ba, bx, by, bx, by)
            FB = window_and_pad(bb, bx, by, bx, by)
            fft2(FA, bx, by, False)
            fft2(FB, bx, by, False)
            surf = poc_surface(FA, FB, bx, by, band)
            peak, _, _ = peak_search(surf, bx, by, BLOCK_SEARCH, BLOCK_SEARCH)
            peaks.append(peak)

    if len(peaks) < MIN_VALID_BLOCKS:
        return None
    return sum(peaks) / len(peaks)


def score_components(ta: Template, tb: Template) -> dict:
    """
    Same computation as score_prepared() but returns every intermediate value.
    Useful for tuning: it lets one evaluation pass produce the numbers for
    every candidate fusion rule.
    """
    if ta.w != tb.w or ta.h != tb.h:
        raise ValueError("image sizes differ")
    out = {"stage1": 0.0, "stage2": 0.0, "block": None, "coverage": 0.0,
           "theta": 0.0, "dx": 0, "dy": 0, "score": 0.0}
    if ta.area == 0 or tb.area == 0:
        return out

    w, h = ta.w, ta.h
    peak1, theta, dx, dy, ra, ma, rb, mb = _global_align(ta, tb)
    cw, ch, ca, cb, cm = _common_region(ra, ma, rb, mb, w, h, dx, dy)
    out["stage1"] = peak1
    out["theta"] = theta
    out["dx"] = dx
    out["dy"] = dy
    if cw <= 0 or ch <= 0:
        return out
    overlap = sum(cm)
    denom = min(ta.area, tb.area)
    out["coverage"] = overlap / denom if denom else 0.0
    if overlap == 0:
        return out
    out["stage2"] = _global_stage2(ca, cb, cw, ch)
    out["block"] = _block_stage3(ca, cb, cm, cw, ch)
    out["score"] = _fuse(out["stage2"], out["block"], out["coverage"])
    return out


def _fuse(g2: float, blk, coverage: float) -> float:
    """Combine the stage-2 and stage-3 scores and apply the overlap penalty."""
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


def score_prepared(ta: Template, tb: Template) -> float:
    """Match two already-prepared templates.  Returns a similarity in [0, 1]."""
    if ta.w != tb.w or ta.h != tb.h:
        raise ValueError("image sizes differ")
    if ta.area == 0 or tb.area == 0:
        return 0.0                      # no foreground: refuse, do not accept

    w, h = ta.w, ta.h
    peak1, theta, dx, dy, ra, ma, rb, mb = _global_align(ta, tb)
    cw, ch, ca, cb, cm = _common_region(ra, ma, rb, mb, w, h, dx, dy)
    if cw <= 0 or ch <= 0:
        return 0.0

    overlap = sum(cm)
    denom = min(ta.area, tb.area)
    coverage = overlap / denom if denom else 0.0
    if overlap == 0:
        return 0.0

    g2 = _global_stage2(ca, cb, cw, ch)
    blk = _block_stage3(ca, cb, cm, cw, ch)
    return _fuse(g2, blk, coverage)


def score(a, b, w: int = 150, h: int = 52) -> float:
    """
    Similarity of two fingerprint images, in [0, 1], higher = more similar.

    `a` and `b` are flat row-major lists of ints (any range; the module
    normalises).  For repeated matching against a gallery use prepare() once
    per stored image and call score_prepared().
    """
    return score_prepared(prepare(a, w, h), prepare(b, w, h))


def score_multi(probe, gallery, w: int = 150, h: int = 52) -> float:
    """
    Match a probe against several enrolled images of one finger.

    libfprint enrolment stores 8 stages; the driver should call this.  The
    maximum is used rather than the mean because the stages deliberately
    sample different parts of the fingertip, so most of them legitimately do
    not overlap the probe.
    """
    tp = prepare(probe, w, h)
    best = 0.0
    for g in gallery:
        s = score_prepared(tp, prepare(g, w, h))
        if s > best:
            best = s
    return best


# =============================================================================
# 6.  SELF-TEST
# =============================================================================

def _synthetic(w, h, angle_deg=25.0, period=9.0, phase=0.0,
               curve=0.0006, blob=True, noise=0.0, seed=1,
               shift_x=0.0, shift_y=0.0):
    """
    A crude but adequate stand-in for a press-sensor capture: curved parallel
    ridges of a given orientation/pitch/phase, plus (optionally) a big smooth
    low-frequency "contact pressure" blob that is IDENTICAL for every finger.

    The blob is the point of the test.  It is what raw normalised cross
    correlation latches onto, and a matcher that is not fooled by it must give
    a clearly lower score for a different ridge field even though the blob
    matches perfectly.
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
            t += curve * s * s * 12.0          # gentle whorl-ish curvature
            val = 128.0 + 70.0 * math.sin(2.0 * math.pi * t / period + phase)
            if blob:
                r2 = ((x - cx) / (w * 0.45)) ** 2 + ((y - cy) / (h * 0.7)) ** 2
                val += 60.0 * math.exp(-r2)
            if noise:
                val += rnd.gauss(0.0, noise)
            out[y * w + x] = max(0, min(255, int(round(val))))
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
    b = [v / 64.0 for v in b]          # fft1 is unscaled in both directions
    err = max(abs(b[i] - a[i]) for i in range(64))
    assert err < 1e-9, err
    print("   1-D forward/inverse round trip max error %.2e" % err)

    plane = [complex((i * 37 % 11) - 5, 0.0) for i in range(32 * 16)]
    ref = list(plane)
    fft2(plane, 32, 16, False)
    fft2(plane, 32, 16, True)
    err = max(abs(plane[i] - ref[i]) for i in range(32 * 16))
    assert err < 1e-9, err
    print("   2-D forward/inverse round trip max error %.2e" % err)

    print("-- shift recovery (sign convention) --------------------------")
    base = _synthetic(W, H, angle_deg=15.0, period=9.0, blob=False)
    for (sx, sy) in ((7, 0), (-11, 0), (0, 3), (5, -2)):
        # b[x] = a[x - s]  ->  peak must land at +s
        moved = _synthetic(W, H, angle_deg=15.0, period=9.0, blob=False,
                           shift_x=sx, shift_y=sy)
        ta, tb = prepare(base, W, H), prepare(moved, W, H)
        peak, theta, dx, dy, *_ = _global_align(ta, tb)
        assert abs(dx - sx) <= 1 and abs(dy - sy) <= 1, \
            "expected (%d,%d) got (%d,%d)" % (sx, sy, dx, dy)
        print("   shift (%+3d,%+3d) recovered as (%+3d,%+3d) peak %.3f"
              % (sx, sy, dx, dy, peak))

    print("-- invariants ------------------------------------------------")
    A = _synthetic(W, H, angle_deg=25.0, period=9.0, phase=0.0, seed=1)
    t0 = time.time()
    s_aa = score(A, A, W, H)
    dt = time.time() - t0
    print("   score(a, a)      = %.12f   (%.2f s per comparison)" % (s_aa, dt))
    assert s_aa == 1.0, s_aa

    B = _synthetic(W, H, angle_deg=-40.0, period=11.0, phase=1.9, seed=2)
    s_ab = score(A, B, W, H)
    s_ba = score(B, A, W, H)
    print("   score(a, b)      = %.12f" % s_ab)
    print("   score(b, a)      = %.12f" % s_ba)
    assert abs(s_ab - s_ba) < 1e-9, (s_ab, s_ba)
    assert 0.0 <= s_ab <= 1.0

    print("-- separation on synthetic data ------------------------------")
    # genuine: same ridge field, different press (shift, rotation, noise)
    genuine = []
    for k, (sx, sy, rot, ns) in enumerate(
            ((4, 1, 0.0, 6.0), (-6, -2, 3.0, 8.0), (9, 2, -4.0, 5.0))):
        g = _synthetic(W, H, angle_deg=25.0 + rot, period=9.0, phase=0.0,
                       seed=10 + k, noise=ns, shift_x=sx, shift_y=sy)
        genuine.append(score(A, g, W, H))
    # impostor: different ridge orientation/pitch/phase, same pressure blob
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
        var = sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)
        return m, math.sqrt(var)

    mg, sg = stats(genuine)
    mi, si = stats(impostor)
    dprime = math.sqrt(2.0) * abs(mg - mi) / math.sqrt(sg * sg + si * si + 1e-12)
    print("   genuine  %s  mean %.3f sd %.3f"
          % (["%.3f" % v for v in genuine], mg, sg))
    print("   impostor %s  mean %.3f sd %.3f"
          % (["%.3f" % v for v in impostor], mi, si))
    print("   d' = %.2f   (synthetic, 3+3 samples -- indicative only)" % dprime)
    assert min(genuine) > max(impostor), \
        "genuine and impostor distributions overlap on synthetic data"

    print("\nall self-tests passed")


if __name__ == "__main__":
    _selftest()
