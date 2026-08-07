"""
Band-Limited Phase-Only Correlation (BLPOC) for small-area fingerprint matching.

Standard NCC correlates intensity, which on this sensor is dominated by pressure
and contact area rather than identity -- measured d' = -0.72 for the shipped
matcher, i.e. worse than chance.

POC throws the magnitude spectrum away entirely and correlates only PHASE, which
is where ridge geometry lives. The inverse transform of the normalised cross-
phase spectrum has a sharp peak for a genuine pair and stays flat for an
impostor; the peak height is the score. It is also translation-invariant by
construction, so no shift search is needed -- the peak's position IS the shift.

Band-limiting matters: the high-frequency end of a 150x52 capture is mostly
sensor noise, and including it buries the peak. Ito et al. report large EER
improvements from restricting to the informative low-frequency band.

Reference: Ito, Nakajima, Kobayashi, Aoki, Higuchi, "A fingerprint matching
algorithm using phase-only correlation", IEICE Trans. Fundamentals, 2004.
"""
import numpy as np


def _window(h, w):
    return np.hanning(h)[:, None] * np.hanning(w)[None, :]


def blpoc(a, b, k_frac_y=0.5, k_frac_x=0.5, use_window=True,
          max_dy=None, max_dx=None):
    """Peak of the band-limited phase-only correlation surface.

    k_frac_* select the retained fraction of each frequency axis (1.0 = plain
    POC). max_d* optionally restrict the peak search to plausible shifts, which
    stops a spurious far-field peak from winning.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape

    if use_window:                     # suppress edge discontinuities
        win = _window(h, w)
        a = a * win
        b = b * win

    A = np.fft.fft2(a)
    B = np.fft.fft2(b)

    R = A * np.conj(B)
    mag = np.abs(R)
    R = np.where(mag > 1e-12, R / mag, 0.0)      # keep phase only

    ky = max(1, int(h * k_frac_y / 2))
    kx = max(1, int(w * k_frac_x / 2))

    Rs = np.fft.fftshift(R)
    cy, cx = h // 2, w // 2
    band = np.zeros_like(Rs)
    band[cy - ky:cy + ky + 1, cx - kx:cx + kx + 1] = \
        Rs[cy - ky:cy + ky + 1, cx - kx:cx + kx + 1]

    # Inverse transform of only the retained band; normalise by the number of
    # retained components so the score stays comparable across band widths.
    n_kept = (2 * ky + 1) * (2 * kx + 1)
    r = np.fft.ifft2(np.fft.ifftshift(band)).real * (h * w) / n_kept

    if max_dy is None and max_dx is None:
        return float(r.max())

    my = max_dy if max_dy is not None else h // 2
    mx = max_dx if max_dx is not None else w // 2
    idx = [(dy % h, dx % w)
           for dy in range(-my, my + 1)
           for dx in range(-mx, mx + 1)]
    ys = np.array([i for i, _ in idx])
    xs = np.array([j for _, j in idx])
    return float(r[ys, xs].max())


def blpoc_rot(a, b, rotate, max_rot=12, rot_step=3, **kw):
    """POC is translation-invariant but NOT rotation-invariant, so rotation
    still needs a search."""
    if max_rot == 0:
        return blpoc(a, b, **kw)
    best = -1.0
    for deg in range(-max_rot, max_rot + 1, rot_step):
        br = rotate(b, deg) if deg else b
        c = blpoc(a, br, **kw)
        if c > best:
            best = c
    return best
