"""
Fast normalized cross-correlation over all translations (Lewis 1995).

The naive matcher recomputes means and variances inside a Python loop for every
candidate shift, which dominates runtime and makes parameter sweeps impractical.
Here the cross term is obtained for ALL shifts at once with one FFT pair, and
the per-overlap sums come from integral images, so a full search costs about the
same as a handful of naive shifts.

Only used for offline experimentation. The C port can keep the direct loop,
which is fast enough at 150x52 once the search window is settled.
"""
import numpy as np


def _integral(x):
    """Summed-area table with a zero row/column, so any rectangle sum is O(1)."""
    return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _rect(ii, y0, y1, x0, x1):
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def ncc_all_shifts(a, b, max_dx, max_dy, min_overlap):
    """Return the best NCC over |dx|<=max_dx, |dy|<=max_dy.

    Overlap convention matches the driver: a[y, x] pairs with b[y-dy, x-dx].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape

    # Cross-correlation for every lag, via FFT. Pad to avoid wraparound.
    ph, pw = h + max_dy * 2 + 1, w + max_dx * 2 + 1
    fa = np.fft.rfft2(a, s=(ph, pw))
    fb = np.fft.rfft2(b, s=(ph, pw))
    corr = np.fft.irfft2(fa * np.conj(fb), s=(ph, pw))   # corr[dy, dx] = sum a*b shifted

    ia, ia2 = _integral(a), _integral(a * a)
    ib, ib2 = _integral(b), _integral(b * b)

    best = -1.0
    for dy in range(-max_dy, max_dy + 1):
        y0, y1 = max(0, dy), min(h, h + dy)
        if y1 <= y0:
            continue
        for dx in range(-max_dx, max_dx + 1):
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

            cross = corr[dy % ph, dx % pw]

            num = cross - sa * sb / n
            va = sa2 - sa * sa / n
            vb = sb2 - sb * sb / n
            den = np.sqrt(max(va, 0.0) * max(vb, 0.0))
            if den > 1e-9:
                c = num / den
                if c > best:
                    best = float(c)
    return best


def ncc_best_fast(a, b, max_dx=20, max_dy=8, min_overlap=3500,
                  max_rot=0, rot_step=4, rotate=None):
    if max_rot == 0:
        return ncc_all_shifts(a, b, max_dx, max_dy, min_overlap)
    best = -1.0
    for deg in range(-max_rot, max_rot + 1, rot_step):
        br = rotate(b, deg) if (deg and rotate) else b
        c = ncc_all_shifts(a, br, max_dx, max_dy, min_overlap)
        if c > best:
            best = c
    return best
