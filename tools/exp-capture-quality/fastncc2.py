"""
Vectorised all-shifts NCC that also reports WHERE the best alignment was and
how much real (foreground-intersecting) area the two frames shared there.

Same convention as tools/fastncc.py: a[y, x] pairs with b[y-dy, x-dx].
Validated against the reference implementation in fastncc.py.
"""
import numpy as np


def _integral(x):
    return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def ncc_map(a, b, max_dx, max_dy, min_overlap):
    """NCC for every shift in the window, as a (2*max_dy+1, 2*max_dx+1) array.

    Shifts whose overlap is below min_overlap are -inf.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape

    ph, pw = h + max_dy * 2 + 1, w + max_dx * 2 + 1
    fa = np.fft.rfft2(a, s=(ph, pw))
    fb = np.fft.rfft2(b, s=(ph, pw))
    corr = np.fft.irfft2(fa * np.conj(fb), s=(ph, pw))

    ia, ia2 = _integral(a), _integral(a * a)
    ib, ib2 = _integral(b), _integral(b * b)

    dys = np.arange(-max_dy, max_dy + 1)
    dxs = np.arange(-max_dx, max_dx + 1)
    DY = dys[:, None]
    DX = dxs[None, :]

    ay0 = np.maximum(0, DY) + 0 * DX
    ay1 = np.minimum(h, h + DY) + 0 * DX
    ax0 = np.maximum(0, DX) + 0 * DY
    ax1 = np.minimum(w, w + DX) + 0 * DY
    by0, by1 = ay0 - DY, ay1 - DY
    bx0, bx1 = ax0 - DX, ax1 - DX

    n = (ay1 - ay0) * (ax1 - ax0)

    def rect(ii, y0, y1, x0, x1):
        return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

    sa = rect(ia, ay0, ay1, ax0, ax1)
    sa2 = rect(ia2, ay0, ay1, ax0, ax1)
    sb = rect(ib, by0, by1, bx0, bx1)
    sb2 = rect(ib2, by0, by1, bx0, bx1)

    cross = corr[DY % ph, DX % pw]

    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1)
        num = cross - sa * sb / nn
        va = sa2 - sa * sa / nn
        vb = sb2 - sb * sb / nn
        den = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
        c = np.where(den > 1e-9, num / den, -np.inf)

    c = np.where(n >= min_overlap, c, -np.inf)
    return c, dys, dxs, n


def ncc_best_align(a, b, max_dx=20, max_dy=8, min_overlap=3500,
                   max_rot=0, rot_step=4, rotate=None):
    """Best NCC over shifts (and optionally rotations of b).

    Returns (score, dx, dy, deg).
    """
    best = (-1.0, 0, 0, 0)
    rots = [0] if max_rot == 0 else list(range(-max_rot, max_rot + 1, rot_step))
    for deg in rots:
        br = rotate(b, deg) if (deg and rotate is not None) else b
        c, dys, dxs, _ = ncc_map(a, br, max_dx, max_dy, min_overlap)
        if not np.isfinite(c).any():
            continue
        k = int(np.nanargmax(np.where(np.isfinite(c), c, -1e30)))
        iy, ix = divmod(k, c.shape[1])
        if c[iy, ix] > best[0]:
            best = (float(c[iy, ix]), int(dxs[ix]), int(dys[iy]), deg)
    return best


def overlap_stats(mask_a, mask_b, dx, dy):
    """Given the alignment, how much area do the two frames actually share, and
    how much of it is finger in BOTH captures?

    Returns (geom_overlap_px, both_fg_px, frac_of_a_fg, frac_of_b_fg).
    """
    h, w = mask_a.shape
    ay0, ay1 = max(0, dy), min(h, h + dy)
    ax0, ax1 = max(0, dx), min(w, w + dx)
    if ay1 <= ay0 or ax1 <= ax0:
        return 0, 0, 0.0, 0.0
    pa = mask_a[ay0:ay1, ax0:ax1]
    pb = mask_b[ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx]
    both = int((pa & pb).sum())
    geom = (ay1 - ay0) * (ax1 - ax0)
    na = int(mask_a.sum())
    nb = int(mask_b.sum())
    return (geom, both,
            both / na if na else 0.0,
            both / nb if nb else 0.0)
