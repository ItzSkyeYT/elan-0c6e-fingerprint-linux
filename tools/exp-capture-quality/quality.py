"""
Per-image capture-quality metrics for the ELAN 0c6e 150x52 press sensor.

Everything here is written so it could be transliterated into C with nothing
beyond libm: separable box/Gaussian blurs, gradients, per-block sums. No FFT,
no linear-algebra library (the 2x2 structure-tensor eigenvalues are closed
form).
"""
import math
import numpy as np

W, H = 150, 52


# --------------------------------------------------------------- helpers --

def _gauss_kernel(sigma):
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def sep_blur(img, sigma):
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="edge")
    out = np.empty_like(img)
    for c in range(img.shape[1]):
        out[:, c] = (p[:, c:c + len(k)] * k).sum(axis=1)
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    res = np.empty_like(img)
    for rr in range(img.shape[0]):
        res[rr, :] = (p[rr:rr + len(k), :] * k[:, None]).sum(axis=0)
    return res


def local_contrast_norm(img, sigma=6.0, eps=1e-3):
    m = sep_blur(img, sigma)
    d = img - m
    v = np.sqrt(np.maximum(sep_blur(d * d, sigma), 0.0))
    return d / (v + eps * v.mean() + 1e-6)


def bandpass(img, lo=1.0, hi=4.0):
    return sep_blur(img, lo) - sep_blur(img, hi)


def block_reduce_sum(x, b):
    """Sum over non-overlapping bxb blocks, cropping the ragged edge."""
    h, w = x.shape
    hh, ww = (h // b) * b, (w // b) * b
    return x[:hh, :ww].reshape(hh // b, b, ww // b, b).sum(axis=(1, 3))


# ------------------------------------------------------- structure tensor --

def structure_tensor(img, block=8, presmooth=1.0):
    """Per-block structure tensor of the (pre-smoothed) image.

    Returns gxx, gyy, gxy summed over each block.
    """
    s = sep_blur(img, presmooth) if presmooth else img
    gy, gx = np.gradient(s)
    return (block_reduce_sum(gx * gx, block),
            block_reduce_sum(gy * gy, block),
            block_reduce_sum(gx * gy, block))


def tensor_metrics(gxx, gyy, gxy):
    """Closed-form 2x2 eigen-decomposition summaries.

    anisotropy = (l1-l2)/(l1+l2) in [0,1]; 1 == perfectly oriented structure
    (a clean ridge flow), 0 == isotropic (noise or flat background).
    energy = l1+l2 = trace.
    """
    tr = gxx + gyy
    diff = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2)
    aniso = np.where(tr > 1e-12, diff / (tr + 1e-12), 0.0)
    return aniso, tr


# ---------------------------------------------------- foreground / masks --

# Absolute band-pass energy floor separating 'ridges present' from 'nothing
# here'. Calibrated from the pooled block-energy distribution over the whole
# dataset (1st pct 23, 10th pct 130, median 465) and cross-checked against
# ASCII renders of the block-energy maps: blocks below ~250 are visibly blank.
# An ABSOLUTE gate matters -- a per-image relative gate can never say "this
# capture covers less of the sensor", which is exactly what we need to measure.
FG_ENERGY_FLOOR = 250.0


def foreground_mask(img, block=8, thresh=FG_ENERGY_FLOOR):
    """A block is 'finger' when its band-passed variance clears an absolute
    floor. Background on this sensor after the driver's own subtraction is
    close to flat, so a simple energy gate works.
    """
    bp = bandpass(img)
    v = block_reduce_sum(bp * bp, block) / (block * block)
    return v >= thresh, v


def foreground_mask_px(img, block=8):
    """Same gate, expanded back to pixel resolution (HxW bool)."""
    m, _ = foreground_mask(img, block)
    up = np.kron(m, np.ones((block, block), dtype=bool))
    out = np.zeros(img.shape, dtype=bool)
    out[:up.shape[0], :up.shape[1]] = up
    # ragged right/bottom edge: copy the nearest block
    if up.shape[1] < img.shape[1]:
        out[:up.shape[0], up.shape[1]:] = up[:, -1:][:, [0] * (img.shape[1] - up.shape[1])]
    if up.shape[0] < img.shape[0]:
        out[up.shape[0]:, :] = out[up.shape[0] - 1:up.shape[0], :]
    return out


# ------------------------------------------------- orientation coherence --

def orientation_coherence(img, block=8, presmooth=1.0):
    """Coherence of the doubled-angle orientation field over a 3x3 block
    neighbourhood: |sum v| / sum |v|. High where the ridge flow is consistent.
    """
    gxx, gyy, gxy = structure_tensor(img, block, presmooth)
    vx = 2 * gxy
    vy = gxx - gyy
    mag = np.sqrt(vx ** 2 + vy ** 2)

    def box3(a):
        p = np.pad(a, 1, mode="edge")
        s = np.zeros_like(a)
        for dy in range(3):
            for dx in range(3):
                s += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
        return s

    num = np.sqrt(box3(vx) ** 2 + box3(vy) ** 2)
    den = box3(mag)
    return np.where(den > 1e-12, num / (den + 1e-12), 0.0)


# ------------------------------------------------ ridge-band spectral SNR --

def ridge_band_ratio(img, block=16):
    """Fraction of band-passed energy that sits in the ridge spatial-frequency
    band, estimated in the spatial domain as (energy of a ridge-band DoG) /
    (energy of a wider-band DoG). Cheap proxy for 'is this actually ridges or
    is it mush/noise'.
    """
    ridge = sep_blur(img, 0.8) - sep_blur(img, 2.4)      # ~ periods 4-12 px
    wide = sep_blur(img, 0.4) - sep_blur(img, 8.0)
    er = float((ridge ** 2).sum())
    ew = float((wide ** 2).sum())
    return er / (ew + 1e-9)


# ------------------------------------------------------------ the bundle --

def image_quality(img, block=8):
    """All per-image metrics in one pass. Returns a dict of scalars."""
    lcn = local_contrast_norm(img)
    fg_blocks, blk_energy = foreground_mask(img, block)
    fg_frac = float(fg_blocks.mean())

    gxx, gyy, gxy = structure_tensor(lcn, block, presmooth=1.0)
    aniso, energy = tensor_metrics(gxx, gyy, gxy)
    coh = orientation_coherence(lcn, block, presmooth=1.0)

    if fg_blocks.any():
        aniso_fg = float(aniso[fg_blocks].mean())
        coh_fg = float(coh[fg_blocks].mean())
        # energy-weighted anisotropy over the whole frame: robust to a bad mask
    else:
        aniso_fg = float(aniso.mean())
        coh_fg = float(coh.mean())

    wsum = energy.sum()
    aniso_w = float((aniso * energy).sum() / (wsum + 1e-12))
    coh_w = float((coh * energy).sum() / (wsum + 1e-12))

    p5, p95 = np.percentile(img, [5, 95])
    bp = bandpass(img)

    # a block is USABLE if it has energy AND oriented structure
    usable = fg_blocks & (aniso >= 0.55)

    return {
        "usable_frac": float(usable.mean()),
        "dyn_range": float(p95 - p5),
        "std": float(img.std()),
        "sat_lo": float((img <= 2).mean()),
        "sat_hi": float((img >= 253).mean()),
        "fg_frac": fg_frac,
        "aniso_fg": aniso_fg,
        "aniso_w": aniso_w,
        "coh_fg": coh_fg,
        "coh_w": coh_w,
        "ridge_band": ridge_band_ratio(img),
        "bp_energy": float(np.sqrt((bp ** 2).mean())),
        "blk_energy_med": float(np.median(blk_energy)),
    }
