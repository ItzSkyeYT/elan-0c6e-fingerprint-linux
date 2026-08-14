#!/usr/bin/env python3
"""
Judge an ASSEMBLED SWIPE image from the ELAN 04f3:0c6e sensor.

    ./analyze-swipe.py assembled.pgm          # the thing you came for
    ./analyze-swipe.py --baseline             # press-capture reference numbers
    ./analyze-swipe.py --selftest             # controls with known ground truth
    ./analyze-swipe.py assembled.pgm --json   # machine readable

WHY
---
The 0c6e is 150 x 52 px at ~500 dpi = 7.6 x 2.6 mm.  2.6 mm is swipe geometry.
Upstream libfprint declares FP_SCAN_TYPE_SWIPE for this PID and reconstructs a
tall image with fpi_do_movement_estimation() + fpi_assemble_frames().  This
script answers one question about the result -- IS THE ASSEMBLY ANY GOOD --
from a single image, because the first swipe ever captured has nothing to be
compared against.  It never reports a matching accuracy and cannot: no swipe
data existed when it was written.

WHAT THE ASSEMBLER ACTUALLY DOES
--------------------------------
Read out of libfprint master on 2026-08-14 (libfprint/fpi-assembling.c,
libfprint/drivers/elan.c), not recalled:

  elan.c:329-331   frame 150x52; image_width = frame_width * 3/2 = 225
  fpi-assembling.c:116  find_overlap() searches dy in [2, 52) and dx in
                   [-8, 8) by SAD for each adjacent frame pair
  fpi-assembling.c:240  aes_blit_stripe() OVERWRITES.  No blending, no
                   interpolation, no resampling.
  fpi-assembling.c:299  x starts at (225-150)/2 = 37 and accumulates dx
  elan.c:938-943   0c6e uses elan_process_frame_thirds: each frame is
                   normalised on its OWN percentiles before assembly

Five consequences, and every metric below follows from one of them:

  1. The canvas is 225 px wide with ~37 px of never-written margin on each
     side.  The full rectangle is not captured area.  Everything here is
     measured inside the data column span, eroded by 6 px, because the hard
     zero edge otherwise dominates local contrast normalisation.
  2. The blit overwrites, so only the TOP delta_y rows of each frame survive.
     The image is a stack of thin slabs -- one seam per frame, ~25 of them,
     pitch >= 2 px and not necessarily constant.
  3. Because nothing is resampled, a WRONG delta_y does NOT stretch the image
     smoothly.  Over-estimating replays source rows (exact duplicates at lag
     delta_y_est - delta_y_true); under-estimating throws source rows away.
     So a fitted affine scale factor is the wrong tool -- what works is row
     duplication counting, ridge-pitch drift, and the seam comb.  Measured on
     CONTROL C: dy 3->6 puts 77% of rows in exact-duplicate pairs at lag 3,
     exactly the over-estimate.
  4. delta_y is CLAMPED to >= 2.  A finger that does not move still assembles:
     the same frame stacks ~26 times.  That is the likeliest failure of a first
     swipe attempt and it is reproduced from REAL press captures as CONTROL F.
  5. Per-frame normalisation gives seams a photometric signature independent of
     the ridge signature.  The two are scored separately: a photometric-only
     comb is cosmetic and, usefully, reveals the delta_y the driver chose; a
     ridge-continuity comb means the geometry is wrong.

MEASURED / INFERRED / UNKNOWN
  measured   every number under GEOMETRY, RIDGE CLARITY, RIDGE FREQUENCY,
             SEAMS, DUPLICATION, MINUTIAE.
  inferred   the VERDICT.  Its thresholds come from the 45 real press captures
             (--baseline) and the seven controls (--selftest), whose ground
             truth is known by construction.  Each constant names its source.
  unknown    what a real swipe from this device looks like.  "USABLE" means the
             image survives every failure test that could be defined in
             advance; it is not a claim that it will match.

Polarity: fpi_assemble_frames sets FPI_IMAGE_COLORS_INVERTED (plus H/V flip),
so a raw dump of img->data may have ridges bright.  Everything except the
minutiae skeleton is polarity-invariant, so the minutiae stage runs on both
polarities and both counts are printed.

Reused, not reimplemented:
  tools/exp-capture-quality/quality.py  foreground mask (absolute band-pass
      energy floor 250, calibrated on this dataset), structure tensor,
      anisotropy, orientation coherence, ridge-band ratio.
  tools/exp-py-minutiae/minutiae.py     the crossing-number pipeline (segment,
      LCN, orientation, Gabor, binarise, Zhang-Suen, CN, spur/merge cleanup).
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = Path(os.path.expanduser("~/.local/share/elan-fp/dataset"))

# --------------------------------------------------------------- constants --

DEFAULT_DPI = 500.0            # the sensor reports 508; --dpi changes it
FRAME_W, FRAME_H = 150, 52     # elan.c, read from the device for the 0c6e
ASSEMBLED_W = FRAME_W * 3 // 2  # elan.c:331 -> 225
DY_MIN, DY_MAX = 2, FRAME_H - 1  # fpi-assembling.c:116
EDGE_CROP = 6                  # px trimmed off the data column span

# Literature anchor, not measured here: adult ridge pitch 0.4-0.6 mm.  The real
# press captures agree -- --baseline measures 8.85 px median at 500 dpi =
# 0.450 mm (p10 0.390, p90 0.515) over 45 captures, and
# exp-capture-quality/s5_information.py got 10.0 px = 0.50 mm at 508 dpi by
# autocorrelation rather than by spectrum.  The band below is widened to
# 0.38-0.62 mm so the estimator's own spread does not fail a good image: at
# 0.38-0.62 exactly one of the 45 real captures is rejected (0.36 mm).
PERIOD_MM_LO, PERIOD_MM_HI = 0.38, 0.62

# ---- verdict thresholds.  Every one is a measurement; see VERDICT_NOTES.
T_RIDGE_BAND = 0.18      # noise 0.126 vs press 0.311 (p10 0.268)
T_USABLE_FRAC = 0.20     # noise 0.07 vs press 0.64 (p10 0.35)
T_ESTIMABLE = 0.20       # noise 0.05 vs press 0.67 (p10 0.54)
T_DUP_FRAC = 0.12        # good controls 0.000-0.029 vs C 0.77, F 0.44
T_COMB_Z = 5.0           # good controls <= -0.3 vs C 13.5, D 8.0
T_FLAG_FRAC = 0.05       # good controls 0.015-0.018, press p90 0.020,
                         # vs C/D/E 0.108-0.217
T_BAND_CV = 0.18         # good controls 0.06-0.07 vs E 0.19, D 0.26.  Loosest of
                         # the thresholds: a real finger's pitch varies more
                         # across a 15 mm swipe than the synthetic control's
                         # does, so expect this rule to need widening first.
T_CONT_MED = 0.60        # good controls 0.78-0.82, press 0.855; stationary
                         # stack 0.33
MIN_MINUTIAE = 12        # forensic convention, NOT measured here
MIN_AREA_MM2 = 40.0      # 2.5x a press capture; below this a swipe is pointless


# ------------------------------------------------------------------ setup ---

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(HERE / "exp-py-minutiae"))
CQ = _load("cq_quality", HERE / "exp-capture-quality" / "quality.py")
MN = _load("mn_minutiae", HERE / "exp-py-minutiae" / "minutiae.py")


def load_pgm(path):
    data = Path(path).read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"{path}: not a binary PGM (P5)")
    fields, idx = [], 2
    while len(fields) < 3:
        while data[idx:idx + 1].isspace():
            idx += 1
        if data[idx:idx + 1] == b"#":
            while data[idx] != 0x0A:
                idx += 1
            continue
        s = idx
        while not data[idx:idx + 1].isspace():
            idx += 1
        fields.append(int(data[s:idx]))
    idx += 1
    w, h, _ = fields
    px = np.frombuffer(data[idx:idx + w * h], dtype=np.uint8)
    if px.size != w * h:
        raise ValueError(f"{path}: truncated ({px.size} of {w*h} px)")
    return px.reshape(h, w).astype(np.float32)


def save_pgm(path, img):
    a = np.asarray(img, np.float64)
    lo, hi = float(a.min()), float(a.max())
    b = np.zeros_like(a) if hi <= lo else (a - lo) * 255.0 / (hi - lo)
    h, w = a.shape
    Path(path).write_bytes(b"P5\n%d %d\n255\n" % (w, h) + b.astype(np.uint8).tobytes())


# ------------------------------------------------------------ primitives ----

def foreground(img, block=8):
    """Pixel-resolution finger mask: the calibrated absolute band-pass energy
    gate from exp-capture-quality/quality.py (FG_ENERGY_FLOOR = 250)."""
    return CQ.foreground_mask_px(img, block)


def data_span(img, crop=EDGE_CROP):
    """Contiguous column span holding finger data, eroded by `crop`.

    The 225-px canvas has hard zero margins; local contrast normalisation
    across that edge produces a huge artificial line in EVERY row, which by
    itself makes unrelated rows correlate ~0.59 (measured on the noise
    control).  Every row-wise statistic below is computed inside this span."""
    fg = foreground(img)
    cf = fg.mean(axis=0)
    if cf.max() <= 0:
        return 0, img.shape[1]
    idx = np.where(cf >= 0.25 * cf.max())[0]
    x0, x1 = int(idx[0]) + crop, int(idx[-1]) + 1 - crop
    if x1 - x0 < 24:
        return int(idx[0]), int(idx[-1]) + 1
    return x0, x1


def prepped(img):
    """LCN, cropped to the data span, with the fixed column profile removed.
    Column-mean removal matters: sensor fixed-pattern noise and any residual
    background is shared by every row and would otherwise read as vertical
    self-similarity, i.e. as fake duplication."""
    x0, x1 = data_span(img)
    L = CQ.local_contrast_norm(img)[:, x0:x1]
    return L - L.mean(axis=0, keepdims=True)


def _rownorm(A):
    A = A - A.mean(axis=1, keepdims=True)
    n = np.sqrt((A * A).sum(axis=1, keepdims=True))
    return A / np.where(n > 1e-9, n, 1.0), (n[:, 0] > 1e-9)


def _robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-6 else (x.std() + 1e-9)
    return (x - med) / scale


# ------------------------------------------------------------ 1. geometry ---

def geometry(img, dpi):
    h, w = img.shape
    mm = 25.4 / dpi
    fg = foreground(img)
    x0, x1 = data_span(img)
    rows_used = fg.any(axis=1)
    # left/right edge of the finger data per row -> cumulative delta_x, but
    # only meaningful when the canvas actually has margins to drift in.
    has_margin = w >= FRAME_W + 16
    drift = jump = float("nan")
    if has_margin:
        left = np.array([np.argmax(r) if r.any() else np.nan for r in fg], float)
        cen = left + np.array([r.sum() for r in fg], float) / 2.0
        cen = cen[np.isfinite(cen)]
        if cen.size > 2:
            drift = float(np.percentile(cen, 98) - np.percentile(cen, 2))
            jump = float(np.percentile(np.abs(np.diff(cen)), 95))
    return {
        "w": int(w), "h": int(h),
        "mm_w": w * mm, "mm_h": h * mm,
        "rect_mm2": w * h * mm * mm,
        "fg_px": int(fg.sum()),
        "fg_frac": float(fg.mean()),
        "area_mm2": float(fg.sum()) * mm * mm,
        "rows_with_finger": int(rows_used.sum()),
        "travel_mm": float(rows_used.sum()) * mm,
        "data_span": [int(x0), int(x1)],
        "canvas_is_libfprint": bool(w == ASSEMBLED_W),
        "x_drift_px": drift,
        "x_jump_p95": jump,
        "area_vs_press": float(fg.sum()) * mm * mm / 15.6,
        "_fg": fg,
    }


# ------------------------------------------------------ 2. ridge clarity ----

def clarity(img, block=8):
    """Anisotropy and orientation coherence from the reused structure-tensor
    code.  Both are reported over the whole image AND over the foreground: at
    8x8 blocks the foreground-only figures are biased upward by selection (the
    noise control scores 0.74 anisotropy over its surviving 9% of blocks but
    0.29 over the whole image), so the whole-image numbers are the ones the
    verdict trusts."""
    L = CQ.local_contrast_norm(img)
    gxx, gyy, gxy = CQ.structure_tensor(L, block, presmooth=1.0)
    aniso, energy = CQ.tensor_metrics(gxx, gyy, gxy)
    coh = CQ.orientation_coherence(L, block, presmooth=1.0)
    fgb, _ = CQ.foreground_mask(img, block)
    sel = fgb if fgb.any() else np.ones_like(fgb, bool)
    vx, vy = 2 * gxy, (gxx - gyy)
    v = (vx + 1j * vy)[sel]
    r = float(abs(v.sum()) / (np.abs(v).sum() + 1e-12))
    spread = math.degrees(math.sqrt(max(-2 * math.log(max(r, 1e-9)), 0.0))) / 2
    ridge_dir = math.degrees((0.5 * np.angle(v.sum()) + math.pi / 2) % math.pi)
    p5, p95 = np.percentile(img, [5, 95])
    return {
        "aniso_all": float(aniso.mean()),
        "aniso_fg": float(aniso[sel].mean()),
        "aniso_w": float((aniso * energy).sum() / (energy.sum() + 1e-12)),
        "coh_all": float(coh.mean()),
        "coh_fg": float(coh[sel].mean()),
        "usable_frac": float((fgb & (aniso >= 0.55)).mean()),
        "orient_spread_deg": float(spread),
        "ridge_dir_deg": float(ridge_dir),
        "ridge_band": float(CQ.ridge_band_ratio(img)),
        "dyn_range": float(p95 - p5),
    }


# --------------------------------------------------- 3. ridge frequency -----

def _spectral_period(patch, lo=4.0, hi=24.0, nb=36):
    """Dominant ridge period (px) from the radial power spectrum, searched only
    inside a plausible band.  Validated: 9.74 px on a real press capture, where
    exp-capture-quality got 10.0 px by autocorrelation, and 9.74 px on a
    synthetic control built with a true period of 10.0."""
    a = np.asarray(patch, float)
    if a.shape[0] < 16 or a.shape[1] < 16 or a.std() < 1e-6:
        return float("nan")
    # Never search for a period longer than a quarter of the smallest side: at
    # fewer than ~4 cycles the periodogram cannot tell a ridge pitch from the
    # residual shape of the finger.  Unclamped, this reported 0.9-1.1 mm
    # "ridge pitch" on 7 of the 45 real 52-row press captures -- physically
    # impossible, and enough to fail them on the pitch-plausibility rule.
    hi = min(hi, min(a.shape) / 4.0)
    if hi <= lo + 1.0:
        return float("nan")
    a = a - a.mean()
    a = a * np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(a.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(a.shape[1]))[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    bins = np.linspace(1.0 / hi, 1.0 / lo, nb + 1)
    band = (rad >= bins[0]) & (rad <= bins[-1])
    if not band.any():
        return float("nan")
    idx = np.clip(np.digitize(rad, bins) - 1, 0, nb - 1)
    prof = np.zeros(nb)
    for i in range(nb):
        m = band & (idx == i)
        if m.any():
            prof[i] = P[m].mean()
    if prof.max() <= 0:
        return float("nan")
    i = int(prof.argmax())
    return float(1.0 / (0.5 * (bins[i] + bins[i + 1])))


def _block_estimable(L, block=16):
    """Fraction of blocks where a ridge pitch can be measured at all, by
    projecting along the local ridge and counting peak spacings across it (the
    method in minutiae.ridge_frequency, but reporting validity instead of
    silently substituting a prior).  Noise 0.05, press 0.70."""
    h, w = L.shape
    MN.W, MN.H = w, h
    th, _ = MN.orientation(L, block=8)
    ok = tot = 0
    for by in range(h // block):
        for bx in range(w // block):
            cy, cx = by * block + block / 2, bx * block + block / 2
            t = float(th[int(cy), int(cx)])
            uu, vv = np.mgrid[-12:12, -8:9].astype(np.float32)
            X = np.clip(np.round(cx + uu * math.cos(t) - vv * math.sin(t)), 0, w - 1).astype(int)
            Y = np.clip(np.round(cy + uu * math.sin(t) + vv * math.cos(t)), 0, h - 1).astype(int)
            prof = L[Y, X].mean(axis=0)
            prof = prof - prof.mean()
            pk = [i for i in range(1, len(prof) - 1)
                  if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > 0]
            tot += 1
            if len(pk) >= 2:
                d = float(np.diff(pk).mean())
                if d > 0 and 4.0 <= d <= 24.0:
                    ok += 1
    return ok / max(tot, 1)


def frequency(img, dpi):
    """Ridge pitch overall and band by band along the assembly axis.

    This is the sharpest single test of an assembly, because the assembler does
    not resample: over-estimating delta_y replays rows and the apparent pitch
    grows (CONTROL C: 15.4 px = 0.78 mm), under-estimating drops rows and it
    shrinks (CONTROL D: 5.6 px = 0.29 mm), while both good controls and the
    real press captures sit at 9.7-10.4 px = 0.49-0.53 mm."""
    L = prepped(img)
    h = L.shape[0]
    mm = 25.4 / dpi
    per = _spectral_period(L)
    # Bands along the swipe axis, OVERLAPPING by half.  Tall bands (>= 64 rows)
    # because _spectral_period caps its search at a quarter of the smallest
    # side -- narrow bands cannot see a pitch that has been inflated by an
    # over-estimated delta_y, which is the discontinuity being looked for.
    bh = int(max(64, h // 8))
    step = max(16, bh // 2)
    starts = list(range(0, max(1, h - bh + 1), step))
    band = [_spectral_period(L[s:s + bh]) for s in starts] if h >= 128 else []
    nb = len(band)
    b = np.array([v for v in band if np.isfinite(v)], float)
    cv = float(b.std() / b.mean()) if b.size > 2 else float("nan")
    jump = float(np.max(np.abs(np.diff(b)) / b.mean())) if b.size > 2 else float("nan")
    return {
        "period_px": per,
        "period_mm": per * mm,
        "plausible_pitch": bool(np.isfinite(per) and PERIOD_MM_LO <= per * mm <= PERIOD_MM_HI),
        "estimable_frac": _block_estimable(L),
        "n_bands": nb,
        "band_period_px": [None if not np.isfinite(v) else round(float(v), 2) for v in band],
        "band_cv": cv,
        "band_max_jump": jump,
    }


# --------------------------------------------------------------- 4. seams --

def _comb(profile, pmin=DY_MIN, pmax=DY_MAX, nperm=200, seed=0):
    """Is the discontinuity profile periodic, and at what pitch?

    For each candidate pitch p the profile is folded and the strongest phase is
    scored as a t-like statistic normalised by the number of folds, so the
    score is comparable across p (an unnormalised fold score grows with p and
    always picks the largest pitch -- the first version of this function did
    exactly that and missed every real seam).  Significance comes from a
    permutation null: shuffling destroys periodicity but preserves the
    marginal distribution."""
    x = np.asarray(profile, float)
    n = x.size
    pmax = int(min(pmax, n // 4))
    if pmax < pmin or n < 12:
        return {"pitch": None, "score": 0.0, "z": 0.0, "p": 1.0}
    sd = x.std() + 1e-12

    def best(v):
        s_best, p_best = -np.inf, None
        for p in range(pmin, pmax + 1):
            m = (v.size // p) * p
            f = v[:m].reshape(-1, p)
            ph = f.mean(axis=0)
            s = (ph.max() - ph.mean()) / (sd / math.sqrt(f.shape[0]) + 1e-12)
            if s > s_best:
                s_best, p_best = s, p
        return s_best, p_best

    s0, p0 = best(x)
    rng = np.random.default_rng(seed)
    null = np.array([best(rng.permutation(x))[0] for _ in range(nperm)])
    return {"pitch": int(p0), "score": float(s0),
            "z": float((s0 - null.mean()) / (null.std() + 1e-9)),
            "p": float((null >= s0).mean())}


def seams(img, known_seams=None):
    """Horizontal discontinuities, on three independent channels.

    ridge     row-to-row correlation at ZERO lateral lag.  Zero lag matters:
              letting the correlator search a lateral offset lets it re-find
              the shifted content and hides the seam (measured: the same
              control scores comb z 13.5 at zero lag and 2.6 with a free lag).
              The free-lag version is still computed, as a lateral-jump channel.
    photo     step in row mean / row sd.  Each frame is normalised on its own
              percentiles, so slabs differ in gain and offset even when the
              geometry is perfect.  A photometric-only comb is cosmetic -- and
              it reveals the delta_y the driver used.
    lateral   change in the best lateral lag between adjacent rows = bad dx.
    """
    L = prepped(img)
    h, m = L.shape
    if h < 8 or m < 24:
        return {"error": "image too small for seam analysis"}

    ml = 3
    Ah, ok = _rownorm(L[:, ml:m - ml])
    cont0 = np.where(ok[:-1] & ok[1:], (Ah[:-1] * Ah[1:]).sum(axis=1), 0.0)
    best = np.full(h - 1, -1.0)
    blag = np.zeros(h - 1, int)
    for lag in range(-ml, ml + 1):
        Bh, okb = _rownorm(L[:, ml + lag:m - ml + lag])
        c = np.where(ok[:-1] & okb[1:], (Ah[:-1] * Bh[1:]).sum(axis=1), -1.0)
        upd = c > best
        best[upd], blag[upd] = c[upd], lag

    x0, x1 = data_span(img)
    raw = img[:, x0:x1].astype(np.float64)
    dmean = np.abs(np.diff(raw.mean(axis=1)))
    dstd = np.abs(np.diff(raw.std(axis=1)))

    z_ridge = np.maximum(-_robust_z(cont0), 0.0)
    z_photo = np.maximum(np.maximum(_robust_z(dmean), _robust_z(dstd)), 0.0)
    z_lat = np.maximum(_robust_z(np.abs(np.diff(blag, prepend=blag[0]))), 0.0)

    flagged = z_ridge > 4.0
    res = {
        "cont_median": float(np.median(cont0)),
        "cont_p10": float(np.percentile(cont0, 10)),
        "cont_freelag_median": float(np.median(best)),
        "lag_changes": int((np.diff(blag) != 0).sum()),
        "n_boundaries": int(h - 1),
        "n_flagged": int(flagged.sum()),
        "frac_flagged": float(flagged.mean()),
        "seam_rows": [int(i + 1) for i in np.where(flagged)[0]][:64],
        "comb_ridge": _comb(z_ridge),
        "comb_photo": _comb(z_photo),
        "comb_lateral": _comb(z_lat),
    }
    if known_seams:
        k = np.array(sorted({int(s) for s in known_seams if 1 <= s < h}))
        if k.size:
            other = np.setdiff1d(np.arange(1, h), k)
            res["gt_seam_z"] = float(np.mean(z_ridge[k - 1]))
            res["gt_other_z"] = float(np.mean(z_ridge[other - 1])) if other.size else None
            res["gt_detected_frac"] = float((z_ridge[k - 1] > 4.0).mean())
    return res


# --------------------------------------------------------- 5. duplication ---

def duplication(img, kmax=24, thr=0.95):
    """Does the image contain more rows than it contains information?

    Because aes_blit_stripe copies pixels verbatim, an over-estimated delta_y
    leaves rows that are EXACT replays of earlier rows, at lag
    (delta_y_est - delta_y_true).  So the test is literal: what fraction of
    rows have a near-identical partner (correlation > 0.95, column profile
    removed) somewhere within 24 rows?  Measured -- clean 0.0%, correct
    assembly 0.0%, real press 0.0%, noise 0.0%, jittered 2.9%, dy 3->6 77.0%
    at lag 3 (the exact over-estimate), stationary finger 44.1% at lag 4.

    self_sim(k) and the effective rank are reported as context.  Neither is
    used as a threshold: a real print recurs strongly at multiples of its own
    vertical ridge spacing, so a high self-similarity is not by itself
    evidence of duplication."""
    L = prepped(img)
    Ah, ok = _rownorm(L)
    Ah = Ah[ok]
    h = Ah.shape[0]
    if h < 8:
        return {"error": "too few rows"}
    best = np.zeros(h)
    bestk = np.zeros(h, int)
    S = []
    for k in range(1, min(kmax, h // 2) + 1):
        c = (Ah[:-k] * Ah[k:]).sum(axis=1)
        S.append(float(c.mean()))
        if k >= 2:
            for a, b in ((0, h - k), (k, h)):
                upd = c > best[a:b]
                best[a:b] = np.where(upd, c, best[a:b])
                bestk[a:b] = np.where(upd, k, bestk[a:b])
    hits = best > thr
    lag = int(np.bincount(bestk[hits]).argmax()) if hits.any() else None
    sv = np.linalg.svd(Ah, compute_uv=False)
    p = sv ** 2
    p = p / (p.sum() + 1e-12)
    erank = float(math.exp(-(p * np.log(p + 1e-12)).sum()))
    return {
        "dup_frac": float(hits.mean()),
        "dup_lag": lag,
        "best_partner_median": float(np.median(best)),
        "self_sim": [round(v, 3) for v in S],
        "erank": erank,
        "erank_ratio": erank / max(h, 1),
        "rows": int(h),
    }


# ------------------------------------------------------------ 6. minutiae ---

def minutiae(img, dpi, fg=None, both_polarities=True, cfg=None):
    """Run the existing crossing-number pipeline unchanged
    (exp-py-minutiae/minutiae.py).  Its module-level W,H are patched to this
    image's size because every stage reads those globals.

    Counts are reported raw and restricted to the calibrated foreground mask
    eroded by 6 px -- minutiae in the untouched margins of the 225-px canvas
    are artefacts of the canvas, not of the finger."""
    h, w = img.shape
    if fg is None:
        fg = foreground(img)
    inner = MN._erode(fg, 6)
    out = {}
    variants = [("as-given", img)] + ([("inverted", 255.0 - img)] if both_polarities else [])
    for tag, im in variants:
        MN.W, MN.H = w, h
        t = MN.make_template(im, dict(cfg or {}))
        keep = [x for x in t.m if inner[int(round(x[1])), int(round(x[0]))]]
        nend = sum(1 for x in keep if x[3] == 1)
        out[tag] = {"n_total": len(t.m), "n_in_mask": len(keep),
                    "endings": nend, "bifurcations": len(keep) - nend}
    mm2 = float(fg.sum()) * (25.4 / dpi) ** 2
    best = max(out.values(), key=lambda d: d["n_in_mask"])
    out["best_n"] = best["n_in_mask"]
    out["area_mm2"] = mm2
    out["density_per_mm2"] = best["n_in_mask"] / mm2 if mm2 > 0 else float("nan")
    return out


# ------------------------------------------------------------- 7. verdict ---

def verdict(g, c, f, sm, dp, mi):
    ev = []

    # -- is there a fingerprint here at all?
    noise_hits = []
    if c["ridge_band"] < T_RIDGE_BAND:
        noise_hits.append(f"ridge-band energy ratio {c['ridge_band']:.3f} < {T_RIDGE_BAND} "
                          f"(press captures 0.311, gaussian noise 0.126)")
    if c["usable_frac"] < T_USABLE_FRAC:
        noise_hits.append(f"only {c['usable_frac']*100:.0f}% of blocks are foreground AND "
                          f"oriented (press 64%, noise 7%)")
    if f["estimable_frac"] < T_ESTIMABLE:
        noise_hits.append(f"a ridge pitch is measurable in only "
                          f"{f['estimable_frac']*100:.0f}% of blocks (press 67%, noise 5%)")
    if len(noise_hits) >= 2:
        return "NOISE - no ridge structure to assemble", noise_hits, []

    # -- duplicated / smeared
    dup_hits = []
    if dp.get("dup_frac", 0) > T_DUP_FRAC:
        dup_hits.append(f"{dp['dup_frac']*100:.0f}% of rows have a near-identical partner "
                        f"at lag {dp['dup_lag']} px -- content is replayed, which is what an "
                        f"OVER-estimated delta_y does (delta_y_est - delta_y_true = "
                        f"{dp['dup_lag']})")
    if sm.get("cont_median", 1.0) < T_CONT_MED:
        dup_hits.append(f"row-to-row continuity {sm['cont_median']:.2f} is far below the "
                        f"0.78-0.82 of the controls and the 0.855 of a real press capture")
    if dup_hits:
        if dp.get("dup_lag") == DY_MIN or (dp.get("dup_lag") or 9) <= 4:
            dup_hits.append("lag <= 4 px is the signature of the delta_y >= 2 clamp acting "
                            "on a finger that barely moved -- press, not swipe")
        return ("DUPLICATED / SMEARED assembly - the image is taller than the information "
                "in it"), dup_hits, []

    # -- geometry wrong but structure present
    seam_hits = []
    cr = sm.get("comb_ridge", {})
    if cr.get("z", 0) > T_COMB_Z and cr.get("pitch"):
        seam_hits.append(f"ridge continuity breaks periodically at pitch {cr['pitch']} px "
                         f"(z={cr['z']:.1f} vs permutation null, p={cr['p']:.3f}) -- that "
                         f"pitch is the frame slab height, so delta_y is mis-estimated")
    if sm.get("frac_flagged", 0) > T_FLAG_FRAC:
        seam_hits.append(f"{sm['n_flagged']}/{sm['n_boundaries']} row boundaries break "
                         f"(z>4); a correct assembly gives 1.5-1.8%")
    if not f["plausible_pitch"] and np.isfinite(f["period_px"]):
        d = "too coarse (rows replayed)" if f["period_mm"] > PERIOD_MM_HI else \
            "too fine (rows dropped)"
        seam_hits.append(f"ridge pitch {f['period_px']:.1f} px = {f['period_mm']:.2f} mm is "
                         f"outside the adult range {PERIOD_MM_LO}-{PERIOD_MM_HI} mm, {d}")
    if np.isfinite(f["band_cv"]) and f["band_cv"] > T_BAND_CV:
        seam_hits.append(f"ridge pitch varies {f['band_cv']*100:.0f}% (CV) between bands "
                         f"along the swipe; a correct assembly gives 5%")
    if np.isfinite(g["x_jump_p95"]) and g["x_jump_p95"] > 4.0:
        seam_hits.append(f"the finger column jumps {g['x_jump_p95']:.1f} px between rows "
                         f"(delta_x instability)")
    if seam_hits:
        return ("SEAM-BROKEN assembly - ridge structure is present but the inter-frame "
                "displacement is mis-estimated"), seam_hits, []

    # -- passes every failure test
    flags = []
    if not mi.get("skipped") and mi["best_n"] < MIN_MINUTIAE:
        flags.append(f"only {mi['best_n']} minutiae inside the mask; the forensic "
                     f"convention wants ~{MIN_MINUTIAE} MATCHED")
    if g["area_mm2"] < MIN_AREA_MM2:
        flags.append(f"captured area {g['area_mm2']:.1f} mm2 is only "
                     f"{g['area_vs_press']:.1f}x a press capture (15.6 mm2) -- short swipe")
    ev = [f"clarity: anisotropy {c['aniso_all']:.2f}, coherence {c['coh_all']:.2f}, "
          f"ridge band {c['ridge_band']:.2f}",
          f"pitch {f['period_px']:.1f} px = {f['period_mm']:.2f} mm, band CV " +
          (f"{f['band_cv']:.2f}" if np.isfinite(f["band_cv"])
           else "n/a (too short for band analysis)"),
          f"continuity {sm['cont_median']:.2f}, {sm['n_flagged']}/{sm['n_boundaries']} "
          f"boundaries flagged, no periodic break (z={cr.get('z', 0):.1f})",
          f"{dp['dup_frac']*100:.0f}% duplicated rows",
          f"{g['area_mm2']:.1f} mm2 = {g['area_vs_press']:.1f}x a press capture"]
    if not mi.get("skipped"):
        ev.append(f"{mi['best_n']} minutiae ({mi['density_per_mm2']:.2f}/mm2)")
    return "USABLE fingerprint image", ev, flags


# -------------------------------------------------------------- reporting ---

def analyse(img, dpi=DEFAULT_DPI, known_seams=None, fast=False, label=""):
    g = geometry(img, dpi)
    c = clarity(img)
    f = frequency(img, dpi)
    sm = seams(img, known_seams=known_seams)
    dp = duplication(img)
    if fast:
        mi = {"best_n": -1, "density_per_mm2": float("nan"),
              "area_mm2": g["area_mm2"], "skipped": True}
    else:
        mi = minutiae(img, dpi, fg=g["_fg"])
    v, reasons, flags = verdict(g, c, f, sm, dp, mi)
    return {"label": label,
            "geometry": {k: x for k, x in g.items() if not k.startswith("_")},
            "clarity": c, "frequency": f, "seams": sm, "duplication": dp,
            "minutiae": mi, "verdict": v, "verdict_reasons": reasons,
            "verdict_flags": flags}


def report(r, dpi):
    g, c, f, sm, dp, mi = (r["geometry"], r["clarity"], r["frequency"],
                           r["seams"], r["duplication"], r["minutiae"])
    P = print
    P("=" * 78)
    P(f"  {r['label']}")
    P("=" * 78)
    P("GEOMETRY")
    P(f"  image             {g['w']} x {g['h']} px = {g['mm_w']:.1f} x {g['mm_h']:.1f} mm "
      f"at {dpi:.0f} dpi")
    P(f"  full rectangle    {g['rect_mm2']:.1f} mm2   (NOT captured area)")
    P(f"  finger area       {g['area_mm2']:.1f} mm2  = {g['area_vs_press']:.1f}x a press "
      f"capture (15.6 mm2), {g['fg_frac']*100:.0f}% of the canvas")
    P(f"  travel            {g['rows_with_finger']} rows = {g['travel_mm']:.1f} mm along "
      f"the swipe axis")
    P(f"  data columns      {g['data_span'][0]}..{g['data_span'][1]} "
      f"({'225-px libfprint canvas' if g['canvas_is_libfprint'] else 'non-standard width'})")
    if np.isfinite(g["x_drift_px"]):
        P(f"  lateral drift     {g['x_drift_px']:.1f} px total, p95 row-to-row jump "
          f"{g['x_jump_p95']:.1f} px  (cumulative delta_x)")
    P("")
    P("RIDGE CLARITY")
    P(f"  anisotropy        {c['aniso_all']:.3f} whole image, {c['aniso_fg']:.3f} over the "
      f"finger, {c['aniso_w']:.3f} energy-weighted")
    P(f"  orient coherence  {c['coh_all']:.3f} whole image, {c['coh_fg']:.3f} over the finger")
    P(f"  usable blocks     {c['usable_frac']*100:.0f}%   ridge-band ratio {c['ridge_band']:.3f}"
      f"   dyn range {c['dyn_range']:.0f}")
    P(f"  ridge direction   {c['ridge_dir_deg']:.0f} deg, circular spread "
      f"{c['orient_spread_deg']:.1f} deg")
    P("")
    P("RIDGE FREQUENCY")
    P(f"  ridge pitch       {f['period_px']:.2f} px = {f['period_mm']:.3f} mm    "
      f"plausible (0.38-0.62 mm): {f['plausible_pitch']}")
    P(f"  blocks estimable  {f['estimable_frac']*100:.0f}%")
    if f["n_bands"] >= 3:
        P(f"  along the swipe   {f['band_period_px']}")
        P(f"                    CV {f['band_cv']:.3f}, largest adjacent jump "
          f"{f['band_max_jump']*100:.0f}%   (a seam shows as a pitch discontinuity)")
    else:
        P(f"  along the swipe   image too short for band analysis ({f['n_bands']} bands)")
    P("")
    P(f"SEAMS   (expected slab pitch {DY_MIN}-{DY_MAX} px; ~25 seams in a full swipe)")
    if "error" in sm:
        P(f"  {sm['error']}")
    else:
        P(f"  row continuity    median {sm['cont_median']:.3f} (zero lag), p10 "
          f"{sm['cont_p10']:.3f}; free-lag median {sm['cont_freelag_median']:.3f}")
        P(f"  broken boundaries {sm['n_flagged']}/{sm['n_boundaries']} "
          f"({sm['frac_flagged']*100:.1f}%)   lateral lag changes {sm['lag_changes']}")
        for k, name in (("comb_ridge", "ridge break "), ("comb_photo", "photometric"),
                        ("comb_lateral", "lateral jump")):
            cb = sm[k]
            P(f"  comb {name}  pitch {str(cb['pitch']):>4s} px   score {cb['score']:6.2f}   "
              f"z {cb['z']:+6.1f}   p {cb['p']:.3f}")
        P(f"                    (a photometric-only comb is cosmetic and its pitch is the "
          f"delta_y the driver used)")
        if sm["seam_rows"]:
            P(f"  broken at rows    {sm['seam_rows'][:20]}"
              f"{' ...' if len(sm['seam_rows']) > 20 else ''}")
        if "gt_seam_z" in sm:
            P(f"  GROUND TRUTH      mean z {sm['gt_seam_z']:.2f} at the known seams vs "
              f"{sm['gt_other_z']:.2f} elsewhere; {sm['gt_detected_frac']*100:.0f}% detected")
    P("")
    P("DUPLICATION")
    if "error" in dp:
        P(f"  {dp['error']}")
    else:
        P(f"  replayed rows     {dp['dup_frac']*100:.1f}% have a near-identical partner "
          f"(r>0.95) at lag {dp['dup_lag']} px")
        P(f"  best partner      median r {dp['best_partner_median']:.3f}")
        P(f"  effective rank    {dp['erank']:.1f} over {dp['rows']} rows "
          f"(ratio {dp['erank_ratio']:.3f})  [context only]")
        P(f"  self-similarity   {dp['self_sim'][:14]}")
    P("")
    P("MINUTIAE  (crossing-number pipeline, exp-py-minutiae/minutiae.py, unchanged)")
    if mi.get("skipped"):
        P("  skipped (--fast)")
    else:
        for tag in ("as-given", "inverted"):
            if tag in mi:
                d = mi[tag]
                P(f"  {tag:9s}       {d['n_in_mask']:3d} in mask ({d['n_total']} raw), "
                  f"{d['endings']} endings / {d['bifurcations']} bifurcations")
        P(f"  best estimate     {mi['best_n']} minutiae over {mi['area_mm2']:.1f} mm2 "
          f"= {mi['density_per_mm2']:.2f}/mm2")
    P("")
    P(f"VERDICT: {r['verdict']}")
    for x in r["verdict_reasons"]:
        P(f"    - {x}")
    for x in r["verdict_flags"]:
        P(f"    ! {x}")
    P("")


# --------------------------------------------------- controls / validation --

def synth_ridges(h, w, period=10.0, angle_deg=30.0, warp=1.5, noise=0.15, seed=0):
    """A fingerprint-LIKE oriented texture: a plane wave whose phase is warped
    by low-pass noise, so orientation and pitch vary the way ridge flow does.
    Not a fingerprint -- a control whose geometry is known exactly."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    a = math.radians(angle_deg)
    n = CQ.sep_blur(rng.standard_normal((h, w)).astype(np.float32), 12.0).astype(np.float64)
    n /= (n.std() + 1e-9)
    img = np.cos(2 * math.pi / period * (x * math.sin(a) + y * math.cos(a)) + warp * n)
    img = img + noise * rng.standard_normal((h, w))
    return np.clip(128 + 90 * img / (np.abs(img).max() + 1e-9), 0, 255).astype(np.float32)


def _thirds(frame):
    """The driver's per-frame normalisation (elan_process_frame_thirds): the
    30th and 65th percentiles are pinned to fixed output levels.  Reproduced so
    the controls carry the same photometric seam signature as real output."""
    s = np.sort(frame.ravel())
    l0, l1 = s[0], s[int(len(s) * 3 / 10)]
    l2, l3 = s[int(len(s) * 65 / 100)], s[-1]
    l1 = max(l1, l0 + 1); l2 = max(l2, l1 + 1); l3 = max(l3, l2 + 1)
    out = np.empty_like(frame)
    a, b, cc = frame < l1, (frame >= l1) & (frame < l2), frame >= l2
    out[a] = (frame[a] - l0) * 99 / (l1 - l0)
    out[b] = 99 + (frame[b] - l1) * 56 / (l2 - l1)
    out[cc] = 155 + (frame[cc] - l2) * 100 / (l3 - l2)
    return np.clip(out, 0, 255)


def assemble(frames, dys, dxs=None):
    """fpi_assemble_frames + aes_blit_stripe, reimplemented for the controls
    only: overwrite semantics, 225-px canvas, x centred then accumulating dx.
    The image under test comes from the driver, never from here."""
    n = len(frames)
    fh, fw = frames[0].shape
    dxs = list(dxs) if dxs is not None else [0] * n
    height = int(sum(dys)) + fh
    img = np.zeros((height, ASSEMBLED_W), np.float32)
    y, x, seam_rows = 0, (ASSEMBLED_W - fw) // 2, []
    for i, fr in enumerate(frames):
        y += int(dys[i]); x += int(dxs[i])
        if i:
            seam_rows.append(y)
        x0, fx0 = max(0, x), max(0, -x)
        x1 = min(ASSEMBLED_W, x + fw)
        y1 = min(height, y + fh)
        if x1 > x0 and y1 > y:
            img[y:y1, x0:x1] = fr[:y1 - y, fx0:fx0 + (x1 - x0)]
    return img, seam_rows


def controls():
    """Seven cases whose ground truth is known by construction.  These are what
    calibrate every threshold in this file; re-run --selftest after any change
    to the metrics and re-derive the constants if the numbers move."""
    rng = np.random.default_rng(3)
    ox = (ASSEMBLED_W - FRAME_W) // 2
    clean = synth_ridges(400, FRAME_W, period=10.0, seed=1)

    def frames(dy_true, n=30):
        out, k = [], 0
        while k + FRAME_H <= 400 and len(out) < n:
            out.append(_thirds(clean[k:k + FRAME_H]))
            k += dy_true
        return out

    big = np.zeros((400, ASSEMBLED_W), np.float32)
    big[:, ox:ox + FRAME_W] = clean
    out = [("CONTROL A: clean synthetic ridges, never sliced (truth: perfect)", big, None)]

    fB, sB = assemble(frames(6), [0] + [6] * 29)
    out.append(("CONTROL B: correctly assembled, dy_true = dy_est = 6 "
                "(truth: perfect geometry, per-frame normalisation)", fB, sB))

    fC, sC = assemble(frames(3), [0] + [6] * 29)
    out.append(("CONTROL C: delta_y OVER-estimated 3 -> 6 "
                "(truth: rows replayed at lag 3, 2x too tall)", fC, sC))

    fD, sD = assemble(frames(6), [0] + [3] * 29)
    out.append(("CONTROL D: delta_y UNDER-estimated 6 -> 3 "
                "(truth: half the rows thrown away)", fD, sD))

    fr = frames(6)
    dys = [0] + list(rng.integers(DY_MIN, 11, len(fr) - 1))
    dxs = [0] + list(rng.integers(-2, 3, len(fr) - 1))
    fE, sE = assemble(fr, dys, dxs)
    out.append(("CONTROL E: delta_y/delta_x jittered around the truth "
                "(truth: irregular seams, no periodicity)", fE, sE))

    pgms = sorted((DATASET / "right-index").glob("*.pgm"))
    if pgms:
        base = load_pgm(pgms[0])
        st = [_thirds(base + rng.normal(0, 2.0, base.shape).astype(np.float32))
              for _ in range(26)]
        fF, sF = assemble(st, [0] + [DY_MIN] * 25)
        out.append(("CONTROL F: REAL press capture, finger stationary, delta_y clamped "
                    "to 2 (truth: one frame stacked 26 times)", fF, sF))

    nz = np.zeros((300, ASSEMBLED_W), np.float32)
    nz[:, ox:ox + FRAME_W] = np.clip(128 + 40 * rng.standard_normal((300, FRAME_W)), 0, 255)
    out.append(("CONTROL G: gaussian noise through the same canvas", nz, None))
    return out


def baseline(dpi, fast=False):
    """Reference numbers from the 45 real PRESS captures.  They are single
    150x52 frames, NOT assemblies: seam and duplication figures are printed
    only so the swipe numbers have something to sit beside -- one frame cannot
    have an inter-frame seam.  Area, clarity, pitch and minutiae ARE the
    comparison that matters."""
    rows = []
    for d in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        for p in sorted(d.glob("*.pgm")):
            img = load_pgm(p)
            g, c, f = geometry(img, dpi), clarity(img), frequency(img, dpi)
            sm, dp = seams(img), duplication(img)
            mi = ({"best_n": -1, "density_per_mm2": float("nan")} if fast
                  else minutiae(img, dpi, fg=g["_fg"]))
            rows.append({
                "set": d.name, "name": p.name,
                "area_mm2": g["area_mm2"], "fg_frac": g["fg_frac"],
                "aniso_all": c["aniso_all"], "aniso_fg": c["aniso_fg"],
                "coh_all": c["coh_all"], "usable": c["usable_frac"],
                "ridge_band": c["ridge_band"], "spread": c["orient_spread_deg"],
                "period_px": f["period_px"], "period_mm": f["period_mm"],
                "estimable": f["estimable_frac"],
                "cont_med": sm.get("cont_median", float("nan")),
                "frac_flagged": sm.get("frac_flagged", float("nan")),
                "comb_z": sm.get("comb_ridge", {}).get("z", float("nan")),
                "dup_frac": dp.get("dup_frac", float("nan")),
                "erank_ratio": dp.get("erank_ratio", float("nan")),
                "minutiae": mi["best_n"], "density": mi["density_per_mm2"],
            })
    return rows


def _stats(rows, key):
    v = np.array([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "n/a"
    return (f"median {np.median(v):8.3f}   p10 {np.percentile(v, 10):8.3f}   "
            f"p90 {np.percentile(v, 90):8.3f}")


def main():
    ap = argparse.ArgumentParser(
        description="Judge an assembled ELAN 0c6e swipe image.",
        epilog="Run --baseline and --selftest first; they print the numbers the "
               "verdict thresholds are derived from.")
    ap.add_argument("images", nargs="*", help="assembled PGM(s)")
    ap.add_argument("--dpi", type=float, default=DEFAULT_DPI)
    ap.add_argument("--seams", help="known seam rows 'a,b,c', or a file of delta_y "
                                    "values (one per line) dumped by the capture tool")
    ap.add_argument("--fast", action="store_true", help="skip the minutiae stage")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="press-capture reference numbers from the dataset")
    ap.add_argument("--selftest", action="store_true",
                    help="run the controls whose ground truth is known")
    ap.add_argument("--dump-controls", metavar="DIR",
                    help="also write the control images as PGMs")
    a = ap.parse_args()

    if a.baseline:
        rows = baseline(a.dpi, fast=a.fast)
        print("=" * 78)
        print(f"  PRESS BASELINE -- {len(rows)} real captures, 150x52, {a.dpi:.0f} dpi")
        print("=" * 78)
        print("  Single frames, not assemblies.  Seam/duplication rows are context")
        print("  only: one frame has no inter-frame seam, so those numbers show what")
        print("  the detectors read off a KNOWN-GOOD image and are the false-positive")
        print("  floor.  Area, clarity, pitch and minutiae are the real comparison.")
        print()
        for k, lbl in (("area_mm2", "finger area mm2"), ("fg_frac", "foreground frac"),
                       ("aniso_all", "anisotropy (all)"), ("aniso_fg", "anisotropy (fg)"),
                       ("coh_all", "orient coherence"), ("usable", "usable block frac"),
                       ("ridge_band", "ridge-band ratio"), ("spread", "orient spread deg"),
                       ("period_px", "ridge pitch px"), ("period_mm", "ridge pitch mm"),
                       ("estimable", "estimable blocks"),
                       ("cont_med", "row continuity"), ("frac_flagged", "flagged boundaries"),
                       ("comb_z", "seam comb z"), ("dup_frac", "duplicated rows"),
                       ("erank_ratio", "erank / rows"),
                       ("minutiae", "minutiae in mask"), ("density", "minutiae / mm2")):
            print(f"  {lbl:22s} {_stats(rows, k)}")
        print()
        for d in sorted({r["set"] for r in rows}):
            sub = [r for r in rows if r["set"] == d]
            print(f"  {d:20s} n={len(sub):2d}  area {np.median([r['area_mm2'] for r in sub]):5.1f} mm2"
                  f"  aniso {np.median([r['aniso_all'] for r in sub]):.2f}"
                  f"  pitch {np.median([r['period_px'] for r in sub]):5.2f} px"
                  f"  minutiae {np.median([r['minutiae'] for r in sub]):.0f}")
        if a.json:
            print(json.dumps(rows, indent=2, default=str))
        return 0

    if a.selftest:
        res = []
        for label, img, sr in controls():
            if a.dump_controls:
                Path(a.dump_controls).mkdir(parents=True, exist_ok=True)
                save_pgm(Path(a.dump_controls) / (label.split(":")[0].replace(" ", "_") + ".pgm"), img)
            r = analyse(img, a.dpi, known_seams=sr, fast=a.fast, label=label)
            res.append(r)
            report(r, a.dpi)
        print("=" * 78)
        print("  CONTROL SUMMARY -- ground truth known by construction")
        print("=" * 78)
        hdr = (f"  {'ctl':3s} {'band':>5s} {'usbl':>5s} {'pitch':>6s} {'bCV':>5s} "
               f"{'cont':>5s} {'flag%':>6s} {'combZ':>6s} {'pitch':>5s} {'dup%':>5s} "
               f"{'lag':>4s} {'minu':>5s}  verdict")
        print(hdr)
        for r in res:
            sm, f, dp = r["seams"], r["frequency"], r["duplication"]
            cb = sm.get("comb_ridge", {})
            print(f"  {r['label'].split(':')[0][-1]:3s} "
                  f"{r['clarity']['ridge_band']:5.2f} {r['clarity']['usable_frac']:5.2f} "
                  f"{f['period_px']:6.2f} {f['band_cv']:5.2f} "
                  f"{sm.get('cont_median', float('nan')):5.2f} "
                  f"{sm.get('frac_flagged', 0)*100:6.1f} {cb.get('z', 0):6.1f} "
                  f"{str(cb.get('pitch')):>5s} {dp.get('dup_frac', 0)*100:5.1f} "
                  f"{str(dp.get('dup_lag')):>4s} {r['minutiae']['best_n']:5d}  "
                  f"{r['verdict'].split(' -')[0]}")
        print()
        print("  Expected: A and B USABLE (B may be flagged for area), C and F")
        print("  DUPLICATED, D and E SEAM-BROKEN, G NOISE.")
        if a.json:
            print(json.dumps(res, indent=2, default=str))
        return 0

    if not a.images:
        ap.print_help()
        return 1

    known = None
    if a.seams:
        if Path(a.seams).exists():
            known = list(np.cumsum([int(x) for x in Path(a.seams).read_text().split()]))
        else:
            known = [int(x) for x in a.seams.split(",")]

    outs = []
    for p in a.images:
        r = analyse(load_pgm(p), a.dpi, known_seams=known, fast=a.fast, label=str(p))
        outs.append(r)
        if not a.json:
            report(r, a.dpi)
    if a.json:
        print(json.dumps(outs, indent=2, default=str))
    return 0


VERDICT_NOTES = """
Every threshold and the measurement it came from.  All of it is reproducible on
this machine: --baseline over the 45 real press captures, --selftest over the
seven controls.  Re-run both after any change to a metric and re-derive.

  threshold          good side                                    bad side
  T_RIDGE_BAND 0.18  press 0.311 (p10 0.268), controls 0.24-0.33  noise 0.126
  T_USABLE_FRAC 0.20 press 0.64 (p10 0.35), controls 0.68-0.72    noise 0.07
  T_ESTIMABLE 0.20   press 0.67 (p10 0.54)                        noise 0.05
  T_DUP_FRAC 0.12    A/B/D/G 0.000, press 0.000, E 0.029          C 0.770, F 0.441
  T_COMB_Z 5.0       A -1.8, B -2.0, E 1.3, press p90 0.49        C 13.5, D 8.0
  T_FLAG_FRAC 0.05   A 0.015, B 0.018, press p90 0.020            C 0.142, D 0.217, E 0.108
  T_BAND_CV 0.18     A 0.07, B 0.06                               E 0.19, D 0.26
  T_CONT_MED 0.60    A 0.82, B/C/D/E 0.78-0.79, press 0.855       F 0.33
  PERIOD_MM 0.38-0.62  press 0.450 (p10 0.390, p90 0.515)         C 0.75, D 0.28, F 0.75
  MIN_MINUTIAE 12    forensic convention, NOT measured here, and it counts
                     MATCHED minutiae -- 12 detected is a floor, not a pass.
  MIN_AREA_MM2 40    2.5x the 16.1 mm2 median of a real press capture.

False-alarm rate, measured: put all 45 real press captures through the full
verdict and 44 come back USABLE (flagged for tiny area and too few minutiae,
which is correct -- they are 16 mm2).  One fails the pitch rule at 0.36 mm.
So the failure detectors fire on ~2% of known-good real images.

Sensitivity, measured: on CONTROL C the --seams ground-truth mode detects
100% of the 29 known seams, mean z 15.16 at the seams against 0.48 elsewhere,
and the comb recovers the true slab pitch of 6 px.  On CONTROL B, where the
assembly is geometrically correct, the same known seams score z 2.0 -- the
detector distinguishes "there is a frame boundary here" from "the frame
boundary is wrong", which is the whole point.

What none of this establishes: that a real swipe will match.  There is no
swipe data.  A "USABLE" verdict means only that the image survived every
failure mode that could be defined in advance.
"""

if __name__ == "__main__":
    sys.exit(main())
