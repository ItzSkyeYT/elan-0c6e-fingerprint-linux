#!/usr/bin/env python3
"""Sanity checks that must pass before any number below is believable."""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *            # noqa
from oflib import _corr_planes
sys.path.insert(0, TOOLS)
import fastncc

DS = os.path.expanduser("~/.local/share/elan-fp/dataset")
ds = load_dataset(DS)
idx = ds["right-index"]
imgs = [im for _, im in idx]

print("== 1. vectorised NCC reproduces fastncc.ncc_all_shifts exactly ==")
worst = 0.0
for i in range(3):
    for j in range(3):
        a = local_contrast_norm(imgs[i]); b = local_contrast_norm(imgs[j])
        r1 = fastncc.ncc_all_shifts(a, b, 20, 8, 3500)
        r2 = ncc_all_shifts_vec(a, b, 20, 8, 3500)
        worst = max(worst, abs(r1 - r2))
print(f"   max |difference| over 9 pairs: {worst:.3e}   {'OK' if worst < 1e-8 else 'FAIL'}")

print("\n== 2. orientation score of an image with ITSELF is ~ +1 ==")
ph, pw, iy, ix = _corr_planes(20, 8)
for block in (4, 6, 8):
    ss = []
    for im in imgs[:5]:
        f = make_field(im, block=block)
        s, dx, dy = orient_score(fft_pack(f, ph, pw),
                                 fft_pack(f, ph, pw, conj=True),
                                 ph, pw, iy, ix, min_blocks=20, block=block)
        ss.append(s)
    print(f"   block={block}: self-score {min(ss):.4f}..{max(ss):.4f}"
          f"   {'OK' if min(ss) > 0.99 else 'FAIL'}")

print("\n== 3. doubled-angle domain: a field vs itself rotated 90deg must be ~ -1 ==")
# 90deg rotation turns every orientation into its perpendicular; in the doubled
# angle domain that is exactly antipodal.  If we had compared raw angles this
# test would report ~0 and 180deg-apart ridges would look maximally different.
im = imgs[0]
f0 = make_field(im, block=8)
th = 0.5 * np.arctan2(f0.ay, f0.ax)
perp = th + math.pi / 2
ax2 = f0.r * np.cos(2 * perp); ay2 = f0.r * np.sin(2 * perp)
fp = Field(ax2, ay2, f0.r, f0.v)


def zero_shift(fa, fb):
    """The score at dx=dy=0 only -- orient_score maximises over shifts, and the
    best shift against a perpendicular field is naturally better than -1."""
    return float((fa.ax * fb.ax + fa.ay * fb.ay).sum() / (fa.r * fb.r).sum())


s = zero_shift(f0, fp)
print(f"   perpendicular-field score {s:.4f}   {'OK' if s < -0.99 else 'FAIL'}")
th180 = th + math.pi           # a 180deg-equivalent orientation: must score +1
ax3 = f0.r * np.cos(2 * th180); ay3 = f0.r * np.sin(2 * th180)
f3 = Field(ax3, ay3, f0.r, f0.v)
s3 = zero_shift(f0, f3)
print(f"   180deg-shifted field score {s3:.4f}   {'OK' if s3 > 0.99 else 'FAIL'}")

print("\n== 4. rotation search recovers a synthetic rotation ==")
# rotate the image by +6deg, then search: the best rotation of the probe should
# be about -6deg.  This is the test that catches 'resampled the field but forgot
# to add the rotation angle' -- we recompute the field from the rotated image,
# so the angles come out right by construction.
for truth in (-8, -4, 4, 8):
    rot_img, _ = rotate_valid(local_contrast_norm(im), truth)
    fa = make_field(im, block=8)
    FA = fft_pack(fa, ph, pw)
    best, bestdeg = -2, None
    for deg in range(-12, 13, 2):
        fb = make_field(None, block=8, deg=deg, pre=rot_img)
        s, _, _ = orient_score(FA, fft_pack(fb, ph, pw, conj=True),
                               ph, pw, iy, ix, 20, 8)
        if s > best:
            best, bestdeg = s, deg
    ok = abs(bestdeg + truth) <= 2
    print(f"   image rotated {truth:+3d}deg -> best probe rotation {bestdeg:+3d}deg "
          f"(score {best:.3f})  {'OK' if ok else 'FAIL'}")

print("\n== 5. dataset labelling ==")
for k, v in ds.items():
    print(f"   {k:<20} {len(v)}")
