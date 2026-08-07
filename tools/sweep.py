#!/usr/bin/env python3
"""Parameter sweep over matcher variants, evaluated as template-set vs probe
(which is how the driver actually works). Run from the tools/ directory.

Uses the FFT-based NCC in fastncc.py, validated exact against the naive loop.
"""
import math
import sys
import time

import numpy as np

# reuse loading / metrics / filtering, but not main()
exec(open("matcher-lab.py").read().split("def main(")[0])
from fastncc import ncc_all_shifts

DS = load_dataset("/home/melb/.local/share/elan-fp/dataset")
GEN = "right-index"


def gabor2(img, freq=0.11, n_orient=8, ksize=9, sigma=2.5, perp=True):
    """Gabor enhancement with an explicit orientation convention, so the
    90-degree ambiguity is settled by measurement rather than argument."""
    img = local_contrast_norm(img)
    theta = orientation_field(img)
    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    angles = np.linspace(0, math.pi, n_orient, endpoint=False)
    resp = np.empty((n_orient,) + img.shape, dtype=np.float32)
    p = np.pad(img, r, mode="edge")
    for ai, a in enumerate(angles):
        xr = xx * math.cos(a) + yy * math.sin(a)
        yr = -xx * math.sin(a) + yy * math.cos(a)
        k = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * math.pi * freq * xr)
        k -= k.mean()
        acc = np.zeros_like(img)
        for dy in range(ksize):
            for dx in range(ksize):
                acc += k[dy, dx] * p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
        resp[ai] = acc
    d = (theta + math.pi / 2) % math.pi if perp else theta % math.pi
    idx = np.clip((d / math.pi * n_orient).astype(np.int32), 0, n_orient - 1)
    return normalize_global(np.take_along_axis(resp, idx[None, ...], axis=0)[0])


def binarize(img):
    """Sign of the enhanced image: keeps ridge topology, discards amplitude."""
    e = local_contrast_norm(img)
    return np.where(e > 0, 1.0, -1.0).astype(np.float32)


# Preprocessing is expensive and reused across shifts, so cache per (name, image).
_CACHE = {}


def prep(name, fn, img):
    key = (name, id(img))
    if key not in _CACHE:
        _CACHE[key] = fn(img)
    return _CACHE[key]


def matcher(pname, pfn, dx, dy, ov, rot, rstep):
    def fn(a, b):
        pa = prep(pname, pfn, a)
        pb_raw = prep(pname, pfn, b)
        if rot == 0:
            return ncc_all_shifts(pa, pb_raw, dx, dy, ov)
        best = -1.0
        for deg in range(-rot, rot + 1, rstep):
            pb = _rotate(pb_raw, deg) if deg else pb_raw
            c = ncc_all_shifts(pa, pb, dx, dy, ov)
            if c > best:
                best = c
        return best
    return fn


LCN = lambda s: (f"lcn{s}", lambda x: local_contrast_norm(x, s))

VARIANTS = []
for s in (3.0, 6.0, 9.0, 14.0):
    n, f = LCN(s)
    VARIANTS.append((f"lcn(s={s})+rot12", matcher(n, f, 20, 8, 3500, 12, 4)))
for rot, st in ((0, 4), (8, 4), (12, 2), (18, 3), (24, 4), (30, 5)):
    n, f = LCN(6.0)
    VARIANTS.append((f"lcn6+rot{rot}/{st}", matcher(n, f, 20, 8, 3500, rot, st)))
for dx, dy, ov in ((15, 6, 5000), (25, 10, 3500), (35, 12, 3000), (50, 18, 2200)):
    n, f = LCN(6.0)
    VARIANTS.append((f"lcn6+t{dx}x{dy}/ov{ov}", matcher(n, f, dx, dy, ov, 12, 3)))
for perp in (True, False):
    VARIANTS.append((f"gabor(perp={perp})+rot12",
                     matcher(f"gab{perp}", lambda x, p=perp: gabor2(x, perp=p),
                             20, 8, 3500, 12, 3)))
for lo, hi in ((0.8, 3.0), (1.2, 5.0)):
    VARIANTS.append((f"bandpass({lo},{hi})+rot12",
                     matcher(f"bp{lo}_{hi}", lambda x, l=lo, h=hi: bandpass(x, l, h),
                             20, 8, 3500, 12, 3)))
VARIANTS.append(("binarized+rot12",
                 matcher("bin", binarize, 20, 8, 3500, 12, 3)))

results = []
t0 = time.time()
for name, fn in VARIANTS:
    gen, imp = evaluate_template(fn, DS, GEN, n_enroll=8)
    d = dprime(gen, imp)
    e, _ = eer(gen, imp)
    f10, _ = far_at_frr(gen, imp, 0.10)
    results.append((d, e, f10, name))
    print(f"  {name:<28} d'={d:5.2f}  EER={e*100:5.1f}%  FAR@10%FRR={f10*100:5.1f}%",
          flush=True)

print(f"\n  ({time.time()-t0:.0f}s total)")
print("=" * 68)
print(f"  {'variant':<30}{'d-prime':>9}{'EER':>9}{'FAR@10%':>10}")
print("  " + "-" * 64)
for d, e, f10, name in sorted(results, reverse=True):
    print(f"  {name:<30}{d:9.2f}{e*100:8.1f}%{f10*100:9.1f}%")
