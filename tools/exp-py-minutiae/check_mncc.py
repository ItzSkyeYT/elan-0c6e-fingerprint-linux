#!/usr/bin/env python3
"""Validate masked_ncc_map against the trusted fastncc implementation, and
report how much of each capture is actually finger."""
import sys, math
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from fastncc import ncc_all_shifts
from common import load_ds
import minutiae as M
import mncc

ds = load_ds()
ORDER = ["right-index", "right-index-cover", "right-middle"]

imgs, labels = [], []
for lbl in ORDER:
    for nm, im in ds[lbl]:
        imgs.append(im); labels.append(lbl)

# 1) all-ones mask must reproduce fastncc exactly
ones = np.ones((52, 150), bool)
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(12):
    i, j = rng.integers(0, len(imgs), 2)
    a = M.local_contrast_norm(imgs[i]); b = M.local_contrast_norm(imgs[j])
    ref = ncc_all_shifts(a, b, 20, 8, 3500)
    s, n = mncc.masked_ncc_map(a, ones, b, ones, 20, 8, 3500)
    worst = max(worst, abs(ref - s.max()))
print(f"validation: max |fastncc - masked(all-ones)| = {worst:.2e}   (want <1e-9)")

# 2) foreground coverage
print("\nforeground fraction per label (block-variance segmentation):")
for lbl in ORDER:
    fr = []
    for nm, im in ds[lbl]:
        m = M.segment(im)
        fr.append(m.mean())
    print(f"  {lbl:20s} mean {np.mean(fr):.2f}  min {np.min(fr):.2f}  max {np.max(fr):.2f}")

# 3) intensity/greyscale stats: are impostor captures systematically different?
print("\nraw capture statistics:")
for lbl in ORDER:
    st = [ (im.mean(), im.std()) for nm, im in ds[lbl] ]
    st = np.array(st)
    print(f"  {lbl:20s} mean {st[:,0].mean():6.1f}   sd {st[:,1].mean():5.1f}")
