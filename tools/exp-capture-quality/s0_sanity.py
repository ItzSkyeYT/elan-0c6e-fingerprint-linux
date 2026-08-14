#!/usr/bin/env python3
"""
Stage 0 -- independent sanity checks before any number is trusted.

  1. fastncc2.ncc_map must agree with the naive per-shift NCC in matcher-lab.
  2. A matcher must score ~1.0 comparing an image with itself.
  3. The genuine/impostor split must be what the protocol says it is.
  4. protocol_A / protocol_B must never let a probe be its own template.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(TOOLS)
exec(open("matcher-lab.py").read().split("def main(")[0])      # noqa

from fastncc2 import ncc_map, ncc_best_align                   # noqa: E402
import evalproto as E                                          # noqa: E402

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LABELS = ["right-index", "right-index-cover", "right-middle"]

ds = load_dataset(DATASET)                                     # noqa: F821
names, labels, imgs = [], [], []
for li, lbl in enumerate(LABELS):
    for n, im in ds[lbl]:
        names.append(n); labels.append(li); imgs.append(im)
labels = np.array(labels)
N = len(imgs)
print(f"loaded {N} images: " +
      ", ".join(f"{l}={int((labels==i).sum())}" for i, l in enumerate(LABELS)))
assert (labels == 0).sum() == 12 and (labels == 1).sum() == 19 and (labels == 2).sum() == 14
for im in imgs:
    assert im.shape == (52, 150), im.shape

print("\n[1] fastncc2 vs naive NCC, 6 random pairs, all shifts in +-20/+-8")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(6):
    i, j = rng.integers(0, N, 2)
    a, b = local_contrast_norm(imgs[i]), local_contrast_norm(imgs[j])   # noqa: F821
    c, dys, dxs, n = ncc_map(a, b, 20, 8, 3500)
    for yi, dy in enumerate(dys):
        for xi, dx in enumerate(dxs):
            ref = _ncc_shift(a, b, int(dx), int(dy), 3500)              # noqa: F821
            got = c[yi, xi]
            if not np.isfinite(got):
                assert ref == -1.0, (dx, dy, ref)
                continue
            worst = max(worst, abs(ref - got))
print(f"    max |fast - naive| = {worst:.3e}   {'OK' if worst < 1e-6 else 'FAIL'}")
assert worst < 1e-6

print("\n[2] self-comparison must score ~1.0")
for k in (0, 20, 40):
    a = local_contrast_norm(imgs[k])                                    # noqa: F821
    s, dx, dy, deg = ncc_best_align(a, a, 20, 8, 3500, 12, 4, rotate=_rotate)  # noqa: F821
    print(f"    img {k:2d} {names[k]:<28} self-score {s:.6f} at dx={dx} dy={dy} deg={deg}")
    assert s > 0.999, s

print("\n[3] a shifted copy of an image must still score ~1.0 (alignment works)")
a = imgs[3]
b = np.roll(a, (3, -7), axis=(0, 1))
s, dx, dy, deg = ncc_best_align(local_contrast_norm(a), local_contrast_norm(b),  # noqa: F821
                                20, 8, 3500, 0, 4, rotate=_rotate)
print(f"    rolled by (dy=3,dx=-7): score {s:.4f} recovered dx={dx} dy={dy}")
assert s > 0.95

print("\n[4] protocol leak guard: a NaN diagonal must raise if a probe self-matches")
S = np.random.rand(N, N); np.fill_diagonal(S, np.nan)
gen = np.where(labels < 2)[0]; imp = np.where(labels == 2)[0]
try:
    E._maxscore(S, gen[:5], int(gen[2]))
    print("    FAIL: no exception raised")
    sys.exit(1)
except AssertionError:
    print("    OK: self-comparison raised AssertionError")

rA = E.protocol_A(S, gen, imp, n_enroll=10, reps=8, seed=1)
rB = E.protocol_B(S, np.where(labels == 1)[0], np.where(labels == 0)[0], imp)
print(f"    random-score control: A d'={rA['dprime']:+.3f}  B d'={rB['dprime']:+.3f}"
      "   (must be ~0)")
assert abs(rA["dprime"]) < 0.8 and abs(rB["dprime"]) < 1.2

print("\nALL SANITY CHECKS PASSED")
