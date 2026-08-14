#!/usr/bin/env python3
"""Extractor stability under a KNOWN synthetic transform.

If a 4-pixel shift of the same image already changes the minutiae set, the
extractor -- not the finger -- is the problem.  This separates the two.
"""
import math
import sys
import numpy as np

sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
import minutiae as M
from common import load_ds

W, H = 150, 52


def shift_img(img, dx, dy):
    out = np.zeros_like(img)
    xs = slice(max(0, dx), min(W, W + dx))
    xd = slice(max(0, -dx), min(W, W - dx))
    ys = slice(max(0, dy), min(H, H + dy))
    yd = slice(max(0, -dy), min(H, H - dy))
    out[ys, xs] = img[yd, xd]
    # fill the vacated strip with the edge value so segmentation isn't upset
    if dx > 0:
        out[:, :dx] = out[:, dx:dx + 1]
    elif dx < 0:
        out[:, dx:] = out[:, dx - 1:dx]
    if dy > 0:
        out[:dy, :] = out[dy:dy + 1, :]
    elif dy < 0:
        out[dy:, :] = out[dy - 1:dy, :]
    return out


def count_corr(A, B, dx, dy, tol=6.0, pad=8):
    """A shifted by (dx,dy) should coincide with B."""
    if len(A) == 0 or len(B) == 0:
        return 0, 0, 0
    ax, ay = A[:, 0] + dx, A[:, 1] + dy
    x0, x1 = max(0, dx) + pad, min(W, W + dx) - pad
    y0, y1 = max(0, dy) + pad, min(H, H + dy) - pad
    inA = (ax >= x0) & (ax < x1) & (ay >= y0) & (ay < y1)
    inB = (B[:, 0] >= x0) & (B[:, 0] < x1) & (B[:, 1] >= y0) & (B[:, 1] < y1)
    nA, nB = int(inA.sum()), int(inB.sum())
    if nA == 0 or nB == 0:
        return 0, nA, nB
    d = np.hypot(ax[inA][:, None] - B[inB, 0][None, :],
                 ay[inA][:, None] - B[inB, 1][None, :])
    used = np.zeros(nB, bool)
    m = 0
    for i in range(nA):
        j = int(np.argmin(np.where(used, 1e9, d[i])))
        if not used[j] and d[i, j] <= tol:
            used[j] = True
            m += 1
    return m, nA, nB


def main(cfg=None):
    cfg = cfg or {}
    ds = load_ds()
    imgs = [im for lbl in ds for _, im in ds[lbl]]
    res = []
    for img in imgs:
        t0 = M.make_template(img, cfg)
        for dx, dy in [(4, 0), (0, 3), (5, 2), (-6, -2)]:
            t1 = M.make_template(shift_img(img, dx, dy), cfg)
            m, nA, nB = count_corr(t0.m, t1.m, dx, dy)
            res.append((m, nA, nB))
    r = np.array(res, float)
    keep = r[:, 1] > 0
    rate = (r[keep, 0] / np.maximum(r[keep, 1], 1)).mean()
    print(f"  synthetic-shift stability: {r[:,0].mean():.2f} of {r[:,1].mean():.2f} "
          f"minutiae survive a pure translation  ({rate*100:.0f}% repeatable)")
    return rate


if __name__ == "__main__":
    main()
