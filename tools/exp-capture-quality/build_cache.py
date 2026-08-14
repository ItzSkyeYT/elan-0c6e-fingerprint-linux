#!/usr/bin/env python3
"""
Stage 1: compute, once, everything the analysis needs.

  * per-image quality metrics for all 45 captures
  * the full 45x45 LCN+NCC+rotation score matrix (the d'=1.54/1.94 baseline
    matcher), plus the best alignment for each pair
  * a second, WIDE-window alignment pass used only to estimate how far
    placement actually moves and how much area two presses really share

Writes exp-capture-quality/cache.npz
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(TOOLS)

exec(open("matcher-lab.py").read().split("def main(")[0])   # noqa

from fastncc2 import ncc_best_align, overlap_stats          # noqa: E402
import quality as Q                                          # noqa: E402

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LABELS = ["right-index", "right-index-cover", "right-middle"]

# baseline matcher parameters == matcher-lab's m_ncc_lcn_rot
BASE = dict(max_dx=20, max_dy=8, min_overlap=3500, max_rot=12, rot_step=4)
# wide pass, only for placement/overlap measurement. min_overlap=1500 lets a
# 20%-overlap fluke win, so this OVERSTATES displacement -- kept for contrast.
WIDE = dict(max_dx=60, max_dy=20, min_overlap=1500, max_rot=16, rot_step=4)
# honest displacement estimate: wide translation search but a large minimum
# overlap, so a small-overlap coincidence can never be the reported alignment.
WIDE_STRICT = dict(max_dx=60, max_dy=20, min_overlap=4500, max_rot=16, rot_step=4)


def main():
    ds = load_dataset(DATASET)                               # noqa: F821
    names, labels, imgs = [], [], []
    for li, lbl in enumerate(LABELS):
        for n, im in ds[lbl]:
            names.append(n)
            labels.append(li)
            imgs.append(im)
    N = len(imgs)
    print(f"{N} images: " + ", ".join(f"{l}={labels.count(i)}"
                                      for i, l in enumerate(LABELS)))

    print("computing quality metrics ...", flush=True)
    t = time.time()
    qkeys = None
    qrows = []
    masks = []
    for im in imgs:
        q = Q.image_quality(im)
        if qkeys is None:
            qkeys = sorted(q)
        qrows.append([q[k] for k in qkeys])
        masks.append(Q.foreground_mask_px(im))
    qual = np.array(qrows, dtype=np.float64)
    masks = np.array(masks)
    print(f"  {time.time()-t:.1f}s")

    print("pre-filtering (LCN) ...", flush=True)
    L = [local_contrast_norm(im) for im in imgs]              # noqa: F821

    for tag, params in (("base", BASE), ("wide", WIDE),
                        ("widestrict", WIDE_STRICT)):
        print(f"scoring {tag} window {params} ...", flush=True)
        t = time.time()
        S = np.full((N, N), np.nan)
        DX = np.zeros((N, N), dtype=np.int32)
        DY = np.zeros((N, N), dtype=np.int32)
        DEG = np.zeros((N, N), dtype=np.int32)
        GEO = np.zeros((N, N), dtype=np.int32)
        BOTH = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                s, dx, dy, deg = ncc_best_align(
                    L[i], L[j], params["max_dx"], params["max_dy"],
                    params["min_overlap"], params["max_rot"],
                    params["rot_step"], rotate=_rotate)        # noqa: F821
                geo, both, _, _ = overlap_stats(masks[i], masks[j], dx, dy)
                S[i, j] = S[j, i] = s
                DX[i, j], DY[i, j], DEG[i, j] = dx, dy, deg
                DX[j, i], DY[j, i], DEG[j, i] = -dx, -dy, -deg
                GEO[i, j] = GEO[j, i] = geo
                BOTH[i, j] = BOTH[j, i] = both
            print(f"\r  row {i+1}/{N}", end="", flush=True)
        print(f"   {time.time()-t:.1f}s")
        np.savez(HERE / f"scores_{tag}.npz", S=S, DX=DX, DY=DY, DEG=DEG,
                 GEO=GEO, BOTH=BOTH)

    np.savez(HERE / "cache.npz", names=np.array(names), labels=np.array(labels),
             qual=qual, qkeys=np.array(qkeys), fg_px=masks.sum(axis=(1, 2)))
    print("wrote cache.npz, scores_base.npz, scores_wide.npz")


if __name__ == "__main__":
    main()
