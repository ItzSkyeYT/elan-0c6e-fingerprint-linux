#!/usr/bin/env python3
"""
Evaluate cached pair-score matrices, and sweep the fusion weight between the
orientation-field score and the LCN+NCC score.

Fusion happens at PAIR level and the max over the template set is taken after,
which is the only correct order (max of a sum != sum of maxes).

Both scores are correlation coefficients on a comparable [-1,1] scale, so the
fusion is a plain weighted sum with no data-derived normalisation -- nothing is
fitted to the labels except the single weight w, whose whole sweep is printed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                      # noqa
from run import load_flat, show, CACHE    # noqa


def load(name):
    return np.load(os.path.join(CACHE, name + ".npy"))


def main():
    imgs, labels = load_flat()
    names = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith(".npy"))
    mats = {n: load(n) for n in names}

    print("== all cached matrices ==")
    res = []
    for n in names:
        res.append(show(n, mats[n], labels))

    print("\n" + "=" * 84)
    print(f"  {'matcher':<28}{'A d-prime':>11}{'A EER':>9}{'A FAR@10':>10}"
          f"{'B d-prime':>11}{'B EER':>8}{'B FAR@10':>10}")
    for r in sorted(res, key=lambda r: -r["B"][0]):
        print(f"  {r['name']:<28}{r['A'][0][0]:11.2f}{r['A'][0][1]*100:8.1f}%"
              f"{r['A'][0][2]*100:9.1f}%{r['B'][0]:11.2f}{r['B'][1]*100:7.1f}%"
              f"{r['B'][2]*100:9.1f}%")

    base = mats["ncc_lcn_rot"]
    others = [n for n in names if n != "ncc_lcn_rot"]
    print("\n== fusion sweep:  S = (1-w) * NCC + w * orientation ==")
    print(f"  {'partner':<20}" + "".join(f"{w:>7.2f}" for w in np.arange(0, 1.01, 0.1)))
    best = None
    for n in others:
        rowA, rowB = [], []
        for w in np.arange(0, 1.01, 0.1):
            F = (1 - w) * base + w * mats[n]
            r = show("", F, labels, quiet=True)
            rowA.append(r["A"][0][0])
            rowB.append(r["B"][0])
            if best is None or r["B"][0] > best[0]:
                best = (r["B"][0], n, w, r)
        print(f"  {n:<20}A d'" + "".join(f"{v:7.2f}" for v in rowA))
        print(f"  {'':<20}B d'" + "".join(f"{v:7.2f}" for v in rowB))

    print(f"\n  best fusion by scenario-B d': {best[1]} at w={best[2]:.2f}")
    F = (1 - best[2]) * base + best[2] * mats[best[1]]
    show(f"FUSED  ncc + {best[2]:.2f}*{best[1]}", F, labels)


if __name__ == "__main__":
    main()
