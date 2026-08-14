#!/usr/bin/env python3
"""Parameter sweep for the centred orientation-field matcher.

Each config gets its own cached matrix so fuse.py can pick it up.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                      # noqa
from run import load_flat, show, cached, matrix_orient   # noqa

CONFIGS = {
    # block 3 was the best single-scale setting; explore around it
    "orientc_b3_p2":   dict(block=3, rel_pow=2.0),
    "orientc_b3_p4":   dict(block=3, rel_pow=4.0),
    "orientc_b3_q70":  dict(block=3, energy_q=0.70),
    "orientc_b3_mb60": dict(block=3, min_blocks=60),
    "orientc_b3_rot20": dict(block=3, max_rot=20, rot_step=4),
    "orientc_b3_rs2":  dict(block=3, rot_step=2),
    # block size x energy gate
    "orientc_b4_q20":  dict(block=4, energy_q=0.20),
    "orientc_b4_q60":  dict(block=4, energy_q=0.60),
    "orientc_b4_q80":  dict(block=4, energy_q=0.80),
    "orientc_b2":      dict(block=2),
    # reliability sharpening: let confident blocks dominate
    "orientc_b4_p2":   dict(block=4, rel_pow=2.0),
    "orientc_b4_p4":   dict(block=4, rel_pow=4.0),
    "orientc_b6_p2":   dict(block=6, rel_pow=2.0),
    # minimum overlap (in blocks) -- guards against tiny-overlap flukes
    "orientc_b4_mb60": dict(block=4, min_blocks=60),
    "orientc_b4_mb120": dict(block=4, min_blocks=120),
    "orientc_b4_mb10": dict(block=4, min_blocks=10),
    # search window
    "orientc_b4_wide": dict(block=4, max_dx=40, max_dy=14, min_blocks=40),
    "orientc_b4_rot20": dict(block=4, max_rot=20, rot_step=4),
    "orientc_b4_norot": dict(block=4, max_rot=0),
}


def main():
    imgs, labels = load_flat()
    names = sys.argv[1:] or list(CONFIGS)
    res = []
    for nm in names:
        kw = dict(CONFIGS[nm])
        kw.setdefault("centred", True)
        M = cached(nm, lambda kw=kw: matrix_orient(imgs, **kw))
        res.append(show(nm, M, labels))
    print("\n" + "=" * 74)
    print(f"  {'config':<22}{'A d-prime':>11}{'A EER':>9}{'B d-prime':>11}{'B EER':>8}")
    for r in sorted(res, key=lambda r: -r["B"][0]):
        print(f"  {r['name']:<22}{r['A'][0][0]:11.2f}{r['A'][0][1]*100:8.1f}%"
              f"{r['B'][0]:11.2f}{r['B'][1]*100:7.1f}%")


if __name__ == "__main__":
    main()
