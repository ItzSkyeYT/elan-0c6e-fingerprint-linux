#!/usr/bin/env python3
"""Reference NCC baselines under the exact same two protocols, so the
descriptor numbers are compared against something measured here rather than
against a remembered figure."""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import enh
import evaluate as EV
from fastncc import ncc_all_shifts


def ncc_rot(a, b, max_dx=25, max_dy=10, min_overlap=2600, max_rot=12, rot_step=4):
    best = -1.0
    for deg in range(-max_rot, max_rot + 1, rot_step):
        br = enh._rotate(b, deg) if deg else b
        c = ncc_all_shifts(a, br, max_dx, max_dy, min_overlap)
        if c > best:
            best = c
    return best


def run(prep, tag, **kw):
    imgs, labels, names = EV.load()
    P = [prep(im) for im in imgs]
    n = len(P)
    S = np.zeros((n, n), np.float32)
    t0 = time.time()
    for i in range(n):
        for j in range(i, n):
            S[i, j] = S[j, i] = ncc_rot(P[i], P[j], **kw)
    a, asd = EV.protocol_a(S, labels, 6, 16)
    b, g, im = EV.protocol_b(S, labels)
    print(f"{tag:28s} ({time.time()-t0:.0f}s)")
    print(f"   A: d'={a['dprime']:.2f}+/-{asd['dprime']:.2f}  EER={a['eer']*100:.1f}%  FAR@10={a['far10']*100:.1f}%")
    print(f"   B: d'={b['dprime']:.2f}  EER={b['eer']*100:.1f}%  FAR@10={b['far10']*100:.1f}%")
    return S


if __name__ == "__main__":
    run(lambda im: enh.local_contrast_norm(im), "LCN + rot NCC (old baseline)")
    run(lambda im: enh.gabor_enhance(im), "matcher-lab gabor + rot NCC")
    run(lambda im: enh.gabor(im)[0], "FIXED gabor + rot NCC")
