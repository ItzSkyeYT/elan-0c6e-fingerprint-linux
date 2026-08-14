#!/usr/bin/env python3
"""Sweep: preprocessing x search window x surface statistic.

Everything is measured under the two required protocols with the shared
harness in eval.py, so the numbers are directly comparable to the baseline.
"""
import math
import sys
import time
import itertools

import numpy as np

sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr
from eval import scenario_A, scenario_B
import minutiae as M
import nccmap
from mncc import rotate

ORDER = ["right-index", "right-index-cover", "right-middle"]
W, H = 150, 52


# ------------------------------------------------------------ preprocessing

def pre_lcn(img):
    return M.local_contrast_norm(img, sigma=6.0)


def pre_gabor(img):
    n = M.local_contrast_norm(img, sigma=6.0)
    th, coh = M.orientation(n, block=8, smooth=2.5)
    return M.gabor(n, th, 0.110, n_orient=16, ksize=11, sx=4.0, sy=4.0)


def pre_gabor_bin(img):
    """Gabor response hard-limited: keeps ridge geometry, discards contrast."""
    e = pre_gabor(img)
    return np.tanh(e / (np.abs(e).mean() + 1e-6))


def orient_field(img, block=6, smooth=1.5):
    """Doubled-angle orientation vector field, unit length where coherent."""
    n = M.local_contrast_norm(img, sigma=6.0)
    g = M._sep_blur(n, 1.0)
    gx = np.gradient(g, axis=1)
    gy = np.gradient(g, axis=0)
    vx = M._sep_blur(2.0 * gx * gy, smooth)
    vy = M._sep_blur(gx ** 2 - gy ** 2, smooth)
    mag = np.sqrt(vx ** 2 + vy ** 2) + 1e-9
    return (vx / mag).astype(np.float64), (vy / mag).astype(np.float64)


PRE = {
    "lcn": pre_lcn,
    "gabor": pre_gabor,
    "gabtanh": pre_gabor_bin,
}


# ------------------------------------------------------------------ driver

def build_scores(ds, prefn, rots, max_dx, max_dy, min_overlap, corr_area=81.0):
    imgs, labels = [], []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            imgs.append(prefn(im))
            labels.append(lbl)
    P = [nccmap.prep(im, rots, max_dx, max_dy, rotate) for im in imgs]
    n = len(P)
    keys = ["rmax", "z", "psr", "zpsr"]
    S = {k: np.zeros((n, n)) for k in keys}
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            surf, npx = nccmap.surface(P[i], P[j], rots, max_dx, max_dy, min_overlap)
            st = nccmap.stats(surf, npx, corr_area)
            for k in keys:
                S[k][i, j] = S[k][j, i] = st[k]
    el = time.time() - t0
    for k in keys:
        np.fill_diagonal(S[k], np.nan)
    return S, labels, el


def ev(S, labels, n_enroll=6, trials=16):
    ga, ia, pt = scenario_A(S, labels, n_enroll, trials)
    gb, ib = scenario_B(S, labels)
    return (dprime(ga, ia), eer(ga, ia)[0], far_at_frr(ga, ia)[0],
            dprime(gb, ib), eer(gb, ib)[0], far_at_frr(gb, ib)[0])


def main():
    ds = load_ds()
    print(f"{'pre':9s} {'win':14s} {'rot':9s} {'stat':6s} | "
          f"{'A d':>6s} {'A EER':>6s} {'A F10':>6s} | {'B d':>6s} {'B EER':>6s} {'B F10':>6s}")
    print("-" * 92)
    configs = [
        ("lcn",     20, 8,  3500, (-12, 13, 4)),
        ("lcn",     40, 14, 3500, (-12, 13, 4)),
        ("lcn",     60, 20, 3000, (-12, 13, 4)),
        ("lcn",     60, 20, 2000, (-12, 13, 4)),
        ("gabor",   20, 8,  3500, (-12, 13, 4)),
        ("gabor",   40, 14, 3500, (-12, 13, 4)),
        ("gabor",   60, 20, 2000, (-12, 13, 4)),
        ("gabtanh", 40, 14, 3500, (-12, 13, 4)),
        ("gabtanh", 60, 20, 2000, (-12, 13, 4)),
    ]
    for pre, mdx, mdy, mo, (r0, r1, rs) in configs:
        rots = list(range(r0, r1, rs))
        S, labels, el = build_scores(ds, PRE[pre], rots, mdx, mdy, mo)
        for k in ("rmax", "z", "psr", "zpsr"):
            a = ev(S[k], labels)
            print(f"{pre:9s} dx{mdx:<3d}dy{mdy:<3d}o{mo:<5d} {str(rots[0])+'..'+str(rots[-1]):9s} "
                  f"{k:6s} | {a[0]:6.2f} {a[1]*100:5.1f}% {a[2]*100:5.1f}% | "
                  f"{a[3]:6.2f} {a[4]*100:5.1f}% {a[5]*100:5.1f}%")
        print(f"    ({el:.1f}s, {el/(45*44/2)*1000:.1f} ms/pair)")


if __name__ == "__main__":
    main()
