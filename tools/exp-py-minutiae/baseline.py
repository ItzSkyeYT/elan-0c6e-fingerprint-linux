#!/usr/bin/env python3
"""Reproduce the LCN+rotation NCC baseline (reported d'=1.54 scenario A,
d'=1.94 scenario B) inside THIS harness, so the minutiae numbers are
comparable and the harness itself is validated.
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from fastncc import ncc_all_shifts
exec(open('/home/melb/projects/elan-0c6e-linux/tools/matcher-lab.py').read()
     .split('def main(')[0])

from common import load_ds, dprime, eer, far_at_frr
from eval import scenario_A, scenario_B, report

ORDER = ["right-index", "right-index-cover", "right-middle"]


def lcn_rot_matrix(ds, max_rot=12, rot_step=4, max_dx=20, max_dy=8,
                   min_overlap=3500):
    imgs, labels = [], []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            imgs.append(local_contrast_norm(im))
            labels.append(lbl)
    n = len(imgs)
    rots = list(range(-max_rot, max_rot + 1, rot_step))
    prerot = [[_rotate(im, d) if d else im for d in rots] for im in imgs]
    S = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            best = -1.0
            for k in range(len(rots)):
                c = ncc_all_shifts(imgs[i], prerot[j][k], max_dx, max_dy, min_overlap)
                if c > best:
                    best = c
            S[i, j] = S[j, i] = best
    print(f"  baseline matching {time.time()-t0:.1f}s")
    np.fill_diagonal(S, np.nan)
    return S, labels


if __name__ == "__main__":
    ds = load_ds()
    S, labels = lcn_rot_matrix(ds)
    np.save('/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae/S_baseline.npy', S)
    print("\nBASELINE  local contrast norm + rotation + NCC")
    for ne in (4, 6, 8, 10):
        g, i, pt = scenario_A(S, labels, n_enroll=ne, trials=12)
        report(f"A pooled n={ne}", g, i, f" per-trial d' {np.mean(pt):.2f}")
    g, i = scenario_B(S, labels)
    report("B realistic", g, i)
