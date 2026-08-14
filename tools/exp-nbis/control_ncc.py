#!/usr/bin/env python3
"""CONTROL: run the known LCN+rotation+NCC baseline through *this* evaluator.
If evalproto.py is sound it must reproduce roughly d'=1.5 (A) / 1.9 (B).
Guards against the NBIS null result being an artefact of my harness."""
import sys
from pathlib import Path
import numpy as np
import evalproto as ep

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
exec(open(TOOLS / "matcher-lab.py").read().split("def main(")[0])

man = ep.manifest()
imgs = [local_contrast_norm(load_pgm(p)) for _, p in man]
rots = [_rotate(im, d) for im in imgs for d in ()]  # placeholder
n = len(imgs)
S = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            S[i, j] = 1.0
            continue
        if j < i:
            S[i, j] = S[j, i]
            continue
        best = -1.0
        for deg in (-6, -3, 0, 3, 6):
            b = _rotate(imgs[j], deg) if deg else imgs[j]
            v = ncc_best(imgs[i], b, max_dx=20, max_dy=8, min_overlap=3500)
            best = max(best, v)
        S[i, j] = best
    print(".", end="", flush=True)
print()
np.save(HERE / "S_ncc_lcn_rot.npy", S)
ep.summarise("CONTROL lcn+rot+ncc", S, man)
