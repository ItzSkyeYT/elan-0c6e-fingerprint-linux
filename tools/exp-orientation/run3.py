#!/usr/bin/env python3
"""
JOINT search: score the LCN+NCC cue and the orientation-field cue at the SAME
alignment, maximise the weighted sum over shift and rotation.

    M_w[t,p] = max over (dx,dy,rot) of  (1-w) * NCC(dx,dy,rot)
                                      +  w    * ORIENT(dx,dy,rot)

The whole weight sweep comes out of one pass because the two correlation maps
are computed anyway.  w=0 reproduces the LCN+NCC baseline exactly, w=1 the pure
orientation matcher, so the sweep is self-calibrating.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                      # noqa
from oflib import _corr_planes, _rotate
from run import load_flat, show, CACHE    # noqa

WEIGHTS = np.round(np.arange(0.0, 1.001, 0.1), 3)


def matrices_joint(imgs, block=3, energy_q=0.40, rel_pow=1.0, rel_floor=0.05,
                   max_dx=20, max_dy=8, max_rot=12, rot_step=4,
                   min_blocks=25, min_overlap=3500, lcn_sigma=6.0,
                   weights=WEIGHTS):
    n = len(imgs)
    ph, pw, iy, ix = _corr_planes(max_dx, max_dy)
    assert (ph, pw) == (H + 2 * max_dy + 1, W + 2 * max_dx + 1)
    pres = [local_contrast_norm(im, lcn_sigma) for im in imgs]
    kw = dict(block=block, energy_q=energy_q, rel_pow=rel_pow,
              rel_floor=rel_floor)

    FA_o = [fft_pack(make_field(None, pre=p, deg=0, **kw), ph, pw) for p in pres]
    FA_n = [np.fft.rfft2(p, s=(ph, pw)) for p in pres]
    rots = [0] if max_rot == 0 else list(range(-max_rot, max_rot + 1, rot_step))

    M = np.full((len(weights), n, n), -1.0)
    for j in range(n):
        fields = [make_field(None, pre=pres[j], deg=d, **kw) for d in rots]
        FB_o = [fft_pack(f, ph, pw, conj=True) for f in fields]
        brs = [_rotate(pres[j].astype(np.float32), d) for d in rots]
        FB_n = [np.conj(np.fft.rfft2(b, s=(ph, pw))) for b in brs]
        for i in range(n):
            best = np.full(len(weights), -1.0)
            for k in range(len(rots)):
                om = orient_map(FA_o[i], FB_o[k], ph, pw, iy, ix,
                                min_blocks, block)
                nm = ncc_map(pres[i], brs[k], max_dx, max_dy, min_overlap,
                             fa=FA_n[i], fb=FB_n[k])
                good = np.isfinite(om) & np.isfinite(nm)
                if not good.any():
                    continue
                o = om[good]
                c = nm[good]
                for wi, w in enumerate(weights):
                    v = float(np.max((1 - w) * c + w * o))
                    if v > best[wi]:
                        best[wi] = v
            M[:, i, j] = best
    return M


def main():
    imgs, labels = load_flat()
    tag = sys.argv[1] if len(sys.argv) > 1 else "joint_b3"
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.exists(path):
        M = np.load(path)
    else:
        t0 = time.time()
        M = matrices_joint(imgs, block=block)
        np.save(path, M)
        print(f"computed in {time.time()-t0:.0f}s", file=sys.stderr)

    res = []
    for wi, w in enumerate(WEIGHTS):
        res.append(show(f"{tag} w={w:.1f}", M[wi], labels, quiet=True))
    print(f"\n== JOINT search, block {block}: (1-w)*NCC + w*orientation, "
          f"maximised over a COMMON alignment ==")
    print(f"  {'w':>5}{'A d-prime':>12}{'A EER':>9}{'A FAR@10':>10}"
          f"{'B d-prime':>12}{'B EER':>8}{'B FAR@10':>10}")
    for r, w in zip(res, WEIGHTS):
        print(f"  {w:5.1f}{r['A'][0][0]:12.2f}{r['A'][0][1]*100:8.1f}%"
              f"{r['A'][0][2]*100:9.1f}%{r['B'][0]:12.2f}{r['B'][1]*100:7.1f}%"
              f"{r['B'][2]*100:9.1f}%")
    bi = int(np.argmax([r["B"][0] for r in res]))
    show(f"{tag} best w={WEIGHTS[bi]:.1f}", M[bi], labels)


if __name__ == "__main__":
    main()
