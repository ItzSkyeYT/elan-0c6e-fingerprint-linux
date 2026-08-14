#!/usr/bin/env python3
"""
Build full 45x45 pair-score matrices for the orientation-field matcher and for
the LCN+NCC baseline, then evaluate both under the two mandated protocols.

Matrices are cached in .npz so the (cheap) evaluation and the fusion sweep can
be re-run without recomputing scores.

    M[t, p] = score(template image t, probe image p)

The matrix is deliberately asymmetric: only the probe is rotated, exactly as a
driver would do it.
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                      # noqa
from oflib import _corr_planes, _rotate

DS = os.path.expanduser("~/.local/share/elan-fp/dataset")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


def load_flat():
    ds = load_dataset(DS)
    imgs, labels = [], []
    for lbl in ("right-index", "right-index-cover", "right-middle"):
        for _, im in ds[lbl]:
            imgs.append(im)
            labels.append(lbl)
    labels = np.array(labels)
    return imgs, labels


# ------------------------------------------------------------- matrices --

def matrix_orient(imgs, block=8, energy_q=0.40, rel_pow=1.0, rel_floor=0.05,
                  max_dx=24, max_dy=10, max_rot=12, rot_step=3, min_blocks=25,
                  lcn_sigma=6.0, centred=True):
    scorer = orient_score_centred if centred else orient_score
    n = len(imgs)
    ph, pw, iy, ix = _corr_planes(max_dx, max_dy)
    pres = [local_contrast_norm(im, lcn_sigma) for im in imgs]
    kw = dict(block=block, energy_q=energy_q, rel_pow=rel_pow,
              rel_floor=rel_floor)
    FA = [fft_pack(make_field(None, pre=p, deg=0, **kw), ph, pw) for p in pres]
    rots = [0] if max_rot == 0 else list(range(-max_rot, max_rot + 1, rot_step))
    M = np.full((n, n), -1.0)
    for j in range(n):                     # j = probe (rotated)
        FBs = [fft_pack(make_field(None, pre=pres[j], deg=d, **kw), ph, pw,
                        conj=True) for d in rots]
        for i in range(n):
            best = -1.0
            for FB in FBs:
                s, _, _ = scorer(FA[i], FB, ph, pw, iy, ix, min_blocks, block)
                if s > best:
                    best = s
            M[i, j] = best
    return M


def matrix_ncc(imgs, max_dx=20, max_dy=8, min_overlap=3500, max_rot=12,
               rot_step=4, lcn_sigma=6.0):
    """The published baseline: local contrast normalisation + rotation + NCC
    (matcher-lab's m_ncc_lcn_rot), recomputed here under the same protocol so
    the comparison is apples-to-apples."""
    n = len(imgs)
    pres = [local_contrast_norm(im, lcn_sigma) for im in imgs]
    ph, pw = H + 2 * max_dy + 1, W + 2 * max_dx + 1
    FA = [np.fft.rfft2(p, s=(ph, pw)) for p in pres]
    rots = [0] if max_rot == 0 else list(range(-max_rot, max_rot + 1, rot_step))
    M = np.full((n, n), -1.0)
    for j in range(n):
        brs = [_rotate(pres[j].astype(np.float32), d) for d in rots]
        FBs = [np.conj(np.fft.rfft2(b, s=(ph, pw))) for b in brs]
        for i in range(n):
            best = -1.0
            for b, fb in zip(brs, FBs):
                c = ncc_all_shifts_vec(pres[i], b, max_dx, max_dy, min_overlap,
                                       fa=FA[i], fb=fb)
                if c > best:
                    best = c
            M[i, j] = best
    return M


def cached(name, fn, force=False):
    path = os.path.join(CACHE, name + ".npy")
    if os.path.exists(path) and not force:
        return np.load(path)
    t0 = time.time()
    M = fn()
    np.save(path, M)
    print(f"   [{name}] computed in {time.time()-t0:.1f}s", file=sys.stderr)
    return M


# ----------------------------------------------------------- reporting --

def show(name, M, labels, n_enroll=6, trials=16, quiet=False):
    idx_index = np.where(labels == "right-index")[0]
    idx_cover = np.where(labels == "right-index-cover")[0]
    idx_imp = np.where(labels == "right-middle")[0]
    gen_pool = np.concatenate([idx_index, idx_cover])

    mA, sA = eval_pooled(M, gen_pool, idx_imp, n_enroll=n_enroll, trials=trials)
    (dB, eB, fB), gB, iB = eval_realistic(M, idx_cover, idx_index, idx_imp)
    if not quiet:
        print(f"\n--- {name} ---")
        print(f"  A pooled ({n_enroll} templates, {trials} random subsets)"
              f"   d'={mA[0]:5.2f} +-{sA[0]:.2f}   EER={mA[1]*100:5.1f}% +-{sA[1]*100:.1f}"
              f"   FAR@10%FRR={mA[2]*100:5.1f}% +-{sA[2]*100:.1f}")
        print(f"  B realistic (19 cover templates, 12 probes)          "
              f"   d'={dB:5.2f}          EER={eB*100:5.1f}%"
              f"        FAR@10%FRR={fB*100:5.1f}%")
        print(f"      genuine {np.mean(gB):.3f}+-{np.std(gB):.3f}   "
              f"impostor {np.mean(iB):.3f}+-{np.std(iB):.3f}   "
              f"(self-score check: diag mean {np.diag(M).mean():.3f})")
    return dict(name=name, A=(mA, sA), B=(dB, eB, fB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-enroll", type=int, default=6)
    ap.add_argument("--trials", type=int, default=16)
    args = ap.parse_args()

    imgs, labels = load_flat()
    print(f"{len(imgs)} captures: "
          + ", ".join(f"{l}={int((labels==l).sum())}"
                      for l in ("right-index", "right-index-cover", "right-middle")))

    res = []
    Mncc = cached("ncc_lcn_rot", lambda: matrix_ncc(imgs), args.force)
    res.append(show("baseline LCN+rot+NCC", Mncc, labels, args.n_enroll, args.trials))

    for block in (4, 6, 8, 10):
        nm = f"orient_b{block}"
        M = cached(nm, (lambda b=block: matrix_orient(imgs, block=b,
                                                      centred=False)), args.force)
        res.append(show(f"orientation, weighted cos, block {block}", M, labels,
                        args.n_enroll, args.trials))
    for block in (3, 4, 6, 8, 10):
        nm = f"orientc_b{block}"
        M = cached(nm, (lambda b=block: matrix_orient(imgs, block=b,
                                                      centred=True)), args.force)
        res.append(show(f"orientation, CENTRED corr, block {block}", M, labels,
                        args.n_enroll, args.trials))

    print("\n" + "=" * 78)
    print(f"  {'matcher':<34}{'A d-prime':>10}{'A EER':>9}{'B d-prime':>11}{'B EER':>8}")
    for r in res:
        print(f"  {r['name']:<34}{r['A'][0][0]:10.2f}{r['A'][0][1]*100:8.1f}%"
              f"{r['B'][0]:11.2f}{r['B'][1]*100:7.1f}%")


if __name__ == "__main__":
    sys.exit(main())
