#!/usr/bin/env python3
"""
The transitivity result in s2 says the LCN+NCC 'best alignment' does not
compose, i.e. it is not finding a true registration. Before believing that,
validate the test itself; then try to recover a real registration by a
different, lower-frequency route (the block orientation field), because the
overlap question in the brief can only be answered from a real alignment.

 10  positive control: three shifted crops of ONE capture must close to ~0
 11  transitivity restricted to the HIGHEST-scoring genuine triples
 12  orientation-field registration, and its transitivity
 13  overlap distribution from whichever registration is actually consistent
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(TOOLS)
exec(open("matcher-lab.py").read().split("def main(")[0])       # noqa
from fastncc2 import ncc_best_align, overlap_stats             # noqa: E402
import quality as Q                                            # noqa: E402

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LABELS = ["right-index", "right-index-cover", "right-middle"]
BASE = dict(max_dx=20, max_dy=8, min_overlap=3500, max_rot=12, rot_step=4)


def closure(D, triples):
    e = []
    for i, j, k in triples:
        e.append(abs(D[0][i, j] + D[0][j, k] - D[0][i, k]) +
                 abs(D[1][i, j] + D[1][j, k] - D[1][i, k]))
    return np.array(e)


# --------------------------------------------- orientation-field alignment --

def orient_field(img, block=4):
    """Doubled-angle orientation vector per block, as a complex array.
    Magnitude carries how strongly oriented the block is."""
    lcn = Q.local_contrast_norm(img)
    gxx, gyy, gxy = Q.structure_tensor(lcn, block, presmooth=1.0)
    z = (gxx - gyy) + 1j * (2 * gxy)
    # damp the magnitude so a single very strong block cannot dominate
    m = np.abs(z)
    return z / (m + m.mean() + 1e-9) * np.sqrt(m / (m.mean() + 1e-9))


def of_best_align(za, zb, max_bx, max_by, min_blocks):
    """Normalised complex correlation of two orientation fields over integer
    block shifts. Returns (score, dbx, dby)."""
    h, w = za.shape
    best = (-2.0, 0, 0)
    for dy in range(-max_by, max_by + 1):
        ay0, ay1 = max(0, dy), min(h, h + dy)
        if ay1 - ay0 <= 0:
            continue
        for dx in range(-max_bx, max_bx + 1):
            ax0, ax1 = max(0, dx), min(w, w + dx)
            n = (ay1 - ay0) * (ax1 - ax0)
            if n < min_blocks:
                continue
            pa = za[ay0:ay1, ax0:ax1]
            pb = zb[ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx]
            num = float(np.real((pa * np.conj(pb)).sum()))
            den = float(np.sqrt((np.abs(pa) ** 2).sum() * (np.abs(pb) ** 2).sum()))
            if den > 1e-12:
                c = num / den
                if c > best[0]:
                    best = (c, dx, dy)
    return best


def main():
    ds = load_dataset(DATASET)                                 # noqa: F821
    names, labels, imgs = [], [], []
    for li, lbl in enumerate(LABELS):
        for n, im in ds[lbl]:
            names.append(n); labels.append(li); imgs.append(im)
    labels = np.array(labels)
    N = len(imgs)
    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]

    # ================================================== 10. positive control
    print("=" * 78)
    print("10. POSITIVE CONTROL FOR THE TRANSITIVITY TEST")
    print("=" * 78)
    print("   Three synthetic 'captures' made by shifting ONE real capture by")
    print("   known amounts. If the test is sound, closure error must be ~0.")
    src = imgs[5]
    shifts = [(0, 0), (2, -6), (-3, 9), (1, 14), (-2, -11)]
    syn = [np.roll(src, (sy, sx), axis=(0, 1)) for sy, sx in shifts]
    L = [local_contrast_norm(s) for s in syn]                  # noqa: F821
    n = len(syn)
    DX = np.zeros((n, n), int); DY = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            s, dx, dy, deg = ncc_best_align(L[i], L[j], 20, 8, 3500, 0, 4,
                                            rotate=_rotate)     # noqa: F821
            DX[i, j], DY[i, j] = dx, dy
            DX[j, i], DY[j, i] = -dx, -dy
    trip = list(itertools.combinations(range(n), 3))
    e = closure((DX, DY), trip)
    print(f"   synthetic triples n={len(trip)}: |closure| med {np.median(e):.1f} px, "
          f"max {e.max():.1f} px, exact: {(e == 0).mean()*100:.0f}%")
    print(f"   -> the test is sound; a real registration composes exactly."
          if np.median(e) <= 1 else
          f"   -> WARNING: the test itself is broken, ignore section 6.")

    # =============================== 11. transitivity on high-scoring triples
    print("\n" + "=" * 78)
    print("11. TRANSITIVITY ON THE STRONGEST GENUINE TRIPLES")
    print("=" * 78)
    z = np.load(HERE / "scores_base.npz")
    S, dX, dY = z["S"], z["DX"], z["DY"]
    trip = list(itertools.combinations(gen.tolist(), 3))
    tscore = np.array([min(S[i, j], S[j, k], S[i, k]) for i, j, k in trip])
    err = closure((dX, dY), trip)
    rng = np.random.default_rng(3)
    print(f"   {'triple selection':<38} {'n':>6} {'med |closure|':>14} {'<=4px':>8}")
    for lab, sel in (("all genuine triples", np.ones(len(trip), bool)),
                     ("min pair score >= 0.20", tscore >= 0.20),
                     ("min pair score >= 0.25", tscore >= 0.25),
                     ("min pair score >= 0.30", tscore >= 0.30)):
        if sel.sum() < 5:
            continue
        e = err[sel]
        print(f"   {lab:<38} {sel.sum():6d} {np.median(e):14.1f} "
              f"{(e <= 4).mean()*100:7.1f}%")
    # null from shuffling shifts among genuine pairs
    iu = np.triu_indices(len(gen), 1)
    ax = dX[np.ix_(gen, gen)][iu]; ay = dY[np.ix_(gen, gen)][iu]
    ne = []
    for _ in range(20000):
        a, b, d = rng.integers(0, len(ax), 3)
        ne.append(abs(ax[a] + ax[b] - ax[d]) + abs(ay[a] + ay[b] - ay[d]))
    ne = np.array(ne)
    print(f"   {'shuffled null':<38} {len(ne):6d} {np.median(ne):14.1f} "
          f"{(ne <= 4).mean()*100:7.1f}%")

    # ======================================= 12. orientation-field alignment
    print("\n" + "=" * 78)
    print("12. REGISTRATION FROM THE BLOCK ORIENTATION FIELD")
    print("=" * 78)
    print("   Ridge phase aliases (ridges repeat every ~8 px, so a wrong")
    print("   alignment can correlate as well as the right one). The")
    print("   orientation field varies far more slowly, so if ANY reliable")
    print("   registration exists it should show up here.")
    BLK = 4
    ZF = [orient_field(im, BLK) for im in imgs]
    hb, wb = ZF[0].shape
    print(f"   field {hb}x{wb} blocks of {BLK}px, search +-15 x +-4 blocks, "
          f"min 120 blocks overlap")
    OX = np.zeros((N, N), int); OY = np.zeros((N, N), int)
    OS = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(i + 1, N):
            s, dx, dy = of_best_align(ZF[i], ZF[j], 15, 4, 120)
            OX[i, j], OY[i, j], OS[i, j] = dx, dy, s
            OX[j, i], OY[j, i], OS[j, i] = -dx, -dy, s
    for what, pool in (("genuine ", gen), ("impostor", imp)):
        tr = list(itertools.combinations(pool.tolist(), 3))
        if len(tr) > 4000:
            tr = [tr[t] for t in rng.choice(len(tr), 4000, replace=False)]
        e = closure((OX, OY), tr)
        print(f"   {what} triples n={len(tr):5d}: med |closure| {np.median(e):5.1f} "
              f"blocks ({np.median(e)*BLK:4.0f} px), exact(0): {(e==0).mean()*100:4.1f}%,"
              f" <=1 block: {(e<=1).mean()*100:4.1f}%")
    ne = []
    iu = np.triu_indices(len(gen), 1)
    ax = OX[np.ix_(gen, gen)][iu]; ay = OY[np.ix_(gen, gen)][iu]
    for _ in range(20000):
        a, b, d = rng.integers(0, len(ax), 3)
        ne.append(abs(ax[a] + ax[b] - ax[d]) + abs(ay[a] + ay[b] - ay[d]))
    ne = np.array(ne)
    print(f"   shuffled null       : med |closure| {np.median(ne):5.1f} blocks, "
          f"exact(0): {(ne==0).mean()*100:4.1f}%, <=1 block: {(ne<=1).mean()*100:4.1f}%")

    # does the orientation-field SCORE itself separate genuine from impostor?
    gp = list(itertools.combinations(gen.tolist(), 2))
    ip = [(a, b) for a in gen.tolist() for b in imp.tolist()]
    og = np.array([OS[a, b] for a, b in gp]); oi = np.array([OS[a, b] for a, b in ip])
    den = np.sqrt((og.var() + oi.var()) / 2)
    print(f"\n   as a matcher in its own right, pairwise: genuine {og.mean():.3f}"
          f"+-{og.std():.3f}  impostor {oi.mean():.3f}+-{oi.std():.3f}  "
          f"d'={(og.mean()-oi.mean())/den:.2f}")

    # ================================================= 13. overlap, honestly
    print("\n" + "=" * 78)
    print("13. OVERLAP, FROM THE ALIGNMENT THAT ACTUALLY COMPOSES")
    print("=" * 78)
    masks = [Q.foreground_mask_px(im) for im in imgs]
    fg = np.array([m.sum() for m in masks])
    print("   Global least-squares placement: solve for a per-image offset t_i")
    print("   minimising sum w_ij ||(t_i - t_j) - d_ij||^2 over genuine pairs.")
    for tag, DXm, DYm, Wm, scale in (
            ("LCN+NCC pairwise shifts", dX, dY, np.maximum(S, 0) ** 2, 1),
            ("orientation-field shifts", OX, OY, np.maximum(OS, 0) ** 2, BLK)):
        idx = gen
        n = len(idx)
        A = np.zeros((n, n)); bx = np.zeros(n); by = np.zeros(n)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                w = Wm[idx[a], idx[b]]
                if not np.isfinite(w):
                    continue
                A[a, a] += w; A[a, b] -= w
                bx[a] += w * DXm[idx[a], idx[b]]
                by[a] += w * DYm[idx[a], idx[b]]
        A += np.eye(n) * 1e-6
        tx = np.linalg.lstsq(A, bx, rcond=None)[0]
        ty = np.linalg.lstsq(A, by, rcond=None)[0]
        res = []
        for a in range(n):
            for b in range(a + 1, n):
                res.append(abs((tx[a] - tx[b]) - DXm[idx[a], idx[b]]) +
                           abs((ty[a] - ty[b]) - DYm[idx[a], idx[b]]))
        res = np.array(res) * scale
        print(f"   [{tag:<26}] residual med {np.median(res):5.1f} px, "
              f"p90 {np.percentile(res,90):5.1f} px "
              f"(fitted spread of placements: x sd {tx.std()*scale:.1f} px, "
              f"y sd {ty.std()*scale:.1f} px)")

    print("\n   Direct overlap using the orientation-field alignment:")
    for what, pairs in (("genuine ", gp), ("impostor", ip)):
        frac, geo = [], []
        for a, b in pairs:
            g, both, _, _ = overlap_stats(masks[a], masks[b],
                                          OX[a, b] * BLK, OY[a, b] * BLK)
            frac.append(both / max(1, min(fg[a], fg[b])))
            geo.append(g / (150 * 52))
        frac = np.array(frac); geo = np.array(geo)
        p = np.percentile(frac, [5, 25, 50, 75, 95])
        print(f"     {what} n={len(pairs):4d}  shared finger / smaller finger: "
              f"p5={p[0]:.2f} p25={p[1]:.2f} MED={p[2]:.2f} p75={p[3]:.2f} p95={p[4]:.2f}")
        print(f"              geometric frame overlap median {np.median(geo)*100:.0f}% "
              f"of the 150x52 window")


if __name__ == "__main__":
    main()
