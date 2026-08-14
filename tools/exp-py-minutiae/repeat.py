#!/usr/bin/env python3
"""How repeatable is the minutiae extraction?

Uses the LCN+rotation NCC matcher (the d'=1.54/1.94 baseline) to find the
best rigid alignment for each genuine pair, then asks how many of the
minutiae in the overlapping region actually correspond.  If that number is
tiny, minutiae matching cannot work here no matter how good the matcher is.
"""
import math
import sys
import numpy as np

sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
exec(open('/home/melb/projects/elan-0c6e-linux/tools/matcher-lab.py').read()
     .split('def main(')[0])
import minutiae as M
from common import load_ds

W, H = 150, 52


def best_align(a, b, max_dx=30, max_dy=12, max_rot=15, rot_step=3, min_overlap=2500):
    """Return (deg, dx, dy, ncc) of the best LCN alignment: b ~ shift(rot(a))."""
    la, lb = local_contrast_norm(a), local_contrast_norm(b)
    best = (-2.0, 0, 0, 0)
    for deg in range(-max_rot, max_rot + 1, rot_step):
        ar = _rotate(la, deg) if deg else la
        for dy in range(-max_dy, max_dy + 1, 2):
            for dx in range(-max_dx, max_dx + 1, 2):
                # ncc between ar shifted by (dx,dy) and lb
                x0, x1 = max(0, dx), min(W, W + dx)
                y0, y1 = max(0, dy), min(H, H + dy)
                if (x1 - x0) * (y1 - y0) < min_overlap:
                    continue
                pb = lb[y0:y1, x0:x1]
                pa = ar[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
                pa = pa - pa.mean(); pb = pb - pb.mean()
                den = math.sqrt(float((pa*pa).sum()) * float((pb*pb).sum()))
                c = float((pa*pb).sum()/den) if den > 1e-9 else -1
                if c > best[0]:
                    best = (c, deg, dx, dy)
    return best


def main():
    ds = load_ds()
    gen = ds["right-index"] + ds["right-index-cover"]
    imp = ds["right-middle"]
    cfg = {}
    T = {}

    def tm(img, key):
        if key not in T:
            T[key] = M.make_template(img, cfg)
        return T[key]

    def corr(ta, tb, deg, dx, dy, tol=10.0):
        """A is rotated about the image centre by deg then shifted by (dx,dy)."""
        t = math.radians(deg)
        ct, st = math.cos(t), math.sin(t)
        cx, cy = (W-1)/2, (H-1)/2
        A = ta.m
        if len(A) == 0 or len(tb.m) == 0:
            return 0, 0, 0
        # _rotate maps output(x,y) <- input(xr,yr) with xr = ct*(x-cx)+st*(y-cy)+cx
        # so a feature at input (u,v) appears at output (x,y) with
        # x = ct*(u-cx) - st*(v-cy) + cx
        ax = ct*(A[:,0]-cx) - st*(A[:,1]-cy) + cx + dx
        ay = st*(A[:,0]-cx) + ct*(A[:,1]-cy) + cy + dy
        B = tb.m
        # overlap region in B's frame
        x0, x1 = max(0, dx), min(W, W+dx)
        y0, y1 = max(0, dy), min(H, H+dy)
        inA = (ax >= x0+3) & (ax < x1-3) & (ay >= y0+3) & (ay < y1-3)
        inB = (B[:,0] >= x0+3) & (B[:,0] < x1-3) & (B[:,1] >= y0+3) & (B[:,1] < y1-3)
        nA, nB = int(inA.sum()), int(inB.sum())
        if nA == 0 or nB == 0:
            return 0, nA, nB
        d = np.hypot(ax[inA][:,None]-B[inB,0][None,:], ay[inA][:,None]-B[inB,1][None,:])
        used = np.zeros(nB, bool); m = 0
        for i in range(nA):
            j = int(np.argmin(np.where(used, 1e9, d[i])))
            if not used[j] and d[i, j] <= tol:
                used[j] = True; m += 1
        return m, nA, nB

    print("GENUINE pairs (aligned by the LCN-NCC baseline):")
    rows = []
    pairs = [(i, j) for i in range(len(gen)) for j in range(i+1, len(gen))]
    rng = np.random.default_rng(1)
    sel = rng.choice(len(pairs), size=60, replace=False)
    for k in sel:
        i, j = pairs[k]
        a, b = gen[i][1], gen[j][1]
        c, deg, dx, dy = best_align(a, b)
        ta, tb = tm(a, ('g', i)), tm(b, ('g', j))
        m, nA, nB = corr(ta, tb, deg, dx, dy)
        rows.append((c, m, nA, nB))
    r = np.array(rows)
    print(f"  n={len(r)}  ncc {r[:,0].mean():.3f}   matched {r[:,1].mean():.2f}"
          f"  (median {np.median(r[:,1]):.0f}, max {r[:,1].max():.0f})"
          f"   overlap minutiae nA {r[:,2].mean():.1f} nB {r[:,3].mean():.1f}")
    print(f"  pairs with >=3 corresponding minutiae: {(r[:,1]>=3).mean()*100:.0f}%")
    print(f"  pairs with >=5 corresponding minutiae: {(r[:,1]>=5).mean()*100:.0f}%")

    print("IMPOSTOR pairs (same alignment procedure):")
    rows = []
    ipairs = [(i, j) for i in range(len(gen)) for j in range(len(imp))]
    sel = rng.choice(len(ipairs), size=60, replace=False)
    for k in sel:
        i, j = ipairs[k]
        a, b = gen[i][1], imp[j][1]
        c, deg, dx, dy = best_align(a, b)
        ta, tb = tm(a, ('g', i)), tm(b, ('i', j))
        m, nA, nB = corr(ta, tb, deg, dx, dy)
        rows.append((c, m, nA, nB))
    r2 = np.array(rows)
    print(f"  n={len(r2)}  ncc {r2[:,0].mean():.3f}   matched {r2[:,1].mean():.2f}"
          f"  (median {np.median(r2[:,1]):.0f}, max {r2[:,1].max():.0f})"
          f"   overlap minutiae nA {r2[:,2].mean():.1f} nB {r2[:,3].mean():.1f}")
    print(f"  d' of matched-count: {(r[:,1].mean()-r2[:,1].mean())/math.sqrt((r[:,1].var()+r2[:,1].var())/2):.2f}")


if __name__ == "__main__":
    main()
