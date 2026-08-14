#!/usr/bin/env python3
"""ORACLE upper bound: for each image pair, search every translation (and small
rotation) and take the alignment that MAXIMISES the number of coincident
minutiae. This bounds what any translation/rotation-based minutia matcher --
bozorth3 included -- could ever achieve on this data. It is applied identically
to genuine and impostor pairs, so it is not a genuine-only advantage.
"""
import subprocess, sys, math
from pathlib import Path
import numpy as np
import evalproto as ep

HERE = Path(__file__).resolve().parent


def load_xyt(d, n, scale, h_scaled):
    out = []
    for i in range(n):
        p = Path(d) / f"{i}.xyt"
        rows = []
        if p.exists():
            for line in p.read_text().split("\n"):
                if not line.strip():
                    continue
                x, y, t, q = (int(v) for v in line.split())
                rows.append((x / scale, (h_scaled - y) / scale, t, q))
        out.append(np.array(rows, float).reshape(-1, 4))
    return out


def oracle_shared(a, b, tol=8.0, dtol=45.0, max_dx=60, max_dy=25, rots=(0,)):
    if len(a) == 0 or len(b) == 0:
        return 0
    best = 0
    ax, ay, at = a[:, 0], a[:, 1], a[:, 2]
    for rot in rots:
        r = math.radians(rot)
        cx, cy = 75.0, 26.0
        bx = (b[:, 0] - cx) * math.cos(r) - (b[:, 1] - cy) * math.sin(r) + cx
        by = (b[:, 0] - cx) * math.sin(r) + (b[:, 1] - cy) * math.cos(r) + cy
        bt = (b[:, 2] + rot) % 360
        # candidate shifts: every (a_i - b_j) difference vector is a candidate
        for i in range(len(a)):
            for j in range(len(b)):
                dx, dy = ax[i] - bx[j], ay[i] - by[j]
                if abs(dx) > max_dx or abs(dy) > max_dy:
                    continue
                px, py = bx + dx, by + dy
                d2 = (px[None, :] - ax[:, None]) ** 2 + (py[None, :] - ay[:, None]) ** 2
                dt = np.abs(((at[:, None] - bt[None, :] + 180) % 360) - 180)
                ok = (d2 <= tol * tol) & (dt <= dtol)
                # greedy one-to-one
                n = 0
                usedb = set()
                for p in range(len(a)):
                    cand = np.where(ok[p])[0]
                    for q in cand:
                        if q not in usedb:
                            usedb.add(int(q)); n += 1
                            break
                if n > best:
                    best = n
    return best


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "prep/s2bilin/gabor_eq"
    scale = 2 if "/s2" in variant else 1
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    rots = (0,) if len(sys.argv) <= 3 else tuple(range(-12, 13, 4))
    root = HERE / variant
    man = ep.manifest(root)
    mf = HERE / "manifest.txt"
    mf.write_text("\n".join(str(p) for _, p in man) + "\n")
    xd = HERE / "xyt"; xd.mkdir(exist_ok=True)
    subprocess.run([str(HERE / "nbisbatch"), str(mf), "19.685", "0", str(xd)],
                   capture_output=True, check=True)
    M = load_xyt(xd, len(man), scale, 52 * scale)
    idx = ep.index_sets(man)
    gpool = idx["right-index"] + idx["right-index-cover"]
    ipool = idx["right-middle"]

    gp = [(gpool[i], gpool[j]) for i in range(len(gpool)) for j in range(i + 1, len(gpool))]
    ip = [(a, b) for a in gpool for b in ipool]
    gs = np.array([oracle_shared(M[i], M[j], tol, rots=rots) for i, j in gp], float)
    isx = np.array([oracle_shared(M[i], M[j], tol, rots=rots) for i, j in ip], float)
    print(f"variant={variant} tol={tol}px rots={rots}")
    print("  minutiae/image mean %.1f" % np.mean([len(m) for m in M]))
    print("  ORACLE shared minutiae  genuine  n=%d mean %.2f max %d  hist %s"
          % (len(gs), gs.mean(), gs.max(), np.bincount(gs.astype(int), minlength=10)[:10]))
    print("  ORACLE shared minutiae  impostor n=%d mean %.2f max %d  hist %s"
          % (len(isx), isx.mean(), isx.max(), np.bincount(isx.astype(int), minlength=10)[:10]))
    print("  ORACLE pairwise d' = %.3f   (this is an UPPER BOUND, not achievable)"
          % ep.dprime(gs, isx))


if __name__ == "__main__":
    main()
