#!/usr/bin/env python3
"""Diagnostic: are MINDTCT minutiae reproducible across two presses of the same
finger, once the pair is aligned by the (known-good) LCN+NCC translation?

If the same physical minutia is not detected in both captures, no minutia
matcher -- bozorth3 or otherwise -- can possibly work here.
"""
import subprocess, sys, os, math
from pathlib import Path
import numpy as np
import evalproto as ep

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
exec(open(TOOLS / "matcher-lab.py").read().split("def main(")[0])
sys.path.insert(0, str(TOOLS))



def load_xyt(d, n, scale, h_scaled):
    """MINDTCT xyt is in NIST convention: y measured from the BOTTOM.
    Return list of arrays [x, y_top, theta_deg] in ORIGINAL (unscaled) pixels."""
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


def _integral(x):
    return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _rect(ii, y0, y1, x0, x1):
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def best_shift(a, b, max_dx=40, max_dy=16, min_overlap=1800):
    """Best translation aligning b onto a using LCN+NCC.
    Convention (same as fastncc): a[y, x] pairs with b[y-dy, x-dx], so a
    minutia at (bx, by) in b maps to (bx+dx, by+dy) in a's frame."""
    A = local_contrast_norm(a).astype(np.float64)
    B = local_contrast_norm(b).astype(np.float64)
    h, w = A.shape
    ph, pw = h + max_dy * 2 + 1, w + max_dx * 2 + 1
    corr = np.fft.irfft2(np.fft.rfft2(A, s=(ph, pw)) * np.conj(np.fft.rfft2(B, s=(ph, pw))),
                         s=(ph, pw))
    ia, ia2 = _integral(A), _integral(A * A)
    ib, ib2 = _integral(B), _integral(B * B)
    best = (-2.0, 0, 0)
    for dy in range(-max_dy, max_dy + 1):
        y0, y1 = max(0, dy), min(h, h + dy)
        if y1 <= y0:
            continue
        for dx in range(-max_dx, max_dx + 1):
            x0, x1 = max(0, dx), min(w, w + dx)
            if x1 <= x0:
                continue
            n = (y1 - y0) * (x1 - x0)
            if n < min_overlap:
                continue
            sa = _rect(ia, y0, y1, x0, x1); sa2 = _rect(ia2, y0, y1, x0, x1)
            sb = _rect(ib, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
            sb2 = _rect(ib2, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
            num = corr[dy % ph, dx % pw] - sa * sb / n
            den = math.sqrt(max(sa2 - sa * sa / n, 0.0) * max(sb2 - sb * sb / n, 0.0))
            if den > 1e-9 and num / den > best[0]:
                best = (float(num / den), dx, dy)
    return best


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "prep/s2bilin/gabor_eq"
    scale = 2 if "/s2" in variant else 1
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    root = HERE / variant
    man = ep.manifest(root)
    mf = HERE / "manifest.txt"
    mf.write_text("\n".join(str(p) for _, p in man) + "\n")
    xd = HERE / "xyt"
    xd.mkdir(exist_ok=True)
    subprocess.run([str(HERE / "nbisbatch"), str(mf), "19.685", "0", str(xd)],
                   capture_output=True, check=True)
    h_scaled = 52 * scale
    M = load_xyt(xd, len(man), scale, h_scaled)

    orig = ep.manifest()          # raw dataset, same order
    imgs = [load_pgm(p) for _, p in orig]
    idx = ep.index_sets(orig)
    gpool = idx["right-index"] + idx["right-index-cover"]
    ipool = idx["right-middle"]

    print(f"variant={variant} scale={scale} tol={tol}px")
    print("minutiae per image: mean %.1f" % np.mean([len(m) for m in M]))

    def shared(pairs, name, limit=None):
        res = []
        for k, (i, j) in enumerate(pairs):
            if limit and k >= limit:
                break
            v, dx, dy = best_shift(imgs[i], imgs[j])
            a, b = M[i], M[j]
            if len(a) == 0 or len(b) == 0:
                res.append((0, v, len(a), len(b)))
                continue
            # b shifted by (dx,dy) lands on a's frame
            bx = b[:, 0] + dx
            by = b[:, 1] + dy
            n = 0
            used = set()
            for p in range(len(a)):
                d2 = (bx - a[p, 0]) ** 2 + (by - a[p, 1]) ** 2
                order = np.argsort(d2)
                for q in order:
                    if d2[q] > tol * tol:
                        break
                    if q in used:
                        continue
                    dt = abs(((a[p, 2] - b[q, 2] + 180) % 360) - 180)
                    if dt < 40:
                        used.add(int(q)); n += 1
                        break
            res.append((n, v, len(a), len(b)))
        return res

    gpairs = [(gpool[i], gpool[j]) for i in range(len(gpool)) for j in range(i + 1, len(gpool))]
    ipairs = [(a, b) for a in gpool for b in ipool]
    rng = np.random.default_rng(0)
    gs = shared([gpairs[k] for k in rng.choice(len(gpairs), 120, replace=False)], "gen")
    isx = shared([ipairs[k] for k in rng.choice(len(ipairs), 120, replace=False)], "imp")
    gs = np.array(gs); isx = np.array(isx)
    print("genuine  pairs: shared minutiae mean %.2f max %d  (ncc mean %.3f)"
          % (gs[:, 0].mean(), gs[:, 0].max(), gs[:, 1].mean()))
    print("impostor pairs: shared minutiae mean %.2f max %d  (ncc mean %.3f)"
          % (isx[:, 0].mean(), isx[:, 0].max(), isx[:, 1].mean()))
    print("genuine shared-count histogram :", np.bincount(gs[:, 0].astype(int), minlength=8)[:8])
    print("impostor shared-count histogram:", np.bincount(isx[:, 0].astype(int), minlength=8)[:8])
    # restricted to well-overlapping genuine pairs (top NCC quartile)
    top = gs[gs[:, 1] >= np.quantile(gs[:, 1], 0.75)]
    print("genuine pairs with BEST alignment (top NCC quartile): shared mean %.2f max %d"
          % (top[:, 0].mean(), top[:, 0].max()))


if __name__ == "__main__":
    main()
