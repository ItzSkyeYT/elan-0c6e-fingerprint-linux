#!/usr/bin/env python3
"""Final measurement.

Produces, under the exact evaluation protocol:
  * pure minutiae matcher (Hough alignment, overlap-normalised score)
  * the LCN+rotation NCC baseline, recomputed in this harness
  * minutiae-guided NCC (minutiae only propose alignments; NCC scores them)
  * score-level fusion of minutiae + NCC, weight swept

so we can say whether minutiae contribute ANY information here.
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
import minutiae as M
from common import load_ds, dprime, eer, far_at_frr
from eval import build, scenario_A, scenario_B, report
from baseline import lcn_rot_matrix

W, H = 150, 52
ORDER = ["right-index", "right-index-cover", "right-middle"]

CFG = {"merge_dist": 0.0, "border": 3, "spur_len": 4, "min_ridge": 6}
MCFG = dict(rot_max=25.0, rot_bin=6.0, tr_bin=8.0, pos_tol=10.0,
            dir_tol=math.radians(25.0), top_k=6, min_denom=6.0, score="ov")


def minutiae_matrix(tmpl, mcfg):
    n = len(tmpl)
    S = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = M.match(tmpl[i], tmpl[j], **mcfg)
    el = time.time() - t0
    print(f"  minutiae matching {el:.1f}s ({el/(n*(n-1)/2)*1000:.2f} ms/pair)")
    np.fill_diagonal(S, np.nan)
    return S


# ------------------------------------------------ minutiae-guided NCC ------

def hypotheses(ta, tb, rot_max=25.0, rot_bin=6.0, tr_bin=8.0,
               dir_tol=math.radians(25.0), top_k=8):
    """Alignment hypotheses (deg, dx, dy) from the minutiae Hough vote."""
    A, B = ta.m, tb.m
    if len(A) == 0 or len(B) == 0:
        return []
    nrot = int(2 * rot_max / rot_bin) + 1
    rots = np.linspace(-rot_max, rot_max, nrot)
    votes = {}
    for ri, rdeg in enumerate(rots):
        t = math.radians(rdeg)
        ct, st = math.cos(t), math.sin(t)
        # rotate about the image centre so the hypothesis maps onto _rotate()
        cx, cy = (W - 1) / 2, (H - 1) / 2
        rax = ct * (A[:, 0] - cx) - st * (A[:, 1] - cy) + cx
        ray = st * (A[:, 0] - cx) + ct * (A[:, 1] - cy) + cy
        for i in range(len(A)):
            dd = (B[:, 2] - (A[i, 2] + t)) % math.pi
            dd = np.minimum(dd, math.pi - dd)
            ok = dd <= dir_tol
            if not ok.any():
                continue
            for X, Y in zip(B[ok, 0] - rax[i], B[ok, 1] - ray[i]):
                key = (ri, int(round(X / tr_bin)), int(round(Y / tr_bin)))
                votes[key] = votes.get(key, 0) + 1
    best = sorted(votes.items(), key=lambda kv: -kv[1])[:top_k]
    return [(rots[ri], ix * tr_bin, iy * tr_bin) for (ri, ix, iy), _ in best]


def ncc_at(la, lb_rot, dx, dy, min_overlap=1200):
    """NCC of la (rotated already) shifted by (dx,dy) against lb."""
    dx, dy = int(round(dx)), int(round(dy))
    x0, x1 = max(0, dx), min(W, W + dx)
    y0, y1 = max(0, dy), min(H, H + dy)
    if (x1 - x0) * (y1 - y0) < min_overlap:
        return -1.0
    pb = lb_rot[y0:y1, x0:x1]
    pa = la[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    pa = pa - pa.mean()
    pb = pb - pb.mean()
    den = math.sqrt(float((pa * pa).sum()) * float((pb * pb).sum()))
    return float((pa * pb).sum() / den) if den > 1e-9 else -1.0


def guided_matrix(ds, tmpl, refine=3):
    lcn = []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            lcn.append(M.local_contrast_norm(im))
    n = len(tmpl)
    S = np.zeros((n, n))
    t0 = time.time()
    rotcache = {}

    def rot(i, deg):
        k = (i, int(round(deg)))
        if k not in rotcache:
            rotcache[k] = _rotate_img(lcn[i], int(round(deg)))
        return rotcache[k]

    for i in range(n):
        for j in range(i + 1, n):
            hyps = hypotheses(tmpl[i], tmpl[j])
            best = -1.0
            for deg, dx, dy in hyps:
                ai = rot(i, deg)
                for sy in range(-refine, refine + 1, 2):
                    for sx in range(-refine, refine + 1, 2):
                        c = ncc_at(ai, lcn[j], dx + sx, dy + sy)
                        if c > best:
                            best = c
            S[i, j] = S[j, i] = best
    print(f"  guided matching {time.time()-t0:.1f}s")
    np.fill_diagonal(S, np.nan)
    return S


def _rotate_img(img, deg):
    if deg == 0:
        return img
    t = math.radians(deg)
    ct, st = math.cos(t), math.sin(t)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xr = ct * (xx - cx) + st * (yy - cy) + cx
    yr = -st * (xx - cx) + ct * (yy - cy) + cy
    x0 = np.clip(np.floor(xr).astype(int), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(yr).astype(int), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = np.clip(xr - x0, 0, 1)
    fy = np.clip(yr - y0, 0, 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy).astype(np.float32)


def evaluate(name, S, labels, n_enroll=6, trials=16):
    print(f"\n--- {name} ---")
    ga, ia, pt = scenario_A(S, labels, n_enroll, trials)
    ra = report(f"A pooled n={n_enroll}", ga, ia,
                f" per-trial d' {np.mean(pt):.2f}+-{np.std(pt):.2f}")
    gb, ib = scenario_B(S, labels)
    rb = report("B realistic", gb, ib)
    return ra, rb


def zs(S):
    v = S[np.isfinite(S)]
    return (S - v.mean()) / (v.std() + 1e-12)


def main():
    ds = load_ds()
    print("== extraction ==")
    names, labels, tmpl = build(ds, CFG, verbose=True)

    print("\n== pure minutiae ==")
    Sm = minutiae_matrix(tmpl, MCFG)
    np.save('S_minutiae.npy', Sm)
    # sanity: self-match must be maximal
    ss = [M.match(tmpl[i], tmpl[i], **MCFG) for i in range(0, len(tmpl), 4)]
    off = Sm[np.isfinite(Sm)]
    print(f"  SANITY self-match mean {np.mean(ss):.3f}, all-pairs mean {off.mean():.3f}, max {off.max():.3f}")
    evaluate("pure minutiae (Hough + overlap-normalised)", Sm, labels)

    print("\n== baseline ==")
    try:
        Sb = np.load('S_baseline.npy')
        print("  (loaded cached S_baseline.npy)")
    except Exception:
        Sb, _ = lcn_rot_matrix(ds)
        np.save('S_baseline.npy', Sb)
    evaluate("LCN + rotation + NCC (baseline)", Sb, labels)

    print("\n== minutiae-guided NCC ==")
    Sg = guided_matrix(ds, tmpl)
    np.save('S_guided.npy', Sg)
    evaluate("minutiae propose alignment, NCC scores it", Sg, labels)

    print("\n== fusion: does the minutiae score add anything to NCC? ==")
    zb, zm = zs(Sb), zs(Sm)
    for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5):
        Sf = zb + w * zm
        ga, ia, _ = scenario_A(Sf, labels, 6, 16)
        gb, ib = scenario_B(Sf, labels)
        print(f"  w={w:4.1f}   A d'={dprime(ga,ia):5.2f} EER={eer(ga,ia)[0]*100:4.1f}%"
              f"   B d'={dprime(gb,ib):5.2f} EER={eer(gb,ib)[0]*100:4.1f}%"
              f" FAR10={far_at_frr(gb,ib)[0]*100:5.1f}%")


if __name__ == "__main__":
    main()
