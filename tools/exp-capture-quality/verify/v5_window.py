#!/usr/bin/env python3
"""
10. Is the ALIGNMENT SEARCH WINDOW the binding constraint?

35% of genuine pairs align to the edge of the baseline +-20/+-8 window, which
means the true displacement exceeds the search. This measures whether widening
the search (and by how much) changes anything, using the same protocols.

min_overlap is scaled with the window: a wide search with a permissive overlap
floor lets a 20%-overlap coincidence win, which inflates IMPOSTOR scores too.
Both a permissive and a strict floor are reported for each window.
"""
import os, sys, math, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent; TOOLS = EXP.parent
sys.path.insert(0, str(EXP)); os.chdir(TOOLS)
exec(open("matcher-lab.py").read().split("def main(")[0])          # noqa

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LBL = ["right-index", "right-index-cover", "right-middle"]
ROTS = list(range(-12, 13, 4))

ds = load_dataset(DATASET)                                          # noqa
labels, imgs = [], []
for li, l in enumerate(LBL):
    for _n, im in ds[l]:
        labels.append(li); imgs.append(im)
labels = np.array(labels); N = len(imgs)
H, W = imgs[0].shape
L = [local_contrast_norm(im).astype(np.float64) for im in imgs]     # noqa
GEN = np.where(labels < 2)[0]; IMP = np.where(labels == 2)[0]
RI = np.where(labels == 0)[0]; RC = np.where(labels == 1)[0]


def integral(x):
    return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def build(mdx, mdy, minov):
    PH, PW = H + 2 * mdy + 1, W + 2 * mdx + 1
    R = {}
    for i in range(N):
        for d in ROTS:
            b = L[i] if d == 0 else _rotate(L[i].astype(np.float32), d).astype(np.float64)  # noqa
            R[(i, d)] = (np.conj(np.fft.rfft2(b, s=(PH, PW))), integral(b), integral(b * b))
    FA = [np.fft.rfft2(L[i], s=(PH, PW)) for i in range(N)]
    IA = [(integral(L[i]), integral(L[i] * L[i])) for i in range(N)]
    dys = np.arange(-mdy, mdy + 1); dxs = np.arange(-mdx, mdx + 1)
    DY = dys[:, None]; DX = dxs[None, :]
    ay0 = np.maximum(0, DY) + 0 * DX; ay1 = np.minimum(H, H + DY) + 0 * DX
    ax0 = np.maximum(0, DX) + 0 * DY; ax1 = np.minimum(W, W + DX) + 0 * DY
    by0, by1, bx0, bx1 = ay0 - DY, ay1 - DY, ax0 - DX, ax1 - DX
    n = (ay1 - ay0) * (ax1 - ax0); ok = n >= minov; nn = np.maximum(n, 1)
    rect = lambda ii, y0, y1, x0, x1: ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    S = np.full((N, N), np.nan); EDGE = np.zeros((N, N), bool)
    for t in range(N):
        ia, ia2 = IA[t]
        sa = rect(ia, ay0, ay1, ax0, ax1); sa2 = rect(ia2, ay0, ay1, ax0, ax1)
        va = sa2 - sa * sa / nn
        for p in range(N):
            if p == t:
                continue
            best = (-2.0, 0, 0)
            for d in ROTS:
                fbc, ib, ib2 = R[(p, d)]
                corr = np.fft.irfft2(FA[t] * fbc, s=(PH, PW))
                sb = rect(ib, by0, by1, bx0, bx1); sb2 = rect(ib2, by0, by1, bx0, bx1)
                vb = sb2 - sb * sb / nn
                num = corr[DY % PH, DX % PW] - sa * sb / nn
                den = np.sqrt(np.maximum(va, 0.) * np.maximum(vb, 0.))
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.where((den > 1e-9) & ok, num / den, -np.inf)
                k = int(np.argmax(c)); iy, ix = divmod(k, c.shape[1])
                if c[iy, ix] > best[0]:
                    best = (float(c[iy, ix]), int(dxs[ix]), int(dys[iy]))
            S[t, p] = best[0]
            EDGE[t, p] = abs(best[1]) == mdx or abs(best[2]) == mdy
    return S, EDGE


def dprime(g, i):
    g, i = np.asarray(g, float), np.asarray(i, float)
    d = math.sqrt((g.var() + i.var()) / 2)
    return float((g.mean() - i.mean()) / d) if d > 0 else 0.0


def eer(g, i):
    g, i = np.asarray(g, float), np.asarray(i, float)
    bg, be = math.inf, 1.0
    for t in np.unique(np.concatenate([g, i])):
        fr = float((g < t).mean()); fa = float((i >= t).mean())
        if abs(fr - fa) < bg:
            bg, be = abs(fr - fa), (fr + fa) / 2
    return be


def far10(g, i):
    t = float(np.quantile(np.asarray(g, float), 0.10))
    return float((np.asarray(i, float) >= t).mean())


def evalAB(S):
    rng = np.random.default_rng(3); rows = []
    for _ in range(500):
        perm = rng.permutation(GEN)
        T, pr = np.sort(perm[:19]), np.sort(perm[19:])
        g = S[np.ix_(T, pr)].max(axis=0); i = S[np.ix_(T, IMP)].max(axis=0)
        rows.append((dprime(g, i), eer(g, i), far10(g, i)))
    A = np.array(rows)
    g = S[np.ix_(RC, RI)].max(axis=0); i = S[np.ix_(RC, IMP)].max(axis=0)
    B = (dprime(g, i), eer(g, i), far10(g, i))
    return A, B


print("=" * 78)
print("10. ALIGNMENT SEARCH WINDOW SWEEP (LCN + NCC + rotation +-12 deg)")
print("=" * 78)
print(f"  {'window (dx,dy)':<16} {'min_ov':>7} {'edge%':>6} | "
      f"{'A d-prime':>13} {'A EER%':>7} {'A FAR@10':>9} | "
      f"{'B d-prime':>10} {'B EER%':>7} {'B FAR@10':>9}")
CFG = [(20, 8, 3500), (20, 8, 5000), (30, 10, 3500), (30, 10, 4500),
       (40, 12, 3000), (40, 12, 4500), (50, 16, 2500), (50, 16, 4500),
       (60, 20, 4500), (12, 5, 5000), (8, 4, 5500)]
for mdx, mdy, mo in CFG:
    t0 = time.time()
    S, E = build(mdx, mdy, mo)
    A, B = evalAB(S)
    ge = np.array([E[i, j] for i in GEN for j in GEN if i != j]).mean()
    print(f"  +-{mdx:2d},+-{mdy:2d}{'':<8} {mo:7d} {ge*100:5.0f}% | "
          f"{A[:,0].mean():6.2f}+-{A[:,0].std():4.2f} {A[:,1].mean()*100:6.1f} "
          f"{A[:,2].mean()*100:8.1f} | {B[0]:10.2f} {B[1]*100:6.1f} {B[2]*100:8.1f}"
          f"   [{time.time()-t0:.0f}s]")
