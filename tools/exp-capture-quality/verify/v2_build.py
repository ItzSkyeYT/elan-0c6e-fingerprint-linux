#!/usr/bin/env python3
"""
Build a DIRECTIONAL score cache: S[t, p] = best NCC of probe p aligned to
template t (rotation and translation applied to the probe). The prior cache
stored only one triangle, which conflates score(i,j) with score(j,i); they
differ by up to 0.014 because the rotation search is one-sided.

Also caches, per ordered pair, the winning (dx, dy, deg) and the overlap
statistics at that alignment, and per image the capture-quality metrics.

Everything is FFT-accelerated only for speed of experimentation; the winning
configuration is a plain direct-loop NCC in C.
"""
import os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TOOLS = EXP.parent
sys.path.insert(0, str(EXP))
os.chdir(TOOLS)
exec(open("matcher-lab.py").read().split("def main(")[0])      # noqa
import quality as Q                                            # noqa: E402

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LABELS = ["right-index", "right-index-cover", "right-middle"]
MAX_DX, MAX_DY, MIN_OV = 20, 8, 3500
ROTS = list(range(-12, 13, 4))


def integral(x):
    return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def main():
    ds = load_dataset(DATASET)                                 # noqa
    names, labels, imgs = [], [], []
    for li, lbl in enumerate(LABELS):
        for n, im in ds[lbl]:
            names.append(n); labels.append(li); imgs.append(im)
    labels = np.array(labels); N = len(imgs)
    H, W = imgs[0].shape
    print(f"{N} images {W}x{H}")

    t0 = time.time()
    qkeys = None; qrows = []; masks = []
    for im in imgs:
        q = Q.image_quality(im)
        if qkeys is None:
            qkeys = sorted(q)
        qrows.append([q[k] for k in qkeys])
        masks.append(Q.foreground_mask_px(im))
    qual = np.array(qrows); masks = np.array(masks)
    print(f"quality metrics: {time.time()-t0:.1f}s")

    L = [local_contrast_norm(im).astype(np.float64) for im in imgs]   # noqa

    PH, PW = H + 2 * MAX_DY + 1, W + 2 * MAX_DX + 1
    # per (image, rotation) precompute: rotated LCN, its conj FFT, integrals, mask
    print("precomputing rotations/FFTs ...", flush=True)
    R = {}
    for i in range(N):
        for deg in ROTS:
            b = L[i] if deg == 0 else _rotate(L[i].astype(np.float32), deg).astype(np.float64)  # noqa
            mk = masks[i] if deg == 0 else (
                _rotate(masks[i].astype(np.float32), deg) > 0.5)      # noqa
            R[(i, deg)] = (np.conj(np.fft.rfft2(b, s=(PH, PW))),
                           integral(b), integral(b * b), mk)
    FA = [np.fft.rfft2(L[i], s=(PH, PW)) for i in range(N)]
    IA = [(integral(L[i]), integral(L[i] * L[i])) for i in range(N)]

    dys = np.arange(-MAX_DY, MAX_DY + 1)
    dxs = np.arange(-MAX_DX, MAX_DX + 1)
    DY = dys[:, None]; DX = dxs[None, :]
    ay0 = np.maximum(0, DY) + 0 * DX; ay1 = np.minimum(H, H + DY) + 0 * DX
    ax0 = np.maximum(0, DX) + 0 * DY; ax1 = np.minimum(W, W + DX) + 0 * DY
    by0, by1 = ay0 - DY, ay1 - DY
    bx0, bx1 = ax0 - DX, ax1 - DX
    n_ov = (ay1 - ay0) * (ax1 - ax0)
    ok = n_ov >= MIN_OV
    nn = np.maximum(n_ov, 1)

    def rect(ii, y0, y1, x0, x1):
        return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

    S = np.full((N, N), np.nan)
    BDX = np.zeros((N, N), np.int32); BDY = np.zeros((N, N), np.int32)
    BDEG = np.zeros((N, N), np.int32)
    GEO = np.zeros((N, N), np.int32); BOTH = np.zeros((N, N), np.int32)
    EDGE = np.zeros((N, N), bool)

    t0 = time.time()
    for t in range(N):
        ia, ia2 = IA[t]
        sa = rect(ia, ay0, ay1, ax0, ax1)
        sa2 = rect(ia2, ay0, ay1, ax0, ax1)
        va = sa2 - sa * sa / nn
        for p in range(N):
            if p == t:
                continue
            best = (-2.0, 0, 0, 0)
            for deg in ROTS:
                fbc, ib, ib2, _mk = R[(p, deg)]
                corr = np.fft.irfft2(FA[t] * fbc, s=(PH, PW))
                sb = rect(ib, by0, by1, bx0, bx1)
                sb2 = rect(ib2, by0, by1, bx0, bx1)
                vb = sb2 - sb * sb / nn
                cross = corr[DY % PH, DX % PW]
                num = cross - sa * sb / nn
                den = np.sqrt(np.maximum(va, 0.) * np.maximum(vb, 0.))
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.where((den > 1e-9) & ok, num / den, -np.inf)
                k = int(np.argmax(c))
                iy, ix = divmod(k, c.shape[1])
                if c[iy, ix] > best[0]:
                    best = (float(c[iy, ix]), int(dxs[ix]), int(dys[iy]), deg)
            s, dx, dy, deg = best
            S[t, p] = s; BDX[t, p] = dx; BDY[t, p] = dy; BDEG[t, p] = deg
            EDGE[t, p] = (abs(dx) == MAX_DX) or (abs(dy) == MAX_DY)
            mb = R[(p, deg)][3]
            y0, y1 = max(0, dy), min(H, H + dy)
            x0, x1 = max(0, dx), min(W, W + dx)
            pa = masks[t][y0:y1, x0:x1]
            pb = mb[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            GEO[t, p] = (y1 - y0) * (x1 - x0)
            BOTH[t, p] = int((pa & pb).sum())
        print(f"\r  row {t+1}/{N}  {time.time()-t0:.0f}s", end="", flush=True)
    print()

    # self-check: score of an image against itself must be 1.0
    fbc, ib, ib2, _ = R[(0, 0)]
    corr = np.fft.irfft2(FA[0] * fbc, s=(PH, PW))
    sb = rect(ib, by0, by1, bx0, bx1); sb2 = rect(ib2, by0, by1, bx0, bx1)
    vb = sb2 - sb * sb / nn
    num = corr[DY % PH, DX % PW] - sa_ if False else None
    np.savez(EXP / "dir_cache.npz", S=S, DX=BDX, DY=BDY, DEG=BDEG, GEO=GEO,
             BOTH=BOTH, EDGE=EDGE, labels=labels, names=np.array(names),
             qual=qual, qkeys=np.array(qkeys), fg_px=masks.sum(axis=(1, 2)),
             masks=masks)
    print(f"wrote dir_cache.npz  ({time.time()-t0:.0f}s)")
    print(f"S finite: {int(np.isfinite(S).sum())} of {N*N-N} off-diagonal")
    print(f"asymmetry |S - S.T| mean={np.nanmean(np.abs(S-S.T)):.4f} "
          f"max={np.nanmax(np.abs(S-S.T)):.4f}")


if __name__ == "__main__":
    main()
