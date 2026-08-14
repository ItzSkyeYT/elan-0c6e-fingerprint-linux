"""
Descriptor matching v3 -- a SIFT-like descriptor built on RIDGE ORIENTATION.

Measured on this sensor the ridge period is about 10-11 px (autocorrelation
minimum at 5-6 px, next positive lobe at 10 px).  A radius-8 patch therefore
spans barely 1.5 ridges, which is nowhere near distinctive enough for a few
matches to establish identity -- and that, not the matching stage, is why v1/v2
land on top of the correlation baseline.

v3 uses support radii of 14-22 px (3-4 ridge periods) and pools soft-assigned
ridge-orientation histograms over a 4x4 spatial layout, SIFT-style:

    D[b, o] = sum_m  W_spatial[b, m] * coherence[m] * histbin_o(theta[m])

with linear interpolation in orientation and Gaussian soft assignment in space,
then L2 normalise / clip at 0.2 / renormalise.  Orientation is pi-periodic, so
8 bins cover 0..pi.

Samples that fall outside the 150x52 window carry zero weight, so a keypoint may
sit near the border and still be described -- important when the whole image is
only 5 ridges tall.

Rotation is handled globally (the probe is re-described at each of a few
rotations).  A per-keypoint canonical orientation is also implemented, and
measured, because the task asks for it -- but a ridge field is pi-periodic, so
the canonical direction has an unresolvable 180-degree ambiguity.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto import (load_all, scenario_A, scenario_B, summarise,
                   local_contrast_norm, gabor_enhance, bandpass, _sep_blur,
                   _rotate)                                              # noqa
from desc_v1 import enhance, orientation, validity, grid_points, harris_points

H, W = 52, 150


# ------------------------------------------------------ descriptor geometry --

def _layout(R, step, nsp, nori, sfrac=0.42):
    """Sample offsets, spatial pooling weights (nsp^2, M), orientation bin count."""
    o = np.arange(-R, R + 1, step, dtype=np.float32)
    dy, dx = np.meshgrid(o, o, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    inside = (dy ** 2 + dx ** 2) <= (R + 0.5) ** 2
    dy, dx = dy[inside], dx[inside]

    c = (np.arange(nsp, dtype=np.float32) + 0.5) / nsp * 2 * R - R
    sb = sfrac * (2.0 * R / nsp)
    Wm = np.empty((nsp * nsp, dy.size), np.float32)
    for i in range(nsp):
        for j in range(nsp):
            Wm[i * nsp + j] = np.exp(-((dy - c[i]) ** 2 + (dx - c[j]) ** 2)
                                     / (2 * sb * sb))
    # overall Gaussian window over the support
    Wm *= np.exp(-(dy ** 2 + dx ** 2) / (2 * (0.55 * R) ** 2))[None, :]
    return dy.astype(np.int32), dx.astype(np.int32), Wm


def describe_ori(theta, coh, pts, R, step, nsp=4, nori=8, canon=False,
                 clip=0.2):
    """SIFT-like ridge-orientation descriptor.  theta in [0, pi)."""
    K = len(pts)
    if K == 0:
        return np.zeros((0, nsp * nsp * nori), np.float32), np.zeros((0, 2), np.int32)
    dy, dx, Wm = _layout(R, step, nsp, nori)
    M = dy.size
    P = np.array(pts, np.int32)

    yy = P[:, 0:1] + dy[None, :]
    xx = P[:, 1:2] + dx[None, :]
    ok = ((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)).astype(np.float32)
    yc = np.clip(yy, 0, H - 1)
    xc = np.clip(xx, 0, W - 1)

    th = theta[yc, xc]                                   # (K, M) in [0, pi)
    wt = coh[yc, xc] * ok                                # (K, M)

    if canon:
        # dominant local orientation from the doubled-angle mean, subtracted so
        # the descriptor is rotation-normalised in-place
        cx = (wt * np.cos(2 * th)).sum(1)
        sy = (wt * np.sin(2 * th)).sum(1)
        dom = 0.5 * np.arctan2(sy, cx)                   # (K,)
        th = (th - dom[:, None]) % math.pi

    # linear interpolation into nori orientation bins covering [0, pi)
    f = th / math.pi * nori
    b0 = np.floor(f).astype(np.int32) % nori
    fr = (f - np.floor(f)).astype(np.float32)
    b1 = (b0 + 1) % nori

    D = np.empty((K, nsp * nsp * nori), np.float32)
    for o in range(nori):
        A = wt * ((b0 == o) * (1.0 - fr) + (b1 == o) * fr)
        D[:, o::nori] = A @ Wm.T                          # (K, nsp*nsp)

    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    if clip:
        np.clip(D, 0, clip, out=D)
    # Mean-centre before the final normalisation.  A histogram descriptor is
    # all-positive, so the raw cosine between ANY two of them sits around
    # 0.6-0.9 and a similarity threshold means nothing; centring turns the
    # cosine into a correlation coefficient with the usual [-1, 1] range.
    D -= D.mean(axis=1, keepdims=True)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    return D, P


def describe_patch(e, pts, R, step, canon=None):
    if not len(pts):
        return np.zeros((0, 1), np.float32), np.zeros((0, 2), np.int32)
    dy, dx, _ = _layout(R, step, 1, 1)
    P = np.array(pts, np.int32)
    yy = np.clip(P[:, 0:1] + dy[None, :], 0, H - 1)
    xx = np.clip(P[:, 1:2] + dx[None, :], 0, W - 1)
    ok = ((P[:, 0:1] + dy[None, :] >= 0) & (P[:, 0:1] + dy[None, :] < H) &
          (P[:, 1:2] + dx[None, :] >= 0) & (P[:, 1:2] + dx[None, :] < W))
    w = np.exp(-(dy ** 2 + dx ** 2) / (2 * (0.55 * R) ** 2)).astype(np.float32)
    v = e[yy, xx] * w * ok
    v -= v.mean(axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v.astype(np.float32), P


# ------------------------------------------------------------------- views --

class View:
    __slots__ = ("D", "P", "n")

    def __init__(self, D, P):
        self.D, self.P, self.n = D, P, len(P)


def build_views(img, cfg):
    out = []
    for deg in cfg["rots"]:
        src = _rotate(img, deg) if deg else img
        e = enhance(src, cfg["enh"])
        ox, oy, coh = orientation(e, cfg["osmooth"])
        theta = (0.5 * np.arctan2(oy, ox)) % math.pi     # ridge-normal angle
        coh = np.clip(coh, 0, 1) ** cfg["cohpow"]
        mask = validity(src, frac=cfg["vfrac"])
        if cfg["kp"] == "grid":
            pts = grid_points(mask, cfg["margin"], cfg["step"])
        else:
            pts = harris_points(e, mask, cfg["margin"], cfg["nmax"], cfg["nms"])
        parts, P = [], None
        if cfg["desc"] in ("ori", "both"):
            D, P = describe_ori(theta, coh, pts, cfg["R"], cfg["sstep"],
                                cfg["nsp"], cfg["nori"], cfg["canon"], cfg["clip"])
            parts.append(D)
        if cfg["desc"] in ("patch", "both"):
            D, P = describe_patch(e, pts, cfg["R"], cfg["sstep"])
            parts.append(D)
        D = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if len(parts) > 1:
            D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        out.append(View(D.astype(np.float32), P))
    return out


# ---------------------------------------------------------------- matching --

def match_pair(vt, vp, DT2, cfg):
    nt, npb = vt.n, vp.n
    if nt < 5 or npb < 5:
        return None
    S = vp.D @ vt.D.T
    best = np.argmax(S, axis=1)
    sbest = S[np.arange(npb), best]

    if cfg["ratio"] < 1.0:
        S2 = np.where(DT2[best] < cfg["excl"] ** 2, -2.0, S)
        keep = (sbest > cfg["simmin"]) & (S2.max(axis=1) < cfg["ratio"] * sbest)
    else:
        keep = sbest > cfg["simmin"]
    if keep.sum() < 3:
        return None

    ty = vt.P[best][:, 0] - vp.P[:, 0]
    tx = vt.P[best][:, 1] - vp.P[:, 1]
    b = cfg["bin"]
    ny, nx = (2 * H) // b + 3, (2 * W) // b + 3
    iy = np.clip((ty[keep] + H) // b, 0, ny - 1).astype(np.int32)
    ix = np.clip((tx[keep] + W) // b, 0, nx - 1).astype(np.int32)
    wts = (sbest[keep] - cfg["simmin"]).astype(np.float32)
    acc = np.bincount(iy * nx + ix, weights=wts, minlength=ny * nx
                      ).astype(np.float32).reshape(ny, nx)
    pooled = (acc[:-2, :-2] + acc[:-2, 1:-1] + acc[:-2, 2:] +
              acc[1:-1, :-2] + acc[1:-1, 1:-1] + acc[1:-1, 2:] +
              acc[2:, :-2] + acc[2:, 1:-1] + acc[2:, 2:])
    k = int(np.argmax(pooled))
    by, bx = divmod(k, pooled.shape[1])
    mass = float(pooled[by, bx])
    sel = (iy >= by) & (iy <= by + 2) & (ix >= bx) & (ix <= bx + 2)
    cnt = int(sel.sum())
    if cnt == 0:
        return None
    kidx = np.nonzero(keep)[0][sel]
    tyc, txc = float(ty[kidx].mean()), float(tx[kidx].mean())

    tol = cfg["tol"]
    agree = (np.abs(ty - tyc) <= tol) & (np.abs(tx - txc) <= tol) & \
            (sbest > cfg["simmin"])
    mass2 = float((sbest[agree] - cfg["simmin"]).sum())
    cnt2 = int(agree.sum())

    py = vp.P[:, 0] + tyc
    px = vp.P[:, 1] + txc
    n_ov = int((((py >= vt.P[:, 0].min()) & (py <= vt.P[:, 0].max()) &
                 (px >= vt.P[:, 1].min()) & (px <= vt.P[:, 1].max())).sum()))
    den = max(n_ov, cfg["ovfloor"])
    geo = math.sqrt(nt * npb)
    return {"mass": mass, "sqrt": mass / geo, "cnt": cnt / geo,
            "mass2": mass2, "sqrt2": mass2 / geo, "cnt2": cnt2 / geo,
            "ov2": mass2 / den, "mix2": math.sqrt(max(mass2, 0)) * math.sqrt(mass2 / den),
            "n_ov": n_ov}


VARIANTS = ("sqrt", "cnt", "sqrt2", "cnt2", "ov2", "mix2")

DEF3 = dict(enh="lcn", desc="ori", kp="grid", R=16, sstep=2, nsp=4, nori=8,
            clip=0.2, canon=False, osmooth=3.0, cohpow=1.0, vfrac=0.35,
            margin=4, step=4, nmax=200, nms=5,
            ratio=0.92, excl=16, simmin=0.55, bin=5, tol=6.0, ovfloor=60,
            rots=(-12, -6, 0, 6, 12))


def build(cfg, imgs=None, idx=None, verbose=True):
    if imgs is None:
        imgs, _, idx = load_all()
    t0 = time.time()
    views = [build_views(im, cfg) for im in imgs]
    r0 = len(cfg["rots"]) // 2
    counts = [v[r0].n for v in views]
    if verbose:
        print(f"  keypoints/image mean {np.mean(counts):.0f} "
              f"[{min(counts)}..{max(counts)}]  dim {views[0][0].D.shape[1]}  "
              f"({time.time()-t0:.0f}s extract)", flush=True)
    DT2 = [((v[r0].P[None, :, :] - v[r0].P[:, None, :]) ** 2).sum(-1).astype(np.float32)
           for v in views]
    n = len(imgs)
    Ms = {v: np.zeros((n, n), np.float32) for v in VARIANTS}
    # "select the alignment by vote mass, then score it" -- v2 let each variant
    # pick its own best rotation, which let the overlap-normalised scores chase
    # small-overlap flukes.
    for i in range(n):
        vt = views[i][r0]
        for j in range(n):
            bestmass, bestres = -1.0, None
            for r in range(len(cfg["rots"])):
                res = match_pair(vt, views[j][r], DT2[i], cfg)
                if res is not None and res["mass"] > bestmass:
                    bestmass, bestres = res["mass"], res
            for v in VARIANTS:
                Ms[v][i, j] = bestres[v] if bestres else 0.0
    if verbose:
        print(f"  {n*n} pairs ({time.time()-t0:.0f}s)", flush=True)
    return Ms, idx, n, counts


def report(Ms, idx, n, tag, top=99):
    rows = []
    for v, M in Ms.items():
        sc = lambda i, j: float(M[i, j])
        bad = sum(1 for i in range(n)
                  if M[i, i] < max(M[j, i] for j in range(n) if j != i))
        tA, _, _ = scenario_A(sc, idx)
        mB, _, _ = scenario_B(sc, idx)
        rows.append(dict(v=v, bad=bad,
                         dA=float(np.mean([t["dprime"] for t in tA])),
                         sdA=float(np.std([t["dprime"] for t in tA])),
                         eA=float(np.mean([t["eer"] for t in tA])),
                         fA=float(np.mean([t["far10"] for t in tA])),
                         dB=mB["dprime"], eB=mB["eer"], fB=mB["far10"]))
    rows.sort(key=lambda r: -(r["dA"] + r["dB"]))
    for r in rows[:top]:
        print(f"  {tag:20s} {r['v']:7s} selffail {r['bad']:2d}  "
              f"A d'={r['dA']:5.2f}+/-{r['sdA']:.2f} EER={r['eA']*100:5.1f}% "
              f"FAR10={r['fA']*100:5.1f}%  |  B d'={r['dB']:5.2f} "
              f"EER={r['eB']*100:5.1f}% FAR10={r['fB']*100:5.1f}%", flush=True)
    return rows


if __name__ == "__main__":
    imgs, _, idx = load_all()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = [
        ("R16 ori", dict(DEF3)),
        ("R22 ori", dict(DEF3, R=22, sstep=3, excl=22)),
        ("R12 ori", dict(DEF3, R=12, excl=12)),
        ("R16 ori canon", dict(DEF3, canon=True)),
        ("R16 both", dict(DEF3, desc="both")),
        ("R16 gabor", dict(DEF3, enh="gabor")),
        ("R16 harris", dict(DEF3, kp="harris", nmax=180, nms=6, margin=2)),
    ]
    saved = {}
    for tag, cfg in cfgs:
        if which != "all" and which not in tag:
            continue
        print(f"\n===== {tag} =====", flush=True)
        Ms, _, n, counts = build(cfg, imgs, idx)
        report(Ms, idx, n, tag)
        np.savez(f"v3_{tag.replace(' ', '_')}.npz", **Ms)
