"""
Local-descriptor fingerprint matching for the 150x52 ELAN press sensor.

Pipeline
  1. enhance          local contrast normalisation (optionally Gabor)
  2. keypoints        dense grid (or Harris on the enhanced image) restricted to
                      a ridge-validity mask
  3. descriptor       Gaussian-windowed local patch, mean-removed, L2-normalised
                      -- either the enhanced intensity, or the doubled-angle
                      orientation field (cos2t, sin2t) scaled by coherence
  4. rotation         handled GLOBALLY: the probe is re-extracted at each of a
                      few rotations, so the residual transform is a pure
                      translation.  Per-keypoint canonical orientation is
                      unreliable on ridge fields (pi-periodic, so the 180-degree
                      flip is ambiguous), and is offered as a variant to check.
  5. matching         cosine NN + ratio test with spatial exclusion
  6. geometry         Hough vote over translation, 3x3 bin pooling
  7. score            weighted vote mass of the winning translation

Everything is expressible in plain C: separable blurs, a bilinear rotate, dot
products and an integer accumulator.  No FFT, no libraries.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto import (load_all, scenario_A, scenario_B, summarise, selfcheck,
                   local_contrast_norm, gabor_enhance, bandpass, _sep_blur,
                   _rotate)                                            # noqa

H, W = 52, 150


# ------------------------------------------------------------- enhancement --

def enhance(img, kind):
    if kind == "lcn":
        return local_contrast_norm(img)
    if kind == "gabor":
        return gabor_enhance(img)
    if kind == "bp":
        e = bandpass(img)
        return (e - e.mean()) / (e.std() + 1e-6)
    raise ValueError(kind)


def orientation(img, smooth=3.0):
    """Doubled-angle orientation field + coherence, both at pixel resolution."""
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    vx = _sep_blur(2 * gx * gy, smooth)
    vy = _sep_blur(gx * gx - gy * gy, smooth)
    en = _sep_blur(gx * gx + gy * gy, smooth) + 1e-6
    mag = np.sqrt(vx * vx + vy * vy)
    coh = mag / en
    return vx / (mag + 1e-6), vy / (mag + 1e-6), coh


def validity(img, sigma=5.0, frac=0.35):
    """Mask of pixels that actually carry ridge signal.  The driver already
    background-subtracts, but a press rarely fills the whole 150x52 window."""
    m = _sep_blur(img, sigma)
    v = np.sqrt(np.maximum(_sep_blur((img - m) ** 2, sigma), 0.0))
    return v > frac * float(np.median(v[v > 0])) if np.any(v > 0) else v > 0


# --------------------------------------------------------------- keypoints --

def harris_points(e, mask, r, n_max=200, nms=4):
    """Harris corner response on the enhanced image.  On a ridge field the
    strong responses land on ridge endings, bifurcations and sharp curvature --
    minutiae-like, but without a minutiae extractor."""
    gx = np.gradient(e, axis=1)
    gy = np.gradient(e, axis=0)
    a = _sep_blur(gx * gx, 2.0)
    b = _sep_blur(gy * gy, 2.0)
    c = _sep_blur(gx * gy, 2.0)
    resp = (a * b - c * c) - 0.04 * (a + b) ** 2

    ok = np.zeros_like(mask)
    ok[r:H - r, r:W - r] = True
    ok &= mask
    ys, xs = np.nonzero(ok)
    vals = resp[ys, xs]
    order = np.argsort(-vals)
    taken = []
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        if all((y - yy) ** 2 + (x - xx) ** 2 >= nms * nms for yy, xx in taken):
            taken.append((y, x))
            if len(taken) >= n_max:
                break
    return taken


def grid_points(mask, r, step):
    pts = []
    for y in range(r, H - r, step):
        for x in range(r, W - r, step):
            if mask[y, x]:
                pts.append((y, x))
    return pts


# -------------------------------------------------------------- descriptor --

def _offsets(r, sub):
    o = np.arange(-r, r + 1, sub)
    dy, dx = np.meshgrid(o, o, indexing="ij")
    return dy.ravel(), dx.ravel()


def describe(e, ori, pts, r, sub, kind):
    """Return (K, D) L2-normalised descriptors and (K, 2) coordinates."""
    if not pts:
        return np.zeros((0, 1), np.float32), np.zeros((0, 2), np.int32)
    dy, dx = _offsets(r, sub)
    w = np.exp(-(dy.astype(np.float32) ** 2 + dx.astype(np.float32) ** 2)
               / (2 * (0.6 * r) ** 2))
    P = np.array(pts, np.int32)
    yy = P[:, 0:1] + dy[None, :]
    xx = P[:, 1:2] + dx[None, :]

    parts = []
    if kind in ("patch", "both"):
        v = e[yy, xx] * w
        parts.append(v - v.mean(axis=1, keepdims=True))
    if kind in ("orient", "both"):
        ox, oy, coh = ori
        parts.append(ox[yy, xx] * coh[yy, xx] * w)
        parts.append(oy[yy, xx] * coh[yy, xx] * w)
    D = np.concatenate(parts, axis=1).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    return D, P


# ------------------------------------------------------------- one "view" --

class View:
    __slots__ = ("D", "P", "n")

    def __init__(self, D, P):
        self.D, self.P, self.n = D, P, len(P)


def build_views(img, cfg, rots):
    """Descriptors for one image at each global rotation."""
    out = []
    for deg in rots:
        src = _rotate(img, deg) if deg else img
        e = enhance(src, cfg["enh"])
        ori = orientation(e, cfg["osmooth"]) if cfg["desc"] != "patch" else None
        mask = validity(src, frac=cfg["vfrac"])
        if cfg["kp"] == "grid":
            pts = grid_points(mask, cfg["r"], cfg["step"])
        else:
            pts = harris_points(e, mask, cfg["r"], cfg["nmax"], cfg["nms"])
        D, P = describe(e, ori, pts, cfg["r"], cfg["sub"], cfg["desc"])
        out.append(View(D, P))
    return out


# ---------------------------------------------------------------- matching --

def match_score(vt, vp, cfg, acc):
    """Descriptor NN + ratio test + translation Hough between one template view
    and one probe view.  Returns (vote mass, inlier count)."""
    if vt.n < 4 or vp.n < 4:
        return 0.0, 0
    S = vp.D @ vt.D.T                          # (np, nt) cosine similarity
    best = np.argmax(S, axis=1)
    sbest = S[np.arange(vp.n), best]

    if cfg["ratio"] < 1.0:
        # second best, excluding template keypoints near the winner (adjacent
        # patches overlap, so an unguarded ratio test rejects every match)
        bt = vt.P[best]                                    # (np, 2)
        d2 = ((vt.P[None, :, :] - bt[:, None, :]) ** 2).sum(-1)
        S2 = np.where(d2 < cfg["excl"] ** 2, -2.0, S)
        s2 = S2.max(axis=1)
        keep = (sbest > cfg["simmin"]) & (s2 < cfg["ratio"] * sbest)
    else:
        keep = sbest > cfg["simmin"]
    if keep.sum() < 3:
        return 0.0, 0

    ty = vt.P[best][keep, 0] - vp.P[keep, 0]
    tx = vt.P[best][keep, 1] - vp.P[keep, 1]
    wts = (sbest[keep] - cfg["simmin"]).astype(np.float32)

    b = cfg["bin"]
    acc[:] = 0
    iy = np.clip((ty + H) // b, 0, acc.shape[0] - 1).astype(np.int32)
    ix = np.clip((tx + W) // b, 0, acc.shape[1] - 1).astype(np.int32)
    np.add.at(acc, (iy, ix), wts)
    # pool a 3x3 neighbourhood so a match straddling a bin edge is not split
    pooled = (acc[:-2, :-2] + acc[:-2, 1:-1] + acc[:-2, 2:] +
              acc[1:-1, :-2] + acc[1:-1, 1:-1] + acc[1:-1, 2:] +
              acc[2:, :-2] + acc[2:, 1:-1] + acc[2:, 2:])
    k = int(np.argmax(pooled))
    by, bx = divmod(k, pooled.shape[1])
    mass = float(pooled[by, bx])

    # inliers: matches whose translation is inside the winning 3x3 block
    sel = (iy >= by) & (iy <= by + 2) & (ix >= bx) & (ix <= bx + 2)
    return mass, int(sel.sum())


DEFAULT = dict(enh="lcn", desc="patch", kp="grid", r=8, sub=2, step=4,
               osmooth=3.0, vfrac=0.35, ratio=0.92, excl=12, simmin=0.30,
               bin=5, nmax=200, nms=4,
               rots=(-12, -6, 0, 6, 12), norm="sqrt")


def build_matcher(cfg, verbose=True):
    imgs, labels, idx = load_all()
    t0 = time.time()
    views = [build_views(im, cfg, cfg["rots"]) for im in imgs]
    counts = [v[len(cfg["rots"]) // 2].n for v in views]
    if verbose:
        print(f"  keypoints/image: mean {np.mean(counts):.0f} "
              f"[{min(counts)}..{max(counts)}]   descriptor dim "
              f"{views[0][0].D.shape[1]}   ({time.time()-t0:.0f}s extract)")

    acc = np.zeros(((2 * H) // cfg["bin"] + 3, (2 * W) // cfg["bin"] + 3), np.float32)
    n = len(imgs)
    M = np.zeros((n, n), np.float32)
    r0 = len(cfg["rots"]) // 2
    for i in range(n):
        for j in range(n):
            best = 0.0
            for r in range(len(cfg["rots"])):
                mass, cnt = match_score(views[i][r0], views[j][r], cfg, acc)
                if cfg["norm"] == "sqrt":
                    s = mass / math.sqrt(max(views[i][r0].n, 1) * max(views[j][r].n, 1))
                elif cfg["norm"] == "probe":
                    s = mass / max(views[j][r].n, 1)
                else:
                    s = mass
                if s > best:
                    best = s
            M[i, j] = best
    if verbose:
        print(f"  {n*n} ordered pairs scored ({time.time()-t0:.0f}s total)")
    return (lambda i, j: float(M[i, j])), idx, n, M, counts


def run(name, cfg, verbose=True):
    score, idx, n, M, counts = build_matcher(cfg, verbose)
    bad = selfcheck(score, n, name)
    tA, gA, iA = scenario_A(score, idx)
    mB, gB, iB = scenario_B(score, idx)
    res = summarise(name, tA, gA, iA, mB, gB, iB,
                    extra=f"(selfcheck fails {bad})")
    return res, M


if __name__ == "__main__":
    cfg = dict(DEFAULT)
    for a in sys.argv[1:]:
        k, v = a.split("=")
        cfg[k] = type(DEFAULT[k])(v) if not isinstance(DEFAULT[k], tuple) \
            else tuple(float(x) for x in v.split(","))
    run("descriptors " + str({k: cfg[k] for k in ("enh", "desc", "kp", "r", "step")}),
        cfg)
