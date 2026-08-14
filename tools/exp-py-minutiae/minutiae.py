#!/usr/bin/env python3
"""
Minutiae extraction + partial-overlap matching for the ELAN 04f3:0c6e press
sensor (150x52).

Pipeline
    1. segmentation      block variance -> finger mask
    2. normalisation     local contrast normalisation (matcher-lab helper)
    3. orientation field block gradient, doubled-angle averaging
    4. ridge frequency   fixed prior 0.110 cyc/px (measured on this sensor),
                         optionally re-estimated per block and clamped
    5. enhancement       orientation-steered Gabor
    6. binarise + thin   local-mean threshold, Zhang-Suen skeleton
    7. minutiae          crossing number (CN=1 ending, CN=3 bifurcation)
    8. cleanup           border drop, near-pair merge, spur removal
    9. matching          Hough vote over (rotation, translation), greedy
                         pairing, score normalised by the minutiae count in
                         the OVERLAP region

Everything is plain numpy; no scipy, no FFT.  Designed so the whole thing maps
to straightforward C.
"""

import math
import numpy as np

W, H = 150, 52

# ------------------------------------------------------------- primitives --

def _gauss_kernel(sigma):
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _sep_blur(img, sigma):
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(img)
    for i in range(len(k)):
        out += k[i] * p[:, i:i + img.shape[1]]
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    res = np.zeros_like(img)
    for i in range(len(k)):
        res += k[i] * p[i:i + img.shape[0], :]
    return res


def local_contrast_norm(img, sigma=6.0, eps=1e-3):
    m = _sep_blur(img, sigma)
    d = img - m
    v = np.sqrt(np.maximum(_sep_blur(d * d, sigma), 0.0))
    return d / (v + eps * v.mean() + 1e-6)


def _boxmean(img, k):
    """Mean over a (2k+1)^2 window, edge-replicated. Separable, integral-image
    equivalent; trivially portable."""
    p = np.pad(img, ((0, 0), (k, k)), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((img.shape[0], 1), np.float64),
                                  p.astype(np.float64)], axis=1), axis=1)
    out = (c[:, 2 * k + 1:] - c[:, :-(2 * k + 1)]) / (2 * k + 1)
    p = np.pad(out, ((k, k), (0, 0)), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((1, img.shape[1])), p], axis=0), axis=0)
    out = (c[2 * k + 1:, :] - c[:-(2 * k + 1), :]) / (2 * k + 1)
    return out.astype(np.float32)


# ----------------------------------------------------------- segmentation --

def segment(img, block=8, thresh_frac=0.30, min_std=6.0):
    """Block-variance foreground mask, then a little morphology.

    thresh_frac: a block is foreground if its stddev exceeds
    thresh_frac * (global mean block stddev), with an absolute floor.
    """
    nby, nbx = H // block, W // block
    blk = np.zeros((nby, nbx), np.float32)
    for by in range(nby):
        for bx in range(nbx):
            t = img[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            blk[by, bx] = t.std()
    thr = max(min_std, thresh_frac * float(blk.mean()))
    m = blk >= thr
    # majority filter (fill holes / trim isolated blocks)
    for _ in range(2):
        pad = np.pad(m.astype(np.int32), 1, mode="edge")
        s = np.zeros_like(m, dtype=np.int32)
        for dy in range(3):
            for dx in range(3):
                s += pad[dy:dy + nby, dx:dx + nbx]
        m = s >= 5
    mask = np.zeros((H, W), bool)
    for by in range(nby):
        for bx in range(nbx):
            if m[by, bx]:
                mask[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = True
    # remaining rows/cols not covered by whole blocks: copy the last block row
    if nby * block < H:
        mask[nby * block:, :] = mask[nby * block - 1, :]
    if nbx * block < W:
        mask[:, nbx * block:] = mask[:, nbx * block - 1][:, None]
    return mask


def _erode(mask, r):
    """Binary erosion by a (2r+1)^2 box."""
    if r <= 0:
        return mask
    s = _boxmean(mask.astype(np.float32), r)
    return s > 0.999


def _dilate(mask, r):
    if r <= 0:
        return mask
    s = _boxmean(mask.astype(np.float32), r)
    return s > 1e-6


# ------------------------------------------------------- orientation field --

def orientation(img, block=8, pre=1.0, smooth=2.5):
    """Ridge ORIENTATION (direction along the ridge), radians in [0,pi).

    Doubled-angle averaging over blocks then Gaussian-smoothed and returned at
    full resolution (nearest-block lookup).
    """
    g = _sep_blur(img, pre)
    gx = np.gradient(g, axis=1)
    gy = np.gradient(g, axis=0)
    vx = 2.0 * gx * gy
    vy = gx ** 2 - gy ** 2
    nby, nbx = (H + block - 1) // block, (W + block - 1) // block
    bx_ = np.zeros((nby, nbx), np.float32)
    by_ = np.zeros((nby, nbx), np.float32)
    for by in range(nby):
        for bx in range(nbx):
            ys, xs = slice(by * block, (by + 1) * block), slice(bx * block, (bx + 1) * block)
            bx_[by, bx] = vx[ys, xs].sum()
            by_[by, bx] = vy[ys, xs].sum()
    bx_ = _sep_blur(bx_, smooth)
    by_ = _sep_blur(by_, smooth)
    # coherence = |vector sum| / sum of magnitudes, per block
    mag = np.sqrt(bx_ ** 2 + by_ ** 2)
    ang = 0.5 * np.arctan2(bx_, by_)          # ridge-NORMAL angle
    ridge = (ang + math.pi / 2) % math.pi     # ridge direction
    # upsample nearest
    yy = np.clip(np.arange(H) // block, 0, nby - 1)
    xx = np.clip(np.arange(W) // block, 0, nbx - 1)
    full = ridge[np.ix_(yy, xx)]
    coh_full = mag[np.ix_(yy, xx)]
    return full.astype(np.float32), coh_full.astype(np.float32)


# ------------------------------------------------------- ridge frequency ---

def ridge_frequency(img, theta, block=16, lo=0.07, hi=0.16, prior=0.110):
    """Per-block dominant ridge frequency by projecting perpendicular to the
    ridge and finding the mean peak spacing.  Clamped to [lo,hi]; blocks with
    no reliable estimate get the prior."""
    nby, nbx = (H + block - 1) // block, (W + block - 1) // block
    f = np.full((nby, nbx), prior, np.float32)
    for by in range(nby):
        for bx in range(nbx):
            y0, x0 = by * block, bx * block
            cy, cx = y0 + block / 2, x0 + block / 2
            if cy >= H or cx >= W:
                continue
            t = float(theta[int(min(cy, H - 1)), int(min(cx, W - 1))])
            # sample a 32x16 oriented window: u along ridge, v across
            nu, nv = 24, 17
            uu, vv = np.mgrid[-nu // 2:nu // 2, -nv // 2:nv // 2 + 1].astype(np.float32)
            X = cx + uu * math.cos(t) - vv * math.sin(t)
            Y = cy + uu * math.sin(t) + vv * math.cos(t)
            Xi = np.clip(np.round(X).astype(int), 0, W - 1)
            Yi = np.clip(np.round(Y).astype(int), 0, H - 1)
            prof = img[Yi, Xi].mean(axis=0)   # average along the ridge
            prof = prof - prof.mean()
            # peak spacing
            pk = [i for i in range(1, len(prof) - 1)
                  if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > 0]
            if len(pk) >= 2:
                d = np.diff(pk).mean()
                if d > 0:
                    fr = 1.0 / d
                    if lo <= fr <= hi:
                        f[by, bx] = fr
    # smooth
    f = _sep_blur(f, 1.0)
    yy = np.clip(np.arange(H) // block, 0, nby - 1)
    xx = np.clip(np.arange(W) // block, 0, nbx - 1)
    return f[np.ix_(yy, xx)].astype(np.float32)


# ------------------------------------------------------------ enhancement --

def gabor(img, theta, freq, n_orient=16, ksize=11, sx=4.0, sy=4.0):
    """Orientation-steered Gabor bank.  `freq` may be a scalar or an HxW map;
    when a map, it is quantised to a small set of frequencies."""
    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    p = np.pad(img, r, mode="edge")

    if np.isscalar(freq):
        freqs = np.array([float(freq)], np.float32)
        fidx = np.zeros((H, W), np.int32)
    else:
        freqs = np.linspace(float(np.min(freq)), float(np.max(freq)), 3, dtype=np.float32)
        if freqs[-1] - freqs[0] < 1e-6:
            freqs = freqs[:1]
            fidx = np.zeros((H, W), np.int32)
        else:
            fidx = np.clip(np.round((freq - freqs[0]) /
                                    (freqs[-1] - freqs[0]) * (len(freqs) - 1)
                                    ).astype(np.int32), 0, len(freqs) - 1)

    angles = np.linspace(0, math.pi, n_orient, endpoint=False)
    out = np.zeros((H, W), np.float32)
    # quantise orientation
    oidx = np.clip(np.round(theta / math.pi * n_orient).astype(np.int32) % n_orient,
                   0, n_orient - 1)
    for fi, fv in enumerate(freqs):
        for ai, a in enumerate(angles):
            sel = (oidx == ai) & (fidx == fi)
            if not sel.any():
                continue
            # a is the RIDGE direction; the wave varies along the ridge normal
            ca, sa = math.cos(a), math.sin(a)
            xr = xx * ca + yy * sa          # along ridge
            yr = -xx * sa + yy * ca         # across ridge
            k = (np.exp(-(xr ** 2) / (2 * sx ** 2) - (yr ** 2) / (2 * sy ** 2)) *
                 np.cos(2 * math.pi * fv * yr))
            k = k - k.mean()
            acc = np.zeros((H, W), np.float32)
            for dy in range(ksize):
                for dx in range(ksize):
                    if k[dy, dx] != 0.0:
                        acc += k[dy, dx] * p[dy:dy + H, dx:dx + W]
            out[sel] = acc[sel]
    return out


# ------------------------------------------------------ binarise + thin ----

def binarise(enh, mask, k=7):
    """Ridge = above local mean.  `enh` is the Gabor response, ridges positive."""
    b = enh > _boxmean(enh, k)
    return b & mask


_ZS_LUT = None

def _zs_lut():
    """Precompute Zhang-Suen deletion decisions for all 256 neighbourhoods,
    for each of the two sub-iterations."""
    global _ZS_LUT
    if _ZS_LUT is not None:
        return _ZS_LUT
    # neighbour order P2..P9 = N, NE, E, SE, S, SW, W, NW
    lut = np.zeros((2, 256), bool)
    for code in range(256):
        P = [(code >> i) & 1 for i in range(8)]   # P[0]=P2 .. P[7]=P9
        B = sum(P)
        A = 0
        for i in range(8):
            if P[i] == 0 and P[(i + 1) % 8] == 1:
                A += 1
        P2, P3, P4, P5, P6, P7, P8, P9 = P
        if not (2 <= B <= 6 and A == 1):
            continue
        if (P2 * P4 * P6 == 0) and (P4 * P6 * P8 == 0):
            lut[0, code] = True
        if (P2 * P4 * P8 == 0) and (P2 * P6 * P8 == 0):
            lut[1, code] = True
    _ZS_LUT = lut
    return lut


_NB = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _nbcode(b):
    """8-bit neighbour code, bit i = neighbour _NB[i]."""
    p = np.pad(b.astype(np.uint8), 1)
    code = np.zeros(b.shape, np.uint8)
    for i, (dy, dx) in enumerate(_NB):
        code |= (p[1 + dy:1 + dy + b.shape[0], 1 + dx:1 + dx + b.shape[1]] << i)
    return code


def thin(b, max_iter=40):
    """Zhang-Suen thinning, LUT-driven, vectorised."""
    lut = _zs_lut()
    img = b.copy()
    for _ in range(max_iter):
        changed = False
        for sub in (0, 1):
            code = _nbcode(img)
            rm = img & lut[sub][code]
            if rm.any():
                img &= ~rm
                changed = True
        if not changed:
            break
    return img


# --------------------------------------------------------------- minutiae --

_CN_LUT = None
_NN_LUT = None

def _cn_lut():
    """CN(P) = 1/2 * sum_i |P_i - P_{i+1}| over the 8 neighbours in cyclic
    order.  1 = ridge ending, 2 = ordinary ridge pixel, 3 = bifurcation.
    Using the transition count rather than the raw neighbour count matters:
    Zhang-Suen leaves plenty of L-shaped corners with three neighbours that
    are NOT bifurcations."""
    global _CN_LUT, _NN_LUT
    if _CN_LUT is None:
        cn = np.zeros(256, np.int32)
        nn = np.zeros(256, np.int32)
        for code in range(256):
            P = [(code >> i) & 1 for i in range(8)]
            cn[code] = sum(abs(P[i] - P[(i + 1) % 8]) for i in range(8)) // 2
            nn[code] = sum(P)
        _CN_LUT, _NN_LUT = cn, nn
    return _CN_LUT, _NN_LUT


def _crossing_number(s):
    cn, _ = _cn_lut()
    return cn[_nbcode(s)]


def _nneigh(s):
    _, nn = _cn_lut()
    return nn[_nbcode(s)]


def prune_spurs(skel, spur_len=8, rounds=3, min_ridge=10):
    """Remove skeleton branches shorter than `spur_len` that hang off a
    branch point, and delete isolated fragments shorter than `min_ridge`.

    Branch points are cut out, the remaining pixels are grouped into
    8-connected segments, and each segment is classified by how many branch
    points it touches.
    """
    s = skel.copy()
    for _ in range(rounds):
        cn = _crossing_number(s)
        branch = s & (cn >= 3)
        seg = s & ~branch
        ys, xs = np.nonzero(seg)
        if len(ys) == 0:
            break
        idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
        parent = list(range(len(ys)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, (y, x) in enumerate(zip(ys, xs)):
            for dy, dx in _NB:
                j = idx.get((int(y) + dy, int(x) + dx))
                if j is not None:
                    union(i, j)

        comp = {}
        for i in range(len(ys)):
            comp.setdefault(find(i), []).append(i)

        remove = np.zeros(s.shape, bool)
        bpad = np.pad(branch, 1)
        for members in comp.values():
            n = len(members)
            # how many distinct branch pixels does this segment touch?
            touch = set()
            for i in members:
                y, x = int(ys[i]), int(xs[i])
                for dy, dx in _NB:
                    if bpad[y + 1 + dy, x + 1 + dx]:
                        touch.add((y + dy, x + dx))
            if len(touch) >= 2:
                continue                       # bridge between branches: keep
            if len(touch) == 1 and n <= spur_len:
                for i in members:
                    remove[int(ys[i]), int(xs[i])] = True
            elif len(touch) == 0 and n <= min_ridge:
                for i in members:
                    remove[int(ys[i]), int(xs[i])] = True
        if not remove.any():
            break
        s &= ~remove
        # a branch pixel left with <=2 neighbours is no longer a branch; also
        # drop branch pixels that became isolated
        cn2 = _crossing_number(s)
        s &= ~(s & (cn2 == 0))
    return s


def extract_minutiae(skel, theta, mask, ridge_period=9.0,
                     border=6, merge_dist=None, spur_len=8,
                     prune_rounds=3, min_ridge=10):
    """Crossing-number minutiae.  Returns (N,4) array: x, y, direction, type
    (type 1 = ending, 3 = bifurcation)."""
    if merge_dist is None:
        merge_dist = ridge_period

    s = prune_spurs(skel, spur_len=spur_len, rounds=prune_rounds,
                    min_ridge=min_ridge)

    cn = _crossing_number(s)

    # valid interior of the mask
    inner = _erode(mask, border)
    inner[:border, :] = False
    inner[-border:, :] = False
    inner[:, :border] = False
    inner[:, -border:] = False

    cand = []
    ys, xs = np.nonzero(s & inner & ((cn == 1) | (cn == 3)))
    for y, x in zip(ys, xs):
        typ = 1 if cn[y, x] == 1 else 3
        cand.append((float(x), float(y), typ))

    if not cand:
        return np.zeros((0, 4), np.float32)

    # direction: for endings, the direction the ridge leaves along; for
    # bifurcations, the local ridge orientation.  Use the traced ridge for
    # endings so the direction is a full 2*pi quantity.
    out = []
    for x, y, typ in cand:
        d = _trace_dir(s, int(x), int(y), typ, theta, steps=int(ridge_period))
        out.append((x, y, d, typ))
    m = np.array(out, np.float32)

    # merge / drop near pairs (same type within merge_dist -> drop both if
    # they are likely a broken ridge artefact; different types close together
    # -> drop both)
    keepm = np.ones(len(m), bool)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            dx, dy = m[i, 0] - m[j, 0], m[i, 1] - m[j, 1]
            if dx * dx + dy * dy < merge_dist * merge_dist:
                keepm[i] = keepm[j] = False
    m = m[keepm]
    return m


def _trace_dir(skel, x, y, typ, theta, steps=8):
    """Walk along the skeleton away from (x,y) for `steps` pixels and return
    the direction of travel in [0,2pi).  For bifurcations fall back to the
    orientation field (mod pi mapped into [0,pi))."""
    if typ == 3:
        return float(theta[y, x]) % math.pi
    cur = (y, x)
    prev = None
    py, px = y, x
    for _ in range(steps):
        nxt = None
        for dy, dx in _NB:
            ny, nx = cur[0] + dy, cur[1] + dx
            if not (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]):
                continue
            if not skel[ny, nx]:
                continue
            if prev is not None and (ny, nx) == prev:
                continue
            if (ny, nx) == cur:
                continue
            nxt = (ny, nx)
            break
        if nxt is None:
            break
        prev, cur = cur, nxt
    dy = cur[0] - py
    dx = cur[1] - px
    if dx == 0 and dy == 0:
        return float(theta[y, x]) % math.pi
    # direction FROM the endpoint INTO the ridge; minutia direction is the
    # opposite (pointing away), following the usual convention
    return float(math.atan2(-dy, -dx) % (2 * math.pi))


# ------------------------------------------------------------- full stage --

class Template:
    __slots__ = ("m", "mask", "n", "emask", "theta", "enh")

    def __init__(self, m, mask, theta=None, enh=None):
        self.m = m
        self.mask = mask
        self.emask = _erode(mask, 3)
        self.n = len(m)
        self.theta = theta
        self.enh = enh


def make_template(img, cfg=None):
    cfg = cfg or {}
    mask = segment(img, block=cfg.get("seg_block", 8),
                   thresh_frac=cfg.get("seg_frac", 0.30),
                   min_std=cfg.get("seg_min_std", 6.0))
    n = local_contrast_norm(img, sigma=cfg.get("lcn_sigma", 6.0))
    th, coh = orientation(n, block=cfg.get("or_block", 8),
                          smooth=cfg.get("or_smooth", 2.5))
    if cfg.get("est_freq", False):
        fr = ridge_frequency(n, th)
    else:
        fr = cfg.get("freq", 0.110)
    enh = gabor(n, th, fr, n_orient=cfg.get("n_orient", 16),
                ksize=cfg.get("ksize", 11),
                sx=cfg.get("gsx", 4.0), sy=cfg.get("gsy", 4.0))
    b = binarise(enh, mask, k=cfg.get("bin_k", 7))
    sk = thin(b)
    m = extract_minutiae(sk, th, mask,
                         ridge_period=1.0 / (cfg.get("freq", 0.110)),
                         border=cfg.get("border", 6),
                         merge_dist=cfg.get("merge_dist", None),
                         spur_len=cfg.get("spur_len", 8),
                         prune_rounds=cfg.get("prune_rounds", 3),
                         min_ridge=cfg.get("min_ridge", 10))
    return Template(m, mask, theta=th, enh=enh)


# --------------------------------------------------------------- matching --

def _angdiff(a, b):
    d = (a - b) % (2 * math.pi)
    return d - 2 * math.pi if d > math.pi else d


def match(ta, tb, rot_max=25.0, rot_bin=6.0, tr_bin=8.0,
          pos_tol=8.0, dir_tol=math.radians(20.0),
          top_k=6, min_denom=6.0, score="ov"):
    """Hough vote over (rotation, translation) then greedy pairing.

    Returns a similarity in [0,1]-ish.  Larger is better.
    """
    A, B = ta.m, tb.m
    if len(A) == 0 or len(B) == 0:
        return 0.0

    nrot = int(2 * rot_max / rot_bin) + 1
    rots = np.linspace(-rot_max, rot_max, nrot)

    ax, ay, ad = A[:, 0], A[:, 1], A[:, 2]
    bx, by, bd = B[:, 0], B[:, 1], B[:, 2]

    votes = {}
    for ri, rdeg in enumerate(rots):
        t = math.radians(rdeg)
        ct, st = math.cos(t), math.sin(t)
        rax = ct * ax - st * ay
        ray = st * ax + ct * ay
        # candidate translations for every compatible minutia pair
        for i in range(len(A)):
            # direction compatibility (mod pi, since bifurcation dirs are mod pi)
            di = (ad[i] + t)
            dd = (bd - di) % math.pi
            dd = np.minimum(dd, math.pi - dd)
            ok = dd <= dir_tol
            if not ok.any():
                continue
            tx = bx[ok] - rax[i]
            ty = by[ok] - ray[i]
            for X, Y in zip(tx, ty):
                if abs(X) > W or abs(Y) > H:
                    continue
                key = (ri, int(round(X / tr_bin)), int(round(Y / tr_bin)))
                votes[key] = votes.get(key, 0) + 1

    if not votes:
        return 0.0

    # take the top-k bins (plus their neighbours implicitly via refinement)
    best = sorted(votes.items(), key=lambda kv: -kv[1])[:top_k]

    best_score = 0.0
    for (ri, ix, iy), _v in best:
        t = math.radians(rots[ri])
        for sx in (-0.5, 0.0, 0.5):
            for sy in (-0.5, 0.0, 0.5):
                tx = (ix + sx) * tr_bin
                ty = (iy + sy) * tr_bin
                s = _score_alignment(ta, tb, t, tx, ty, pos_tol, dir_tol,
                                     min_denom, score)
                if s > best_score:
                    best_score = s
    return best_score


def _inside(mask, x, y):
    xi = np.round(x).astype(int)
    yi = np.round(y).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    res = np.zeros(len(x), bool)
    if ok.any():
        res[ok] = mask[yi[ok], xi[ok]]
    return res


def _score_alignment(ta, tb, t, tx, ty, pos_tol, dir_tol, min_denom, mode):
    A, B = ta.m, tb.m
    ct, st = math.cos(t), math.sin(t)
    axp = ct * A[:, 0] - st * A[:, 1] + tx
    ayp = st * A[:, 0] + ct * A[:, 1] + ty
    adp = A[:, 2] + t

    # overlap counts: A-minutiae landing in B's mask, and B-minutiae whose
    # inverse transform lands in A's mask.  Erode a little so minutiae right on
    # the mask edge (which the other capture could never see) don't count.
    mb = tb.emask
    ma = ta.emask
    inA = _inside(mb, axp, ayp)
    bxp = ct * (B[:, 0] - tx) + st * (B[:, 1] - ty)
    byp = -st * (B[:, 0] - tx) + ct * (B[:, 1] - ty)
    inB = _inside(ma, bxp, byp)

    nA = int(inA.sum())
    nB = int(inB.sum())
    if nA == 0 or nB == 0:
        return 0.0

    # greedy pairing among the overlapping subsets
    ia = np.nonzero(inA)[0]
    ib = np.nonzero(inB)[0]
    dx = axp[ia][:, None] - B[ib, 0][None, :]
    dy = ayp[ia][:, None] - B[ib, 1][None, :]
    d2 = dx * dx + dy * dy
    dd = (adp[ia][:, None] - B[ib, 2][None, :]) % math.pi
    dd = np.minimum(dd, math.pi - dd)
    cand = (d2 <= pos_tol * pos_tol) & (dd <= dir_tol)
    if not cand.any():
        return 0.0
    cost = np.where(cand, d2 + (dd / dir_tol) ** 2 * pos_tol * pos_tol, np.inf)
    used_a = np.zeros(len(ia), bool)
    used_b = np.zeros(len(ib), bool)
    order = np.argsort(cost, axis=None)
    m = 0
    for k in order:
        if not np.isfinite(cost.flat[k]):
            break
        i, j = divmod(int(k), cost.shape[1])
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = used_b[j] = True
        m += 1

    if mode == "count":
        return float(m)
    if mode == "ov":                       # geometric mean over overlap
        den = math.sqrt(max(nA, min_denom) * max(nB, min_denom))
        return m / den
    if mode == "ov2":                      # Bozorth-ish m^2 / (nA*nB)
        den = max(nA, min_denom) * max(nB, min_denom)
        return (m * m) / den
    if mode == "hyb":                      # ratio, damped by absolute count
        den = math.sqrt(max(nA, min_denom) * max(nB, min_denom))
        return (m / den) * (m / (m + 3.0))
    raise ValueError(mode)
