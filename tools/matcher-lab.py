#!/usr/bin/env python3
"""
Offline matcher evaluation for the ELAN 04f3:0c6e press sensor.

Loads a labelled dataset of captures (see `elan-fp collect`), scores every
genuine and impostor pair with each registered matcher, and reports separation.
Iterating here takes seconds; iterating against the sensor takes twenty finger
presses per data point.

    ./matcher-lab.py [--dataset DIR] [--genuine LABEL] [--only NAME]

Reference measurements from the shipped elanpress driver on real hardware:
    genuine  mean 0.469  sd 0.090      impostor mean 0.452  sd 0.190
    d' = 0.12, and the impostor maximum exceeded the genuine maximum.
`ncc_raw` below reimplements that matcher, so it doubles as a check that this
harness reproduces reality before any conclusion is drawn from the others.

numpy is used for speed. The winning matcher gets ported to plain C afterwards,
so nothing here may depend on numpy semantics that are awkward to reproduce
(no FFT-only tricks unless we accept hand-writing an FFT, which is fine).
"""

import argparse
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np

W, H = 150, 52


# ----------------------------------------------------------------- loading --

def load_pgm(path):
    """Minimal binary PGM (P5) reader. Returns float32 HxW."""
    data = Path(path).read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"{path}: not a P5 PGM")
    fields, idx = [], 2
    while len(fields) < 3:
        while idx < len(data) and data[idx:idx + 1].isspace():
            idx += 1
        if data[idx:idx + 1] == b"#":
            while idx < len(data) and data[idx] != 0x0A:
                idx += 1
            continue
        start = idx
        while idx < len(data) and not data[idx:idx + 1].isspace():
            idx += 1
        fields.append(int(data[start:idx]))
    idx += 1
    w, h, _maxval = fields
    px = np.frombuffer(data[idx:idx + w * h], dtype=np.uint8)
    if px.size != w * h:
        raise ValueError(f"{path}: truncated ({px.size} of {w*h} px)")
    return px.reshape(h, w).astype(np.float32)


def load_dataset(root):
    ds = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        imgs = []
        for f in sorted(d.glob("*.pgm")):
            try:
                imgs.append((f.name, load_pgm(f)))
            except Exception as e:            # a truncated dump shouldn't kill the run
                print(f"  skipping {f.name}: {e}", file=sys.stderr)
        if imgs:
            ds[d.name] = imgs
    return ds


# ----------------------------------------------------------------- metrics --

def dprime(gen, imp):
    g, i = np.asarray(gen), np.asarray(imp)
    denom = math.sqrt((g.var() + i.var()) / 2)
    return float((g.mean() - i.mean()) / denom) if denom > 0 else 0.0


def eer(gen, imp):
    """Equal error rate: the threshold where FRR and FAR are closest, and the
    mean of the two there."""
    g, i = np.asarray(gen), np.asarray(imp)
    best_gap, best_eer, best_t = math.inf, 1.0, 0.0
    for t in np.unique(np.concatenate([g, i])):
        frr = float((g < t).mean())          # genuine rejected
        far = float((i >= t).mean())         # impostor accepted
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap, best_eer, best_t = gap, (frr + far) / 2, float(t)
    return best_eer, best_t


def far_at_frr(gen, imp, target_frr=0.10):
    """False accept rate when the threshold is set to reject `target_frr` of genuine."""
    g, i = np.asarray(gen), np.asarray(imp)
    t = float(np.quantile(g, target_frr))
    return float((i >= t).mean()), t


def report(name, gen, imp, verbose=True):
    g, i = np.asarray(gen), np.asarray(imp)
    d = dprime(g, i)
    e, et = eer(g, i)
    far10, t10 = far_at_frr(g, i, 0.10)
    if verbose:
        print(f"\n=== {name} ===")
        print(f"  genuine   n={g.size:4d}  mean={g.mean():.3f}  sd={g.std():.3f}  "
              f"min={g.min():.3f}  max={g.max():.3f}")
        print(f"  impostor  n={i.size:4d}  mean={i.mean():.3f}  sd={i.std():.3f}  "
              f"min={i.min():.3f}  max={i.max():.3f}")
        print(f"  d-prime   {d:6.2f}      EER {e*100:5.1f}%  (@ {et:.3f})")
        print(f"  FAR at 10% FRR: {far10*100:5.1f}%  (threshold {t10:.3f})")
        overlap = float((i >= g.min()).mean())
        print(f"  impostors scoring above the WORST genuine: {overlap*100:.0f}%")
        verdict = ("UNUSABLE"  if d < 1 else
                   "weak"      if d < 2 else
                   "promising" if d < 3 else
                   "GOOD")
        print(f"  -> {verdict}")
    return {"name": name, "dprime": d, "eer": e, "far10": far10,
            "gen": g, "imp": i}


# --------------------------------------------------------------- filtering --

def _gauss_kernel(sigma):
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _sep_blur(img, sigma):
    """Separable Gaussian blur with edge replication."""
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="edge")
    out = np.empty_like(img)
    for c in range(img.shape[1]):
        out[:, c] = (p[:, c:c + len(k)] * k).sum(axis=1)
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    res = np.empty_like(img)
    for rr in range(img.shape[0]):
        res[rr, :] = (p[rr:rr + len(k), :] * k[:, None]).sum(axis=0)
    return res


def normalize_global(img):
    s = img.std()
    return (img - img.mean()) / s if s > 1e-6 else img - img.mean()


def bandpass(img, lo=1.0, hi=4.0):
    """Difference of Gaussians: strips slow illumination/pressure gradients and
    high-frequency sensor noise, leaving the ridge band."""
    return _sep_blur(img, lo) - _sep_blur(img, hi)


def local_contrast_norm(img, sigma=6.0, eps=1e-3):
    """Divide out the local energy, so a heavy press and a light press of the
    same ridges produce comparable images. This is the cheapest fix for the
    failure mode raw NCC exhibits."""
    m = _sep_blur(img, sigma)
    d = img - m
    v = np.sqrt(np.maximum(_sep_blur(d * d, sigma), 0.0))
    return d / (v + eps * v.mean() + 1e-6)


def orientation_field(img, block=8, smooth=2.0):
    """Gradient-based ridge orientation (Hong-Wan-Jain). Doubled angles are
    averaged so that orientations 180 degrees apart reinforce rather than cancel."""
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    vx = 2 * gx * gy
    vy = gx ** 2 - gy ** 2
    vx = _sep_blur(vx, smooth)
    vy = _sep_blur(vy, smooth)
    return 0.5 * np.arctan2(vx, vy)          # ridge-normal angle


def gabor_enhance(img, freq=0.11, n_orient=8, ksize=9, sigma=2.5):
    """Filter each pixel with the Gabor kernel whose orientation matches the
    local ridge flow. Ridge structure survives; pressure and contact-area
    variation largely does not -- which is the whole point."""
    img = local_contrast_norm(img)
    theta = orientation_field(img)

    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)

    angles = np.linspace(0, math.pi, n_orient, endpoint=False)
    responses = np.empty((n_orient,) + img.shape, dtype=np.float32)
    p = np.pad(img, r, mode="edge")
    for ai, a in enumerate(angles):
        # ridge direction is theta + pi/2; filter along the ridge normal
        xr = xx * math.cos(a) + yy * math.sin(a)
        yr = -xx * math.sin(a) + yy * math.cos(a)
        k = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * math.pi * freq * xr)
        k -= k.mean()
        acc = np.zeros_like(img)
        for dy in range(ksize):
            for dx in range(ksize):
                acc += k[dy, dx] * p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
        responses[ai] = acc

    # pick, per pixel, the response for the nearest quantised orientation
    ridge_dir = (theta + math.pi / 2) % math.pi
    idx = np.clip((ridge_dir / math.pi * n_orient).astype(np.int32), 0, n_orient - 1)
    out = np.take_along_axis(responses, idx[None, ...], axis=0)[0]
    return normalize_global(out)


# ---------------------------------------------------------------- matchers --

def _ncc_shift(a, b, dx, dy, min_overlap):
    x0, x1 = max(0, dx), min(W, W + dx)
    y0, y1 = max(0, dy), min(H, H + dy)
    if (x1 - x0) * (y1 - y0) < min_overlap:
        return -1.0
    pa = a[y0:y1, x0:x1]
    pb = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    pa = pa - pa.mean()
    pb = pb - pb.mean()
    denom = math.sqrt(float((pa * pa).sum()) * float((pb * pb).sum()))
    return float((pa * pb).sum() / denom) if denom > 1e-9 else -1.0


def _rotate(img, deg):
    """Rotate about the centre with bilinear sampling and edge clamping."""
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


def ncc_best(a, b, max_dx=15, max_dy=6, min_overlap=5000,
             max_rot=0, rot_step=4, step=1):
    best = -1.0
    rots = [0] if max_rot == 0 else range(-max_rot, max_rot + 1, rot_step)
    for deg in rots:
        br = _rotate(b, deg) if deg else b
        for dy in range(-max_dy, max_dy + 1, step):
            for dx in range(-max_dx, max_dx + 1, step):
                c = _ncc_shift(a, br, dx, dy, min_overlap)
                if c > best:
                    best = c
    return best


def m_ncc_raw(a, b):
    """Baseline: exactly what the shipped elanpress driver does."""
    return ncc_best(a, b, 15, 6, 5000)


def m_ncc_raw_wide(a, b):
    """The original upstream window, for comparison."""
    return ncc_best(a, b, 60, 20, 1500)


def m_ncc_bandpass(a, b):
    return ncc_best(bandpass(a), bandpass(b), 15, 6, 5000)


def m_ncc_lcn(a, b):
    return ncc_best(local_contrast_norm(a), local_contrast_norm(b), 15, 6, 5000)


def m_ncc_gabor(a, b):
    return ncc_best(gabor_enhance(a), gabor_enhance(b), 15, 6, 5000)


def m_ncc_gabor_rot(a, b):
    """Gabor enhancement plus a rotation search."""
    return ncc_best(gabor_enhance(a), gabor_enhance(b),
                    20, 8, 3500, max_rot=12, rot_step=4)


def m_ncc_gabor_wide(a, b):
    """Gabor plus a wide translation search, but a genuinely large minimum
    overlap so tiny-overlap flukes cannot win."""
    return ncc_best(gabor_enhance(a), gabor_enhance(b),
                    45, 15, 3500, max_rot=12, rot_step=6, step=2)


def m_ncc_lcn_rot(a, b):
    return ncc_best(local_contrast_norm(a), local_contrast_norm(b),
                    20, 8, 3500, max_rot=12, rot_step=4)


MATCHERS = {
    "ncc_raw":       (m_ncc_raw,       "baseline: raw pixels, tightened window (shipped)"),
    "ncc_raw_wide":  (m_ncc_raw_wide,  "raw pixels, original wide window"),
    "ncc_bandpass":  (m_ncc_bandpass,  "difference-of-Gaussians, then NCC"),
    "ncc_lcn":       (m_ncc_lcn,       "local contrast normalisation, then NCC"),
    "ncc_gabor":     (m_ncc_gabor,     "orientation-steered Gabor enhancement, then NCC"),
    "ncc_lcn_rot":   (m_ncc_lcn_rot,   "local contrast norm + rotation search"),
    "ncc_gabor_rot": (m_ncc_gabor_rot, "Gabor + rotation search"),
    "ncc_gabor_wide":(m_ncc_gabor_wide,"Gabor + wide translation + rotation"),
}


# -------------------------------------------------------------- evaluation --

def evaluate(fn, ds, genuine_label, cache=None):
    """Pairwise: all within-label pairs genuine, all cross-label pairs impostor.
    Self-pairs excluded -- comparing an image with itself is meaningless and
    inflates the genuine distribution."""
    gen_imgs = [im for _, im in ds[genuine_label]]
    imp_imgs = [im for lbl, lst in ds.items() if lbl != genuine_label
                for _, im in lst]

    gen = [fn(a, b) for a, b in itertools.combinations(gen_imgs, 2)]
    imp = [fn(a, b) for a in gen_imgs for b in imp_imgs]
    return gen, imp


def evaluate_template(fn, ds, genuine_label, n_enroll=6):
    """Model what the driver actually does: enrol a SET of images, and score a
    probe as the maximum similarity against any of them. Leave-one-out over the
    genuine set, so no probe is ever compared against itself.

    Pairwise scoring understates real performance, because a probe only has to
    resemble ONE enrolled template rather than every other capture."""
    gen_imgs = [im for _, im in ds[genuine_label]]
    imp_imgs = [im for lbl, lst in ds.items() if lbl != genuine_label
                for _, im in lst]
    if len(gen_imgs) <= n_enroll:
        n_enroll = max(1, len(gen_imgs) - 1)

    gen, imp = [], []
    for probe_i in range(len(gen_imgs)):
        template = [gen_imgs[j] for j in range(len(gen_imgs)) if j != probe_i][:n_enroll]
        gen.append(max(fn(t, gen_imgs[probe_i]) for t in template))
    template = gen_imgs[:n_enroll]
    for b in imp_imgs:
        imp.append(max(fn(t, b) for t in template))
    return gen, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.expanduser(
        "~/.local/share/elan-fp/dataset"))
    ap.add_argument("--genuine", default=None,
                    help="label treated as genuine (default: first)")
    ap.add_argument("--only", default=None, help="run just this matcher")
    ap.add_argument("--template", action="store_true",
                    help="evaluate as template-set vs probe (how the driver works)")
    ap.add_argument("--n-enroll", type=int, default=6)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    if len(ds) < 2:
        print(f"need at least two labels in {args.dataset}; found {list(ds)}")
        return 1

    genuine = args.genuine or sorted(ds)[0]
    print(f"dataset: {args.dataset}")
    for lbl, imgs in ds.items():
        mark = "  (genuine)" if lbl == genuine else "  (impostor)"
        print(f"  {lbl:<16} {len(imgs):3d} captures{mark}")
    ng = len(ds[genuine])
    ni = sum(len(v) for k, v in ds.items() if k != genuine)
    print(f"\n  {ng*(ng-1)//2} genuine pairs, {ng*ni} impostor pairs")

    names = [args.only] if args.only else list(MATCHERS)
    results = []
    for name in names:
        fn, desc = MATCHERS[name]
        print(f"\nrunning {name} ({desc}) ...", end="", flush=True)
        if args.template:
            gen, imp = evaluate_template(fn, ds, genuine, args.n_enroll)
        else:
            gen, imp = evaluate(fn, ds, genuine)
        print(" done")
        results.append(report(name, gen, imp))

    if len(results) > 1:
        print("\n" + "=" * 62)
        print(f"  {'matcher':<16} {'d-prime':>9} {'EER':>8} {'FAR@10%FRR':>12}")
        print("  " + "-" * 58)
        for r in sorted(results, key=lambda r: -r["dprime"]):
            print(f"  {r['name']:<16} {r['dprime']:9.2f} "
                  f"{r['eer']*100:7.1f}% {r['far10']*100:11.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
