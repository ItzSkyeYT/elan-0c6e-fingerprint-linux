#!/usr/bin/env python3
"""Generate preprocessed PGM variants of the dataset for MINDTCT."""
import sys, os
from pathlib import Path
import numpy as np

TOOLS = Path(__file__).resolve().parent.parent
exec(open(TOOLS / "matcher-lab.py").read().split("def main(")[0])

DS = Path("/home/melb/.local/share/elan-fp/dataset")
OUT = Path(__file__).resolve().parent / "prep"


def to_u8(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < 1e-6:
        hi = lo + 1
    v = (img - lo) / (hi - lo)
    return np.clip(v * 255.0, 0, 255).astype(np.uint8)


def hist_eq(img):
    f = img.ravel()
    order = np.argsort(f, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(f.size)
    return (rank.reshape(img.shape) * (255.0 / (f.size - 1))).astype(np.uint8)


def upscale(u8, f):
    if f == 1:
        return u8
    return np.repeat(np.repeat(u8, f, axis=0), f, axis=1)


def bilinear2x(u8):
    """2x bilinear upscale, portable to C trivially."""
    h, w = u8.shape
    a = u8.astype(np.float32)
    # build 2x grid sample coords
    ys = (np.arange(h * 2) + 0.5) / 2.0 - 0.5
    xs = (np.arange(w * 2) + 0.5) / 2.0 - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, h - 1); wy = (ys - y0)[:, None]
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, w - 1); wx = (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


def pad_reflect(u8, py, px):
    return np.pad(u8, ((py, py), (px, px)), mode="reflect")


VARIANTS = {}


def variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn
    return deco


@variant("raw")
def v_raw(img):
    return to_u8(img)


@variant("lcn")
def v_lcn(img):
    return to_u8(local_contrast_norm(img, sigma=6.0))


@variant("lcn_eq")
def v_lcn_eq(img):
    return hist_eq(local_contrast_norm(img, sigma=6.0))


@variant("bp_eq")
def v_bp_eq(img):
    return hist_eq(bandpass(img, 1.0, 6.0))


@variant("gabor")
def v_gabor(img):
    return to_u8(gabor_enhance(local_contrast_norm(img, sigma=6.0)))


@variant("gabor_eq")
def v_gabor_eq(img):
    return hist_eq(gabor_enhance(local_contrast_norm(img, sigma=6.0)))


def write_pgm(path, u8):
    h, w = u8.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write(u8.tobytes())


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    mode = sys.argv[2] if len(sys.argv) > 2 else "nn"      # nn | bilin
    pad = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # pixels of reflect pad (pre-scale)
    tag = "s%d%s%s" % (scale, mode, ("p%d" % pad) if pad else "")
    for label in ("right-index", "right-index-cover", "right-middle"):
        for f in sorted((DS / label).glob("*.pgm")):
            img = load_pgm(f)
            for vname, vfn in VARIANTS.items():
                u8 = vfn(img)
                if pad:
                    u8 = pad_reflect(u8, pad, pad)
                if scale == 2 and mode == "bilin":
                    u8 = bilinear2x(u8)
                elif scale > 1:
                    u8 = upscale(u8, scale)
                d = OUT / tag / vname / label
                d.mkdir(parents=True, exist_ok=True)
                write_pgm(d / f.name, u8)
    print("wrote", OUT / tag, "variants:", list(VARIANTS))


if __name__ == "__main__":
    main()
