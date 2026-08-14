#!/usr/bin/env python3
"""
14. Look at the data. A capture-quality study that never inspects a capture is
    guessing. Renders montages (raw and LCN-enhanced, 3x nearest-neighbour) of
    the highest- and lowest-quality captures in each class to PNG.
"""
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(TOOLS)
exec(open("matcher-lab.py").read().split("def main(")[0])       # noqa
import quality as Q                                            # noqa: E402
from s1_measure import composite                               # noqa: E402

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LABELS = ["right-index", "right-index-cover", "right-middle"]


def write_png(path, arr):
    """8-bit greyscale PNG, no dependencies."""
    arr = np.asarray(arr, np.uint8)
    h, w = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def norm8(x):
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def upscale(a, k=3):
    return np.kron(a, np.ones((k, k), np.uint8))


def montage(tiles, cols=1, pad=4):
    th, tw = tiles[0].shape
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (th + pad) + pad, cols * (tw + pad) + pad), 40, np.uint8)
    for n, t in enumerate(tiles):
        r, cc = divmod(n, cols)
        y, x = pad + r * (th + pad), pad + cc * (tw + pad)
        out[y:y + th, x:x + tw] = t
    return out


def main():
    ds = load_dataset(DATASET)                                  # noqa: F821
    names, labels, imgs = [], [], []
    for li, lbl in enumerate(LABELS):
        for n, im in ds[lbl]:
            names.append(n); labels.append(li); imgs.append(im)
    labels = np.array(labels)

    c = np.load(HERE / "cache.npz", allow_pickle=True)
    qkeys = [str(k) for k in c["qkeys"]]
    q = composite(c["qual"], qkeys, ["ridge_band", "aniso_w", "coh_w", "usable_frac"])

    order = np.argsort(-q)
    pick = list(order[:4]) + list(order[-4:])
    print("rendering (top 4 and bottom 4 by composite quality):")
    for r, k in enumerate(pick):
        print(f"  {'TOP   ' if r < 4 else 'BOTTOM'} {LABELS[labels[k]]:<18} "
              f"{names[k]:<28} q={q[k]:.3f}")

    write_png(HERE / "montage_raw.png",
              montage([upscale(norm8(imgs[k])) for k in pick]))
    write_png(HERE / "montage_lcn.png",
              montage([upscale(norm8(Q.local_contrast_norm(imgs[k]))) for k in pick]))
    # everything, LCN, 2 columns -- the whole dataset at a glance
    allpick = np.concatenate([np.where(labels == li)[0] for li in range(3)])
    write_png(HERE / "montage_all.png",
              montage([upscale(norm8(Q.local_contrast_norm(imgs[k])), 2)
                       for k in allpick], cols=3))
    print("wrote montage_raw.png, montage_lcn.png, montage_all.png")


if __name__ == "__main__":
    main()
