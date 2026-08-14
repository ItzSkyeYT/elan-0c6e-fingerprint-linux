#!/usr/bin/env python3
"""PIPELINE SANITY CHECK. Build synthetic 'second presses' by translating a real
capture by a known offset (filling the vacated strip with mid-grey, so the
overlap region is pixel-identical). If MINDTCT + my coordinate handling are
correct, the oracle must recover a large number of shared minutiae here, and
bozorth3 should score high. If this fails, the negative result is a bug.
"""
import subprocess, sys, math
from pathlib import Path
import numpy as np
import evalproto as ep
import diag_oracle as do

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
exec(open(TOOLS / "matcher-lab.py").read().split("def main(")[0])
sys.path.insert(0, str(HERE))
from prep import hist_eq, bilinear2x, write_pgm


def make(img, dx, dy):
    out = np.full_like(img, np.median(img))
    h, w = img.shape
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    out[ys:ye, xs:xe] = img[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def main():
    ds = ep.manifest()
    gi = [p for l, p in ds if l == "right-index"]
    work = HERE / "synth"
    for sub in ("a", "b"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    shifts = [(0, 0), (5, 2), (10, 3), (15, 5), (25, 8), (40, 10)]
    print("shift        bozorth3(A,B)   oracle_shared   nA  nB")
    for dx, dy in shifts:
        paths = []
        for k, p in enumerate(gi):
            img = load_pgm(p)
            for sub, im in (("a", img), ("b", make(img, dx, dy))):
                u8 = bilinear2x(hist_eq(gabor_enhance(local_contrast_norm(im, sigma=6.0))))
                fp = work / sub / f"{k:03d}.pgm"
                write_pgm(fp, u8)
                paths.append(fp)
        # interleaved manifest: a0,b0,a1,b1,...
        order = []
        for k in range(len(gi)):
            order.append(work / "a" / f"{k:03d}.pgm")
            order.append(work / "b" / f"{k:03d}.pgm")
        mf = HERE / "manifest_synth.txt"
        mf.write_text("\n".join(str(p) for p in order) + "\n")
        xd = HERE / "xyt_synth"; xd.mkdir(exist_ok=True)
        out = subprocess.run([str(HERE / "nbisbatch"), str(mf), "19.685", "0", str(xd)],
                             capture_output=True, text=True, check=True).stdout
        n = len(order)
        S = np.zeros((n, n)); cnt = np.zeros(n, int)
        for line in out.splitlines():
            f = line.split()
            if f[0] == "M":
                cnt[int(f[1])] = int(f[2])
            elif f[0] == "S":
                S[int(f[1]), int(f[2])] = int(f[3])
        M = do.load_xyt(xd, n, 2, 104)
        bz = [S[2 * k, 2 * k + 1] for k in range(len(gi))]
        sh = [do.oracle_shared(M[2 * k], M[2 * k + 1], 8.0) for k in range(len(gi))]
        print("dx=%3d dy=%3d   mean %6.1f max %4d   mean %5.2f max %2d   %4.1f %4.1f"
              % (dx, dy, np.mean(bz), max(bz), np.mean(sh), max(sh),
                 cnt[0::2].mean(), cnt[1::2].mean()))


if __name__ == "__main__":
    main()
