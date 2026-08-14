#!/usr/bin/env python3
"""Cleanest possible stability test: crop two overlapping sub-windows out of the
SAME capture. Both windows contain only real sensor content and the overlap is
pixel-identical -- no synthetic fill, no resampling difference. If MINDTCT's
minutiae are stable under a pure shift of the analysis window, the two crops
must yield the same minutiae in the shared area.
"""
import subprocess, sys
from pathlib import Path
import numpy as np
import evalproto as ep
import diag_oracle as do

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
exec(open(TOOLS / "matcher-lab.py").read().split("def main(")[0])
sys.path.insert(0, str(HERE))
from prep import hist_eq, bilinear2x, write_pgm

CW, CH = 118, 44          # crop size


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    enh = sys.argv[2] if len(sys.argv) > 2 else "gabor_eq"
    ds = ep.manifest()
    gi = [p for l, p in ds if l == "right-index"]
    work = HERE / "crop"
    work.mkdir(exist_ok=True)

    def prep(im):
        if enh == "gabor_eq":
            u8 = hist_eq(gabor_enhance(local_contrast_norm(im, sigma=6.0)))
        elif enh == "lcn_eq":
            u8 = hist_eq(local_contrast_norm(im, sigma=6.0))
        else:
            u8 = np.clip(im, 0, 255).astype(np.uint8)
        return bilinear2x(u8) if scale == 2 else u8

    print(f"crop {CW}x{CH} out of 150x52, scale={scale} enh={enh}")
    print("offset        bozorth3      oracle_shared(tol8)   nA   nB")
    for dx, dy in [(0, 0), (4, 0), (8, 2), (16, 4), (24, 6), (30, 8)]:
        order = []
        for k, p in enumerate(gi):
            img = load_pgm(p)
            a = img[0:CH, 0:CW]
            b = img[dy:dy + CH, dx:dx + CW]
            for tag, im in (("a", a), ("b", b)):
                fp = work / f"{k:03d}{tag}.pgm"
                write_pgm(fp, prep(im))
                order.append(fp)
        mf = HERE / "manifest_crop.txt"
        mf.write_text("\n".join(str(p) for p in order) + "\n")
        xd = HERE / "xyt_crop"; xd.mkdir(exist_ok=True)
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
        M = do.load_xyt(xd, n, scale, CH * scale)
        bz = [S[2 * k, 2 * k + 1] for k in range(len(gi))]
        sh = [do.oracle_shared(M[2 * k], M[2 * k + 1], 8.0, max_dx=60, max_dy=25)
              for k in range(len(gi))]
        print("dx=%3d dy=%2d   mean %6.1f max %4d   mean %5.2f max %2d   %4.1f %4.1f"
              % (dx, dy, np.mean(bz), max(bz), np.mean(sh), max(sh),
                 cnt[0::2].mean(), cnt[1::2].mean()))


if __name__ == "__main__":
    main()
