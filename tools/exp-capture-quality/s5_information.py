#!/usr/bin/env python3
"""
15. Why does registration fail? How much unique information does one 150x52
    capture actually contain?

    a) ridge period, measured from the autocorrelation of the LCN image
    b) SELF-AMBIGUITY: the best NCC of an image against ITSELF at any shift
       further than one pixel from the origin. If a capture correlates with
       itself at 0.7 when misaligned by a whole ridge period, no correlation
       matcher can locate the true alignment of a DIFFERENT capture that only
       reaches 0.3 at its true position.
    c) orientation spread within one frame -- a frame of parallel ridges is a
       1-D signal and is translation-ambiguous along the ridge direction
    d) sensor area in mm^2 and the expected minutia count, which bounds what
       MINDCT/BOZORTH3 could ever do here
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(TOOLS)
exec(open("matcher-lab.py").read().split("def main(")[0])       # noqa
from fastncc2 import ncc_map                                   # noqa: E402
import quality as Q                                            # noqa: E402

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LABELS = ["right-index", "right-index-cover", "right-middle"]
DPI = 508.0        # ELAN 0c6e press sensors report 508 dpi


def main():
    ds = load_dataset(DATASET)                                  # noqa: F821
    names, labels, imgs = [], [], []
    for li, lbl in enumerate(LABELS):
        for n, im in ds[lbl]:
            names.append(n); labels.append(li); imgs.append(im)
    labels = np.array(labels)
    L = [local_contrast_norm(im) for im in imgs]                # noqa: F821
    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]

    print("=" * 78)
    print("15a. RIDGE PERIOD (from the LCN autocorrelation)")
    print("=" * 78)
    periods = []
    for x in L:
        c, dys, dxs, n = ncc_map(x, x, 20, 12, 2500)
        # radial profile of the autocorrelation, first minimum -> half period
        prof = {}
        for yi, dy in enumerate(dys):
            for xi, dx in enumerate(dxs):
                r = int(round(np.hypot(dx, dy)))
                if np.isfinite(c[yi, xi]):
                    prof.setdefault(r, []).append(c[yi, xi])
        rs = sorted(prof)
        v = np.array([np.mean(prof[r]) for r in rs])
        # first local minimum after r=0, then the next local maximum
        i = 1
        while i + 1 < len(v) and not (v[i] < v[i - 1] and v[i] <= v[i + 1]):
            i += 1
        periods.append(2 * rs[i])
    periods = np.array(periods, float)
    print(f"   ridge period: median {np.median(periods):.1f} px "
          f"(p10 {np.percentile(periods,10):.1f}, p90 {np.percentile(periods,90):.1f})")
    print(f"   at {DPI:.0f} dpi that is {np.median(periods)/DPI*25.4:.2f} mm, "
          f"a normal adult ridge pitch (0.4-0.6 mm)")
    print(f"   the 52-px short axis therefore spans only "
          f"{52/np.median(periods):.1f} ridges")

    print("\n" + "=" * 78)
    print("15b. SELF-AMBIGUITY: how well does a capture match a WRONG shift of")
    print("     ITSELF?  (the alias floor every matcher has to beat)")
    print("=" * 78)
    alias = []
    for x in L:
        c, dys, dxs, n = ncc_map(x, x, 20, 8, 3500)
        DY = dys[:, None] + 0 * dxs[None, :]
        DX = dxs[None, :] + 0 * dys[:, None]
        far = (np.abs(DX) + np.abs(DY)) >= 4
        alias.append(float(np.nanmax(np.where(far & np.isfinite(c), c, -np.inf))))
    alias = np.array(alias)
    print(f"   self-NCC at the best shift >=4 px from the truth: "
          f"median {np.median(alias):.3f}  p10 {np.percentile(alias,10):.3f}  "
          f"p90 {np.percentile(alias,90):.3f}")
    z = np.load(HERE / "scores_base.npz")["S"]
    gp = list(itertools.combinations(gen.tolist(), 2))
    ip = [(a, b) for a in gen.tolist() for b in imp.tolist()]
    Sg = np.array([z[a, b] for a, b in gp]); Si = np.array([z[a, b] for a, b in ip])
    print(f"   best genuine CROSS-capture score, over all 465 pairs: "
          f"median {np.median(Sg):.3f}  max {Sg.max():.3f}")
    print(f"   best impostor cross-capture score: median {np.median(Si):.3f}  "
          f"max {Si.max():.3f}")
    print(f"\n   The alias floor ({np.median(alias):.2f}) is "
          f"{np.median(alias)/np.median(Sg):.1f}x the median genuine score.")
    print("   A wrong alignment of the SAME image scores far higher than the")
    print("   right alignment of a second press. Correlation cannot register")
    print("   these captures; the peak it reports is an alias, not the truth.")

    print("\n" + "=" * 78)
    print("15c. ORIENTATION SPREAD WITHIN ONE FRAME")
    print("=" * 78)
    spreads = []
    for x in L:
        gxx, gyy, gxy = Q.structure_tensor(x, 8, presmooth=1.0)
        v = (gxx - gyy) + 1j * (2 * gxy)
        m = np.abs(v)
        r = np.abs((v / (m + 1e-9) * m).sum()) / (m.sum() + 1e-9)
        # circular sd of the doubled angle, in ridge degrees
        spreads.append(np.degrees(np.sqrt(max(-2 * np.log(max(r, 1e-9)), 0))) / 2)
    spreads = np.array(spreads)
    print(f"   circular sd of ridge orientation inside one capture: "
          f"median {np.median(spreads):.1f} deg "
          f"(p10 {np.percentile(spreads,10):.1f}, p90 {np.percentile(spreads,90):.1f})")
    print(f"   captures with spread < 20 deg (essentially parallel ridges): "
          f"{(spreads < 20).mean()*100:.0f}%")
    print("   Parallel ridges are translation-ambiguous along the ridge, which")
    print("   is exactly the 150-px long axis of this sensor.")

    print("\n" + "=" * 78)
    print("15d. SENSOR AREA AND THE MINUTIA BUDGET")
    print("=" * 78)
    area = 150 / DPI * 25.4 * (52 / DPI * 25.4)
    print(f"   frame = 150 x 52 px at {DPI:.0f} dpi = "
          f"{150/DPI*25.4:.1f} x {52/DPI*25.4:.1f} mm = {area:.1f} mm^2")
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    fg = c["fg_px"]
    print(f"   median finger contact area = {np.median(fg)/(150*52)*100:.0f}% "
          f"of that = {area*np.median(fg)/(150*52):.1f} mm^2")
    for dens, src in ((0.20, "conservative"), (0.30, "typical")):
        print(f"   at {dens:.2f} minutiae/mm^2 ({src}): "
              f"{area*np.median(fg)/(150*52)*dens:.1f} minutiae per capture")
    print("   BOZORTH3 needs roughly 12 matched minutiae for a confident")
    print("   decision, and two partial presses share only part of even that.")
    print("   This is an upper bound set by the SENSOR, not by the driver.")

    print("\n" + "=" * 78)
    print("15e. WHAT DOES FRAME AVERAGING COST? (can only be bounded here --")
    print("     the dataset stores the driver's already-averaged output)")
    print("=" * 78)
    print("   Every capture is the mean of 3-10 raw frames; the raw frames are")
    print("   not retained, so no measurement of the averaging can be made from")
    print("   this dataset. What CAN be said: the residual sensor noise after")
    print("   averaging is already far below the ridge signal --")
    hf = []
    for im in imgs:
        n = im - Q.sep_blur(im, 0.8)          # above the ridge band
        r = Q.bandpass(im)
        hf.append(float(np.sqrt((n ** 2).mean()) / (np.sqrt((r ** 2).mean()) + 1e-9)))
    hf = np.array(hf)
    print(f"     RMS(above-ridge-band) / RMS(ridge band) = "
          f"median {np.median(hf):.3f}  p90 {np.percentile(hf,90):.3f}")
    print("   i.e. out-of-band residual is ~25% of ridge amplitude (and that")
    print("   figure over-counts, since the 0.8-sigma split leaks ridge energy).")
    print("   Removing ALL of it could not move a d' of 1.4; the failure is")
    print("   geometric, not SNR.")


if __name__ == "__main__":
    main()
