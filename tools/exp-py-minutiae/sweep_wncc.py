#!/usr/bin/env python3
"""Sweep the quality-weighted NCC, plus a validation that it degenerates to the
plain baseline when every weight is 1."""
import math, sys, time
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr
from eval import scenario_A, scenario_B
from fastncc import ncc_all_shifts
import quality as Q
import wncc

ORDER = ["right-index", "right-index-cover", "right-middle"]
W, H = 150, 52


def matrix(ds, rots, mdx, mdy, minw, gamma=1.0, ones=False, stat="rmax",
           energy_ref=0.45):
    qs, ims, labels = [], [], []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            q, n = Q.quality(im, energy_ref=energy_ref, gamma=gamma)
            if ones:
                q = np.ones_like(q)
            qs.append(q); ims.append(n); labels.append(lbl)
    P = [wncc.prep(ims[i], qs[i], rots, mdx, mdy) for i in range(len(ims))]
    n = len(P)
    S = np.zeros((n, n))
    Sw = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            s, wm = wncc.surface(P[i], P[j], rots, mdx, mdy, minw)
            fin = np.isfinite(s)
            if not fin.any():
                v = -1.0; vw = -1.0
            else:
                v = float(s[fin].max())
                # weight-aware: Fisher z scaled by effective independent samples
                neff = np.maximum(wm / 81.0, 4.0)
                z = np.arctanh(np.clip(s, -0.999, 0.999)) * np.sqrt(np.maximum(neff - 3, 1))
                vw = float(np.nanmax(np.where(fin, z, -np.inf)))
            S[i, j] = S[j, i] = v
            Sw[i, j] = Sw[j, i] = vw
    el = time.time() - t0
    np.fill_diagonal(S, np.nan); np.fill_diagonal(Sw, np.nan)
    return S, Sw, labels, el


def ev(S, labels, n_enroll=6, trials=16):
    ga, ia, _ = scenario_A(S, labels, n_enroll, trials)
    gb, ib = scenario_B(S, labels)
    return (dprime(ga, ia), eer(ga, ia)[0], far_at_frr(ga, ia)[0],
            dprime(gb, ib), eer(gb, ib)[0], far_at_frr(gb, ib)[0])


def line(tag, S, labels):
    a = ev(S, labels)
    print(f"{tag:34s} | {a[0]:6.2f} {a[1]*100:5.1f}% {a[2]*100:5.1f}% | "
          f"{a[3]:6.2f} {a[4]*100:5.1f}% {a[5]*100:5.1f}%")


def main():
    ds = load_ds()
    # ---- validation: all-ones weights must equal the plain NCC baseline
    rots = list(range(-12, 13, 4))
    S1, _, labels, _ = matrix(ds, rots, 20, 8, 3500, ones=True)
    Sb = np.load('S_baseline.npy')
    d = np.abs(S1 - Sb)
    print(f"validation vs cached baseline: max |diff| = {np.nanmax(d):.2e}  (want <1e-9)")

    print(f"\n{'config':34s} | {'A d':>6s} {'A EER':>6s} {'A F10':>6s} | "
          f"{'B d':>6s} {'B EER':>6s} {'B F10':>6s}")
    print("-" * 78)
    line("PLAIN ncc (baseline) dx20dy8", S1, labels)

    for (mdx, mdy, minw) in [(20, 8, 1500), (30, 10, 1200), (40, 14, 1000),
                             (60, 20, 800)]:
        for gamma in (1.0, 2.0):
            S, Sz, labels, el = matrix(ds, rots, mdx, mdy, minw, gamma=gamma)
            line(f"wncc g{gamma} dx{mdx} dy{mdy} w{minw} rmax", S, labels)
            line(f"wncc g{gamma} dx{mdx} dy{mdy} w{minw} zmax", Sz, labels)
            np.save(f"Sw_{mdx}_{mdy}_{minw}_{gamma}.npy", S)
            np.save(f"Swz_{mdx}_{mdy}_{minw}_{gamma}.npy", Sz)
        print(f"    ({el:.1f}s)")


if __name__ == "__main__":
    main()
