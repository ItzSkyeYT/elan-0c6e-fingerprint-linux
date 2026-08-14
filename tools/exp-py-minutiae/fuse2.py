#!/usr/bin/env python3
"""Score-level fusion across matchers with different search windows.

CAVEAT, stated up front: the fusion weight is chosen by looking at the same 45
captures that are then reported, so any gain here is optimistic.  It is shown
to establish whether the matchers carry COMPLEMENTARY information at all, not
as a deployable number.
"""
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr
from fuse import labels_of, scenario_A, scenario_B


def zs(S):
    v = S[np.isfinite(S)]
    return (S - v.mean()) / (v.std() + 1e-12)


def ev(S, labels):
    ga, ia = scenario_A(S, labels, "max")
    gb, ib = scenario_B(S, labels, "max")
    return (dprime(ga, ia), eer(ga, ia)[0], far_at_frr(ga, ia)[0],
            dprime(gb, ib), eer(gb, ib)[0], far_at_frr(gb, ib)[0])


def main():
    labels = labels_of(load_ds())
    names = ["P_dx20", "P_dx40", "P_dx60", "Pg_dx20", "S_minutiae"]
    M = {n: zs(np.load(n + ".npy")) for n in names}
    print(f"{'fusion':40s} | {'A d':>6s} {'A EER':>6s} {'A F10':>6s} | "
          f"{'B d':>6s} {'B EER':>6s} {'B F10':>6s}")
    print("-" * 84)
    for n in names:
        a = ev(M[n], labels)
        print(f"{n:40s} | {a[0]:6.2f} {a[1]*100:5.1f}% {a[2]*100:5.1f}% | "
              f"{a[3]:6.2f} {a[4]*100:5.1f}% {a[5]*100:5.1f}%")
    print()
    base = "P_dx20"
    for other in ["P_dx40", "P_dx60", "Pg_dx20", "S_minutiae"]:
        for w in (0.25, 0.5, 0.75, 1.0):
            S = M[base] + w * M[other]
            a = ev(S, labels)
            print(f"{base}+{w:.2f}*{other:22s} | {a[0]:6.2f} {a[1]*100:5.1f}% "
                  f"{a[2]*100:5.1f}% | {a[3]:6.2f} {a[4]*100:5.1f}% {a[5]*100:5.1f}%")
        print()
    S = M["P_dx20"] + 0.5 * M["P_dx60"] + 0.3 * M["S_minutiae"]
    a = ev(S, labels)
    print(f"{'dx20 + .5*dx60 + .3*minutiae':40s} | {a[0]:6.2f} {a[1]*100:5.1f}% "
          f"{a[2]*100:5.1f}% | {a[3]:6.2f} {a[4]*100:5.1f}% {a[5]*100:5.1f}%")


if __name__ == "__main__":
    main()
