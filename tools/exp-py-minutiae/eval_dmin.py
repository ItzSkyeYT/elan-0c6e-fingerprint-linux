#!/usr/bin/env python3
"""Measure the quality-gated, descriptor-augmented minutiae matcher, and
re-run the ceiling analysis in a COUNT-NORMALISED form.

The earlier ceiling analysis compared raw matched-minutiae counts, which is
unfair to the genuine finger: the impostor captures yield 20.8 minutiae per
image against 14.0, and more points means more coincidental agreements.  Here
the ceiling is measured as matched / sqrt(nA*nB) as well, which removes that
confound, so a negative result cannot be blamed on it.
"""
import math, sys, time, itertools
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr
from eval import scenario_A, scenario_B
import dmin

ORDER = ["right-index", "right-index-cover", "right-middle"]
W, H = 150, 52


def build(ds, cfg):
    T, labels = [], []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            T.append(dmin.make_template(im, cfg))
            labels.append(lbl)
    return T, labels


def matrix(T, mcfg):
    n = len(T)
    S = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = dmin.match(T[i], T[j], **mcfg)
    el = time.time() - t0
    np.fill_diagonal(S, np.nan)
    return S, el


def ev(S, labels, n_enroll=6, trials=16):
    ga, ia, _ = scenario_A(S, labels, n_enroll, trials)
    gb, ib = scenario_B(S, labels)
    return (dprime(ga, ia), eer(ga, ia)[0], far_at_frr(ga, ia)[0],
            dprime(gb, ib), eer(gb, ib)[0], far_at_frr(gb, ib)[0])


def line(tag, S, labels):
    a = ev(S, labels)
    print(f"{tag:40s} | {a[0]:6.2f} {a[1]*100:5.1f}% {a[2]*100:5.1f}% | "
          f"{a[3]:6.2f} {a[4]*100:5.1f}% {a[5]*100:5.1f}%")


def main():
    ds = load_ds()
    print(f"{'config':40s} | {'A d':>6s} {'A EER':>6s} {'A F10':>6s} | "
          f"{'B d':>6s} {'B EER':>6s} {'B F10':>6s}")
    print("-" * 84)
    cfgs = [
        ("nomerge, no q-gate",   dict(merge_dist=0.0, min_q=0.0, qthr=0.0, border=3,
                                      spur_len=4, min_ridge=6)),
        ("nomerge, q-gate .35",  dict(merge_dist=0.0, min_q=0.35, qthr=0.25, border=3,
                                      spur_len=4, min_ridge=6)),
        ("nomerge, q-gate .45",  dict(merge_dist=0.0, min_q=0.45, qthr=0.30, border=3,
                                      spur_len=4, min_ridge=6)),
        ("nomerge, q .35, cap12", dict(merge_dist=0.0, min_q=0.35, qthr=0.25, border=3,
                                       spur_len=4, min_ridge=6, cap=12)),
        ("merge 1 period, no gate", dict(merge_dist=None, min_q=0.0, qthr=0.0, border=3,
                                         spur_len=4, min_ridge=6)),
    ]
    for tag, cfg in cfgs:
        T, labels = build(ds, cfg)
        cnts = {l: np.mean([t.n for t, ll in zip(T, labels) if ll == l]) for l in ORDER}
        print(f"  [{tag}] minutiae/image: " +
              "  ".join(f"{k.split('-',1)[1]} {v:.1f}" for k, v in cnts.items()) +
              f"   overall {np.mean([t.n for t in T]):.1f}")
        for use_desc, dthr in ((False, -1.0), (True, 0.40), (True, 0.60), (True, 0.75)):
            mcfg = dict(rot_max=20.0, rot_bin=5.0, tr_bin=8.0, pos_tol=9.0,
                        dir_tol=math.radians(25.0), desc_thr=dthr,
                        top_k=6, min_denom=6.0, use_desc=use_desc)
            S, el = matrix(T, mcfg)
            np.save(f"Sd_{tag.replace(' ','_').replace(',','')}_{use_desc}_{dthr}.npy", S)
            line(f"  desc={use_desc} thr={dthr:.2f} ({el*1000/990:.0f}ms/pair)", S, labels)


if __name__ == "__main__":
    main()
