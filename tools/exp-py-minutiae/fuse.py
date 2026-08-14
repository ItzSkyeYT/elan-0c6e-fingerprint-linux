#!/usr/bin/env python3
"""Two orthogonal levers that cost nothing at match time:

  1. TEMPLATE FUSION.  The driver currently scores a probe as the MAXIMUM over
     the enrolled templates.  With a small window and varying placement an
     impostor only needs one lucky template, while a genuine probe usually
     resembles SEVERAL enrolled captures.  Mean-of-top-k turns that asymmetry
     into signal.

  2. SCORE FUSION across matchers with different failure modes.

Both are evaluated with the same scenario A / B harness, on cached score
matrices, so nothing here can leak: the matrices were built without any
knowledge of the split.
"""
import math, sys, glob, os
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr

GEN = ["right-index", "right-index-cover"]
IMP = "right-middle"
ORDER = ["right-index", "right-index-cover", IMP]


def labels_of(ds):
    return [l for l in ORDER for _ in ds[l]]


def combine(vals, mode):
    v = np.sort(np.asarray(vals, float))[::-1]
    if mode == "max":
        return v[0]
    if mode.startswith("top"):
        k = int(mode[3:])
        k = min(k, len(v))
        return float(v[:k].mean())
    if mode == "mean":
        return float(v.mean())
    if mode.startswith("sm"):            # soft-max / log-sum-exp with temp
        t = float(mode[2:])
        return float(t * np.log(np.exp(v / t).sum()))
    raise ValueError(mode)


def scenario_A(S, labels, mode, n_enroll=6, trials=16, seed=0):
    gi = [k for k, l in enumerate(labels) if l in GEN]
    ii = [k for k, l in enumerate(labels) if l == IMP]
    rng = np.random.default_rng(seed)
    g_all, i_all = [], []
    for _ in range(trials):
        T = list(rng.choice(gi, size=n_enroll, replace=False))
        probes = [k for k in gi if k not in T]
        g_all += [combine([S[t, p] for t in T], mode) for p in probes]
        i_all += [combine([S[t, p] for t in T], mode) for p in ii]
    return g_all, i_all


def scenario_B(S, labels, mode):
    enrol = [k for k, l in enumerate(labels) if l == "right-index-cover"]
    probe = [k for k, l in enumerate(labels) if l == "right-index"]
    imp = [k for k, l in enumerate(labels) if l == IMP]
    g = [combine([S[t, p] for t in enrol], mode) for p in probe]
    m = [combine([S[t, p] for t in enrol], mode) for p in imp]
    return g, m


def row(tag, S, labels, mode):
    ga, ia = scenario_A(S, labels, mode)
    gb, ib = scenario_B(S, labels, mode)
    print(f"{tag:34s} {mode:6s} | {dprime(ga,ia):6.2f} {eer(ga,ia)[0]*100:5.1f}% "
          f"{far_at_frr(ga,ia)[0]*100:5.1f}% | {dprime(gb,ib):6.2f} "
          f"{eer(gb,ib)[0]*100:5.1f}% {far_at_frr(gb,ib)[0]*100:5.1f}%")
    return dprime(ga, ia), dprime(gb, ib)


def zs(S):
    v = S[np.isfinite(S)]
    return (S - v.mean()) / (v.std() + 1e-12)


def main():
    ds = load_ds()
    labels = labels_of(ds)
    mats = {}
    files = sorted(glob.glob('P*.npy')) + ['S_baseline.npy', 'S_minutiae.npy']
    for f in files:
        M = np.load(f)
        if M.shape == (45, 45):
            mats[os.path.splitext(f)[0]] = M
    print(f"{'matrix':34s} {'comb':6s} | {'A d':>6s} {'A EER':>6s} {'A F10':>6s} | "
          f"{'B d':>6s} {'B EER':>6s} {'B F10':>6s}")
    print("-" * 82)
    for name, S in mats.items():
        for mode in ("max", "top2", "top3", "top5", "mean"):
            row(name, S, labels, mode)
        print()


if __name__ == "__main__":
    main()
