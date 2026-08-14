#!/usr/bin/env python3
"""
Final, honest evaluation.

Three things this does that the exploratory scripts do not:

1. PAIRED scenario-A comparison.  The baseline and the candidate are evaluated
   on the SAME random template subsets, and the per-trial difference is reported
   with its own standard deviation.  Two independently-averaged d-primes can
   differ by less than the trial-to-trial noise; the paired difference cannot
   hide that.

2. BOOTSTRAP interval for scenario B.  Scenario B rests on 12 genuine and 14
   impostor scores.  A d-prime from 26 numbers has a very wide interval, and
   quoting it bare would overstate the result.  Genuine probes and impostors are
   resampled with replacement.

3. A FIXED fusion weight, chosen once and applied to both protocols, instead of
   the per-protocol argmax.  Picking w by the score it produces is fitting the
   test set; the sweep is printed so the plateau is visible, but the headline
   number uses the pre-committed weight.

    python3 final.py [--w 0.25] [--tag w4_b8] [--n-enroll 6] [--trials 64]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                       # noqa: F401,F403
from run import load_flat, CACHE           # noqa

WEIGHTS = np.round(np.arange(0.0, 1.001, 0.1), 3)


def gar_at_zero_far(gen, imp):
    """Fraction of genuine probes scoring above EVERY impostor.

    d-prime and EER both average over the whole score range; an unlock decision
    lives entirely in the upper tail, where the threshold sits above the worst
    impostor.  On this dataset the two families of metric disagree sharply, so
    reporting only d' would hide the actual result.  With 14 impostors from a
    single finger this is an optimistic absolute number -- it is quoted only to
    compare matchers evaluated on identical data.
    """
    return float((np.asarray(gen) > np.max(imp)).mean())


def metrics(gen, imp):
    return np.array([dprime(gen, imp), eer(gen, imp)[0],
                     far_at_frr(gen, imp, 0.10)[0], gar_at_zero_far(gen, imp)])


def subsets(gen_idx, n_enroll, trials, seed=20260808):
    rng = np.random.default_rng(seed)
    gen_idx = np.asarray(gen_idx)
    out = []
    for _ in range(trials):
        T = rng.choice(gen_idx, size=n_enroll, replace=False)
        Ts = set(T.tolist())
        out.append((T, np.array([p for p in gen_idx.tolist() if p not in Ts])))
    return out


def scenarioA(M, subs, imp_idx):
    rows = []
    for T, probes in subs:
        g = [M[T, p].max() for p in probes]
        i = [M[T, q].max() for q in imp_idx]
        rows.append(metrics(g, i))
    return np.array(rows)                  # trials x 3


def scenarioB(M, tmpl, probes, imps, boots=4000, seed=7):
    T = np.asarray(tmpl)
    g = np.array([M[T, p].max() for p in probes])
    i = np.array([M[T, q].max() for q in imps])
    point = metrics(g, i)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boots):
        gg = g[rng.integers(0, g.size, g.size)]
        ii = i[rng.integers(0, i.size, i.size)]
        if gg.std() + ii.std() < 1e-9:
            continue
        bs.append(metrics(gg, ii))
    bs = np.array(bs)
    lo = np.percentile(bs, 2.5, axis=0)
    hi = np.percentile(bs, 97.5, axis=0)
    return point, lo, hi, g, i


def line(tag, a, b):
    (pt, lo, hi, _, _) = b
    print(f"  {tag}")
    print(f"      A  d'={a[:,0].mean():5.2f}+-{a[:,0].std():.2f}   "
          f"EER={a[:,1].mean()*100:5.1f}%   FAR@10%FRR={a[:,2].mean()*100:5.1f}%   "
          f"GAR@0FAR={a[:,3].mean()*100:5.1f}%+-{a[:,3].std()*100:.1f}")
    print(f"      B  d'={pt[0]:5.2f} [{lo[0]:.2f},{hi[0]:.2f}]   "
          f"EER={pt[1]*100:5.1f}%   FAR@10%FRR={pt[2]*100:5.1f}%   "
          f"GAR@0FAR={pt[3]*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w4_b8")
    ap.add_argument("--w", type=float, default=0.25)
    ap.add_argument("--n-enroll", type=int, default=6)
    ap.add_argument("--trials", type=int, default=64)
    args = ap.parse_args()

    imgs, labels = load_flat()
    ii = np.where(labels == "right-index")[0]
    ic = np.where(labels == "right-index-cover")[0]
    im = np.where(labels == "right-middle")[0]
    pool = np.concatenate([ii, ic])
    subs = subsets(pool, args.n_enroll, args.trials)

    Mb = np.load(os.path.join(CACHE, "ncc_lcn_rot.npy"))
    S = np.load(os.path.join(CACHE, args.tag + ".npy"))     # (len(WEIGHTS), n, n)
    # interpolate the fused matrix at the pre-committed weight
    k = float(args.w) / 0.1
    k0 = int(np.floor(k))
    k1 = min(k0 + 1, len(WEIGHTS) - 1)
    f = k - k0
    Mf = (1 - f) * S[k0] + f * S[k1]

    print(f"dataset: {len(imgs)} captures  "
          f"(index {ii.size}, cover {ic.size}, middle {im.size})")
    print(f"scenario A: {args.n_enroll} random templates from the pooled 31, "
          f"{args.trials} subsets, leave-the-rest-out")
    print(f"scenario B: 19 cover templates, 12 index probes, 14 impostors, "
          f"4000-resample bootstrap\n")

    print("== sanity ==")
    for nm, M in (("baseline", Mb), (args.tag, Mf)):
        d = np.diag(M)
        print(f"  {nm:<12} self-match (diagonal) mean {d.mean():.4f} "
              f"min {d.min():.4f}   {'OK' if d.min() > 0.95 else 'SUSPECT'}")
    print(f"  scenario B templates and probes disjoint: "
          f"{len(set(ic.tolist()) & set(ii.tolist())) == 0}")
    print(f"  scenario A probe never in its own template set: "
          f"{all(len(set(T.tolist()) & set(P.tolist())) == 0 for T, P in subs)}\n")

    print("== headline ==")
    Ab = scenarioA(Mb, subs, im)
    Bb = scenarioB(Mb, ic, ii, im)
    line("LCN+rot+NCC (baseline)", Ab, Bb)
    Af = scenarioA(Mf, subs, im)
    Bf = scenarioB(Mf, ic, ii, im)
    line(f"fused w={args.w:.2f}", Af, Bf)

    d = Af[:, 0] - Ab[:, 0]
    print(f"\n  PAIRED scenario-A d' difference (fused - baseline) over "
          f"{args.trials} identical subsets:")
    print(f"    mean {d.mean():+.3f}   sd {d.std():.3f}   "
          f"sem {d.std()/np.sqrt(len(d)):.3f}   "
          f"fused wins in {int((d > 0).sum())}/{len(d)} subsets")
    q = Af[:, 3] - Ab[:, 3]
    print(f"  PAIRED scenario-A GAR@zero-FAR difference:")
    print(f"    mean {q.mean()*100:+.1f} pp   sd {q.std()*100:.1f}   "
          f"sem {q.std()/np.sqrt(len(q))*100:.1f}   "
          f"fused wins in {int((q > 0).sum())}/{len(q)}, "
          f"loses in {int((q < 0).sum())}/{len(q)}")

    # Paired bootstrap on scenario B: the SAME resampled probes and impostors
    # score both matchers, so the shared sampling noise cancels.  Comparing two
    # independent confidence intervals (which overlap heavily here) is the wrong
    # test and would understate a real difference.
    gb = np.array([Mb[ic, p].max() for p in ii])
    ib = np.array([Mb[ic, q].max() for q in im])
    gf = np.array([Mf[ic, p].max() for p in ii])
    _if = np.array([Mf[ic, q].max() for q in im])
    rng = np.random.default_rng(11)
    diffs = []
    for _ in range(4000):
        gi = rng.integers(0, gb.size, gb.size)
        qi = rng.integers(0, ib.size, ib.size)
        db = dprime(gb[gi], ib[qi])
        df = dprime(gf[gi], _if[qi])
        diffs.append(df - db)
    diffs = np.array(diffs)
    print(f"  PAIRED scenario-B bootstrap of d'(fused) - d'(baseline):")
    print(f"    mean {diffs.mean():+.2f}   95% CI "
          f"[{np.percentile(diffs,2.5):+.2f}, {np.percentile(diffs,97.5):+.2f}]"
          f"   P(fused better) = {float((diffs>0).mean()):.3f}")

    print(f"\n== weight sweep at the same protocol ({args.tag}) ==")
    print(f"  {'w':>5}{'A d':>8}{'A EER':>8}{'A FAR10':>9}{'A GAR0':>8}"
          f"{'B d':>8}{'B 95% CI':>16}{'B EER':>8}{'B FAR10':>9}{'B GAR0':>8}")
    for wi, w in enumerate(WEIGHTS):
        a = scenarioA(S[wi], subs, im)
        pt, lo, hi, _, _ = scenarioB(S[wi], ic, ii, im, boots=800)
        print(f"  {w:5.1f}{a[:,0].mean():8.2f}{a[:,1].mean()*100:7.1f}%"
              f"{a[:,2].mean()*100:8.1f}%{a[:,3].mean()*100:7.1f}%{pt[0]:8.2f}"
              f"   [{lo[0]:5.2f},{hi[0]:5.2f}]{pt[1]*100:7.1f}%{pt[2]*100:8.1f}%"
              f"{pt[3]*100:7.1f}%")

    print("\n== enrolment-size curve (scenario A, fused vs baseline) ==")
    print(f"  {'n_enroll':>9}{'base d':>9}{'fused d':>9}{'base EER':>10}"
          f"{'fused EER':>11}{'base GAR0':>11}{'fused GAR0':>12}")
    for ne in (3, 4, 6, 8, 10, 12, 15):
        sub = subsets(pool, ne, args.trials)
        ab = scenarioA(Mb, sub, im)
        af = scenarioA(Mf, sub, im)
        print(f"  {ne:9d}{ab[:,0].mean():9.2f}{af[:,0].mean():9.2f}"
              f"{ab[:,1].mean()*100:9.1f}%{af[:,1].mean()*100:10.1f}%"
              f"{ab[:,3].mean()*100:10.1f}%{af[:,3].mean()*100:11.1f}%")


if __name__ == "__main__":
    main()
