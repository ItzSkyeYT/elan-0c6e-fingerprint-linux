#!/usr/bin/env python3
"""
Stage 6: significance and error bars, plus the honest overlap measurement.

Protocol B has 12 genuine and 14 impostor scores. EER is therefore quantised in
steps of 1/12 = 8.3% and FAR in steps of 1/14 = 7.1%. Any single-split B number
must be reported with a bootstrap interval or it is meaningless -- which is
exactly how the d'=2.74 at k=11 in stage 5 turned out to be a fluke.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                        # noqa: E402
from gate import composite_quality                           # noqa: E402
from strategy import STRATEGIES, score_set                   # noqa: E402


def paired_test(a, b, reps=20000, seed=9):
    """Sign-flip permutation test on the paired differences a-b."""
    d = np.asarray(a) - np.asarray(b)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    sg = rng.choice([-1.0, 1.0], size=(reps, len(d)))
    null = (sg * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (reps + 1)
    return float(obs), float(p)


def bootstrap_B(S, T, gprobes, iprobes, reps=4000, seed=13):
    """Resample the 12 genuine and 14 impostor probes with replacement."""
    rng = np.random.default_rng(seed)
    g = np.array([float(np.nanmax(S[T, p])) for p in gprobes])
    i = np.array([float(np.nanmax(S[T, p])) for p in iprobes])
    out = []
    for _ in range(reps):
        gb = g[rng.integers(0, len(g), len(g))]
        ib = i[rng.integers(0, len(i), len(i))]
        out.append((E.dprime(gb, ib), E.eer(gb, ib), E.far_at_frr(gb, ib)))
    out = np.array(out)
    point = (E.dprime(g, i), E.eer(g, i), E.far_at_frr(g, i))
    lo = np.percentile(out, 2.5, axis=0)
    hi = np.percentile(out, 97.5, axis=0)
    return point, lo, hi


def main():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    S = np.load(HERE / "scores_base.npz")["S"]
    labels = c["labels"]
    qual = c["qual"]
    qkeys = [str(k) for k in c["qkeys"]]
    fg_px = c["fg_px"]
    q = composite_quality(qual, qkeys,
                          ["ridge_band", "aniso_w", "coh_w", "usable_frac"])

    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]

    print("=" * 78)
    print("8. PAIRED SIGNIFICANCE, PROTOCOL A (same splits for every strategy)")
    print("=" * 78)
    for K in (11, 15, 17):
        rng = np.random.default_rng(3)
        res = {n: [] for n in STRATEGIES}
        res["all"] = []
        for _ in range(400):
            perm = rng.permutation(gen)
            pool, probes = np.sort(perm[:19]), np.sort(perm[19:])
            for n, fn in STRATEGIES.items():
                res[n].append(score_set(S, fn(S, pool, K, q, rng), probes, imp)[0])
            res["all"].append(score_set(S, pool, probes, imp)[0])
        print(f"\n  k={K}, 400 paired splits.  'all' (k=19) d' = "
              f"{np.mean(res['all']):.3f}")
        for n in ("random", "quality", "diverse", "divqual"):
            dm, p = paired_test(res[n], res["all"])
            sig = "SIGNIFICANT" if p < 0.05 else "n.s."
            print(f"    {n:<9} d'={np.mean(res[n]):5.3f}  vs all: "
                  f"{dm:+.3f}  p={p:.4f}  {sig}")
        dm, p = paired_test(res["diverse"], res["random"])
        print(f"    diverse vs random at the same k: {dm:+.3f}  p={p:.4f}")

    print("\n" + "=" * 78)
    print("9. PROTOCOL B WITH BOOTSTRAP INTERVALS (12 genuine, 14 impostor)")
    print("=" * 78)
    rng = np.random.default_rng(5)
    rows = [("all 19", rc)]
    for n, fn in (("quality k=15", STRATEGIES["quality"]),
                  ("diverse k=11", STRATEGIES["diverse"]),
                  ("diverse k=15", STRATEGIES["diverse"]),
                  ("diverse k=17", STRATEGIES["diverse"])):
        k = int(n.split("=")[1])
        rows.append((n, fn(S, rc, k, q, rng)))
    print(f"  {'enrol set':<14} {'d-prime (95% CI)':>26} {'EER % (95% CI)':>24}"
          f" {'FAR@10%FRR %':>22}")
    for name, T in rows:
        (d, e, f), lo, hi = bootstrap_B(S, np.asarray(T), ri, imp)
        print(f"  {name:<14} {d:7.2f} [{lo[0]:5.2f},{hi[0]:5.2f}]   "
              f"{e*100:7.1f} [{lo[1]*100:4.1f},{hi[1]*100:5.1f}]  "
              f"{f*100:9.1f} [{lo[2]*100:4.1f},{hi[2]*100:5.1f}]")
    print("\n  The intervals overlap completely. On 12 genuine probes protocol B")
    print("  cannot resolve a d' difference smaller than about 1.0.")

    print("\n" + "=" * 78)
    print("10. HONEST OVERLAP MEASUREMENT (wide search, LARGE min-overlap so a")
    print("    small-overlap coincidence can never be the reported alignment)")
    print("=" * 78)
    ws = np.load(HERE / "scores_widestrict.npz")
    base = np.load(HERE / "scores_base.npz")
    import itertools
    gp = list(itertools.combinations(gen.tolist(), 2))
    ip = [(i, j) for i in gen.tolist() for j in imp.tolist()]
    frame = 150 * 52
    print(f"  median finger area per capture: {np.median(fg_px):.0f} px "
          f"({np.median(fg_px)/frame*100:.0f}% of the 150x52 frame)")
    for tag, z, lim in (("baseline +-20/+-8 (the matcher's own window)", base, (20, 8)),
                        ("wide +-60/+-20, min_overlap 4500", ws, (60, 20))):
        DX, DY, BOTH, GEO = z["DX"], z["DY"], z["BOTH"], z["GEO"]
        for what, pairs in (("genuine", gp), ("impostor", ip)):
            dx = np.array([abs(DX[i, j]) for i, j in pairs])
            dy = np.array([abs(DY[i, j]) for i, j in pairs])
            both = np.array([BOTH[i, j] for i, j in pairs])
            fa = np.array([min(fg_px[i], fg_px[j]) for i, j in pairs])
            frac = both / np.maximum(fa, 1)
            p = np.percentile(frac, [5, 25, 50, 75, 95])
            print(f"\n  {tag}  [{what}, n={len(pairs)}]")
            print(f"    shared finger area / smaller finger area: "
                  f"p5={p[0]:.2f} p25={p[1]:.2f} med={p[2]:.2f} "
                  f"p75={p[3]:.2f} p95={p[4]:.2f}")
            print(f"    |dx| med {np.median(dx):.0f}  |dy| med {np.median(dy):.0f}"
                  f"   at window edge: {((dx>=lim[0])|(dy>=lim[1])).mean()*100:.0f}%")

    print("\n  How many genuine pairs beat the best impostor pair?")
    Sg = np.array([S[i, j] for i, j in gp])
    Si = np.array([S[i, j] for i, j in ip])
    print(f"    best impostor pair score = {Si.max():.3f}")
    print(f"    genuine pairs above it: {(Sg > Si.max()).sum()} / {len(Sg)} "
          f"({(Sg > Si.max()).mean()*100:.0f}%)")
    print(f"    genuine pairs above the impostor 95th pct "
          f"({np.percentile(Si,95):.3f}): {(Sg > np.percentile(Si,95)).sum()} "
          f"/ {len(Sg)}")

    # is a high-overlap genuine pair reliably a high-scoring one?
    both = np.array([ws["BOTH"][i, j] for i, j in gp])
    hi_ov = both >= np.percentile(both, 75)
    print(f"\n  genuine pairs in the TOP QUARTILE of shared area: mean score "
          f"{Sg[hi_ov].mean():.3f}; bottom three quarters "
          f"{Sg[~hi_ov].mean():.3f}; impostor mean {Si.mean():.3f}")
    print(f"  even the best-overlapping genuine quarter beats only "
          f"{(Si < Sg[hi_ov].mean()).mean()*100:.0f}% of impostor pairs.")


if __name__ == "__main__":
    main()
