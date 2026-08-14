#!/usr/bin/env python3
"""Final measurement for the v5 local-patch matcher.

ONE config, ONE score variant, both mandated protocols, with bootstrap
confidence intervals and a PAIRED comparison against the control that is nested
inside the very same pipeline (aggregate over all patches instead of the best
K, which is ordinary overlap-restricted NCC = the published baseline).

The paired comparison is the honest claim: the two numbers come from the same
images, the same enhancement, the same alignment search, and differ only in the
aggregation statistic, so the difference is attributable to the trimming rather
than to any incidental difference in the harness.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blockncc as BN
import evaluate as EV
import par4

# chosen on principle, not by scanning the leaderboard:
#   alignment estimated from ALL patches (robust), score read from the best 8
#   (the shared region), probe is the query and the enrolled image is the
#   database, which is how a driver would actually run it.
VARIANT = "sel30sc8"
CONTROL = "max0"          # aggregate over every overlapping patch == plain NCC

CFG = dict(BN.DEFAULT)
CFG.update(enh="lcn", r=8, sub=2, q_step=5, elastic=2,
           rots=(-12.0, -6.0, 0.0, 6.0, 12.0),
           max_tx=55, max_ty=22, min_valid=40)


def scenario_a_scores(S, labels, n_enroll=6, n_trials=32, seed=0):
    """Return the raw (gen, imp) score vectors for each random template subset."""
    gidx = np.where(labels != "right-middle")[0]
    iidx = np.where(labels == "right-middle")[0]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        perm = rng.permutation(gidx)
        gen = [max(S[t, p] for t in [t for t in perm if t != p][:n_enroll])
               for p in gidx]
        tmpl = list(perm[:n_enroll])
        imp = [max(S[t, p] for t in tmpl) for p in iidx]
        out.append((np.array(gen), np.array(imp)))
    return out


def scenario_b_scores(S, labels):
    enrol = np.where(labels == "right-index-cover")[0]
    probe = np.where(labels == "right-index")[0]
    imp = np.where(labels == "right-middle")[0]
    g = np.array([max(S[t, p] for t in enrol) for p in probe])
    i = np.array([max(S[t, p] for t in enrol) for p in imp])
    return g, i


def stats(g, i):
    return dict(dprime=EV.D.dprime(g, i), eer=EV.D.eer(g, i)[0],
                far10=EV.D.far_at_frr(g, i, 0.10)[0])


def boot(g, i, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        gb = g[rng.integers(0, len(g), len(g))]
        ib = i[rng.integers(0, len(i), len(i))]
        out.append(stats(gb, ib))
    return {k: (float(np.percentile([o[k] for o in out], 2.5)),
                float(np.percentile([o[k] for o in out], 97.5)))
            for k in ("dprime", "eer", "far10")}


def boot_diff(g1, i1, g2, i2, n=4000, seed=1):
    """Paired bootstrap of d'(variant) - d'(control): the SAME resampled probes
    are scored by both, so the comparison is paired."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        gi = rng.integers(0, len(g1), len(g1))
        ii = rng.integers(0, len(i1), len(i1))
        d.append(EV.D.dprime(g1[gi], i1[ii]) - EV.D.dprime(g2[gi], i2[ii]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5)), float((d <= 0).mean())


def line(tag, s, ci=None):
    if ci:
        print(f"    {tag:<10s} d'={s['dprime']:5.2f} [{ci['dprime'][0]:.2f},{ci['dprime'][1]:.2f}]"
              f"   EER={s['eer']*100:5.1f}% [{ci['eer'][0]*100:.0f},{ci['eer'][1]*100:.0f}]"
              f"   FAR@10%FRR={s['far10']*100:5.1f}% [{ci['far10'][0]*100:.0f},{ci['far10'][1]*100:.0f}]")
    else:
        print(f"    {tag:<10s} d'={s['dprime']:5.2f}   EER={s['eer']*100:5.1f}%"
              f"   FAR@10%FRR={s['far10']*100:5.1f}%")


def main():
    imgs, labels, names = EV.load()
    cache = os.path.join(HERE, "Q5_final.npz")
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        d = np.load(cache, allow_pickle=True)
        Q = {k: d[k] for k in d.files if k != "labels"}
        labels = d["labels"]
        print("loaded cached score matrices")
    else:
        t0 = time.time()
        Q, T = par4.matrices(CFG, imgs, nproc=15, mod=BN)
        np.savez(cache, labels=labels, **Q)
        print(f"  computed in {time.time()-t0:.0f}s")

    for v in (VARIANT, CONTROL):
        d = np.diag(Q[v]); off = Q[v].copy(); np.fill_diagonal(off, -np.inf)
        print(f"  sanity[{v}]: self mean {d.mean():.3f} min {d.min():.3f}; "
              f"best-other mean {off.max(axis=1).mean():.3f}; "
              f"{int((d < off.max(axis=1)).sum())} image(s) not maximal against "
              f"themselves")

    Sv, Sc = Q[VARIANT].T, Q[CONTROL].T      # S[template, probe]

    print(f"\n=== SCENARIO A (pooled, 31 genuine / 14 impostor, 6 templates, "
          f"32 random subsets) ===")
    av = scenario_a_scores(Sv, labels)
    ac = scenario_a_scores(Sc, labels)
    for tag, runs in (("v5 " + VARIANT, av), ("control " + CONTROL, ac)):
        m = {k: float(np.mean([stats(g, i)[k] for g, i in runs]))
             for k in ("dprime", "eer", "far10")}
        sd = float(np.std([stats(g, i)["dprime"] for g, i in runs]))
        print(f"    {tag:<18s} d'={m['dprime']:5.2f} (sd over subsets {sd:.2f})"
              f"   EER={m['eer']*100:5.1f}%   FAR@10%FRR={m['far10']*100:5.1f}%")
    gv = np.concatenate([g for g, i in av]); iv = np.concatenate([i for g, i in av])
    gc = np.concatenate([g for g, i in ac]); ic = np.concatenate([i for g, i in ac])
    md, lo, hi, p = boot_diff(gv, iv, gc, ic)
    print(f"    paired bootstrap  delta d' = {md:+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]"
          f"   P(no gain) = {p:.3f}")

    print(f"\n=== SCENARIO B (enrol 19 cover, probe 12 index, 14 impostor) ===")
    gv, iv = scenario_b_scores(Sv, labels)
    gc, ic = scenario_b_scores(Sc, labels)
    line("v5", stats(gv, iv), boot(gv, iv))
    line("control", stats(gc, ic), boot(gc, ic))
    md, lo, hi, p = boot_diff(gv, iv, gc, ic)
    print(f"    paired bootstrap  delta d' = {md:+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]"
          f"   P(no gain) = {p:.3f}")
    print(f"    genuine  {gv.mean():.3f} +/- {gv.std():.3f}  [{gv.min():.3f}..{gv.max():.3f}]")
    print(f"    impostor {iv.mean():.3f} +/- {iv.std():.3f}  [{iv.min():.3f}..{iv.max():.3f}]")

    print("\n=== trim family curve (scenario B, sym=none): is the gain "
          "systematic or a lucky variant? ===")
    for k in list(CFG["trims"]) + [f"f{int(f*100)}" for f in CFG["fracs"]]:
        for fam in ("max", "sel30sc"):
            v = f"{fam}{k}"
            if v not in Q:
                continue
            g, i = scenario_b_scores(Q[v].T, labels)
            a = scenario_a_scores(Q[v].T, labels)
            ad = float(np.mean([stats(gg, ii)["dprime"] for gg, ii in a]))
            print(f"    {v:<12s}  A d'={ad:5.2f}   B d'={stats(g, i)['dprime']:5.2f}")


if __name__ == "__main__":
    main()
