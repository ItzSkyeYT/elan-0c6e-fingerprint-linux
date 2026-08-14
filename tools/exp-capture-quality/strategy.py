#!/usr/bin/env python3
"""
Stage 5: enrolment-SELECTION strategies, measured over many random splits so
that no single lucky subset (and no post-hoc choice of k) can carry the result.

Strategies, all of which only ever look at the enrolment pool:
  random    -- k drawn uniformly (what the driver effectively does)
  quality   -- top k by the composite capture-quality score
  diverse   -- k chosen greedily to MINIMISE mutual similarity, i.e. to cover
               as many distinct placements as possible
  divqual   -- diverse, but seeded from and tie-broken by quality
  all       -- use the whole pool (no selection)

Protocol A here is the pooled protocol from the brief, restated as an explicit
enrol-pool / probe split so that a selection strategy has something to select
from. Probes are never in the pool, so no image is ever its own template.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                        # noqa: E402
from gate import composite_quality                           # noqa: E402


def sel_random(S, pool, k, q, rng):
    return rng.choice(pool, size=k, replace=False)


def sel_quality(S, pool, k, q, rng):
    pool = np.asarray(pool)
    return pool[np.argsort(-q[pool])[:k]]


def _greedy_diverse(S, pool, k, seed_idx):
    sel = list(seed_idx)
    while len(sel) < k:
        rest = [p for p in pool if p not in sel]
        if not rest:
            break
        cand = [(max(S[p, s] for s in sel), p) for p in rest]
        cand.sort()
        sel.append(cand[0][1])
    return np.array(sel)


def sel_diverse(S, pool, k, q, rng):
    pool = list(pool)
    lo = min(((S[i, j], i, j) for i, j in itertools.combinations(pool, 2)))
    return _greedy_diverse(S, pool, k, [lo[1], lo[2]])


def sel_divqual(S, pool, k, q, rng):
    """Diverse, but start from the highest-quality capture rather than from
    whichever pair happens to correlate worst (which can be two bad images)."""
    pool = list(pool)
    start = max(pool, key=lambda p: q[p])
    return _greedy_diverse(S, pool, k, [start])


def sel_all(S, pool, k, q, rng):
    return np.asarray(pool)


STRATEGIES = {"random": sel_random, "quality": sel_quality,
              "diverse": sel_diverse, "divqual": sel_divqual}


def score_set(S, T, gprobes, iprobes):
    g = [float(np.nanmax(S[T, p])) for p in gprobes]
    i = [float(np.nanmax(S[T, p])) for p in iprobes]
    assert not np.isnan(S[np.ix_(T, gprobes)]).any(), "self-comparison leak"
    return E.dprime(g, i), E.eer(g, i), E.far_at_frr(g, i)


def protocolA(S, gen, imp, q, pool_size=19, k=11, reps=200, seed=3):
    """Random enrol-pool / probe split over the 31 pooled genuine captures."""
    rng = np.random.default_rng(seed)
    out = {n: [] for n in STRATEGIES}
    out["all"] = []
    for _ in range(reps):
        perm = rng.permutation(gen)
        pool, probes = np.sort(perm[:pool_size]), np.sort(perm[pool_size:])
        assert not set(pool.tolist()) & set(probes.tolist())
        for name, fn in STRATEGIES.items():
            T = fn(S, pool, k, q, rng)
            out[name].append(score_set(S, T, probes, imp))
        out["all"].append(score_set(S, pool, probes, imp))
    return {n: np.array(v) for n, v in out.items()}


def protocolB(S, rc, ri, imp, q, k, reps=200, seed=5):
    """The fixed realistic split. Only 'random' needs repetitions."""
    rng = np.random.default_rng(seed)
    res = {}
    for name, fn in STRATEGIES.items():
        if name == "random":
            v = [score_set(S, fn(S, rc, k, q, rng), ri, imp)
                 for _ in range(reps)]
            res[name] = np.array(v)
        else:
            res[name] = np.array([score_set(S, fn(S, rc, k, q, rng), ri, imp)])
    res["all"] = np.array([score_set(S, rc, ri, imp)])
    return res


def show(title, res, note=""):
    print(f"\n{title} {note}")
    print(f"  {'strategy':<10} {'d-prime':>16} {'EER %':>15} {'FAR@10%FRR %':>16}")
    for n in ("all", "random", "quality", "diverse", "divqual"):
        v = res[n]
        if len(v) > 1:
            print(f"  {n:<10} {v[:,0].mean():9.2f}+-{v[:,0].std():4.2f} "
                  f"{v[:,1].mean()*100:9.1f}+-{v[:,1].std()*100:4.1f} "
                  f"{v[:,2].mean()*100:10.1f}+-{v[:,2].std()*100:4.1f}")
        else:
            print(f"  {n:<10} {v[0,0]:9.2f}      {v[0,1]*100:9.1f}      "
                  f"{v[0,2]*100:10.1f}")


def main():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    S = np.load(HERE / "scores_base.npz")["S"]
    labels = c["labels"]
    qual = c["qual"]
    qkeys = [str(k) for k in c["qkeys"]]
    q = composite_quality(qual, qkeys,
                          ["ridge_band", "aniso_w", "coh_w", "usable_frac"])

    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]

    print("=" * 78)
    print("7. ENROLMENT SELECTION, VALIDATED OVER RANDOM SPLITS")
    print("=" * 78)
    print("  Protocol A restated as enrol-pool(19)/probe(12) random splits of")
    print("  the 31 pooled genuine captures, 200 repetitions, impostors = 14.")

    print("\n  sweeping k (templates kept out of a 19-image pool):")
    print(f"  {'k':>3} | " + " | ".join(f"{n:>13}" for n in
                                        ("random", "quality", "diverse", "divqual")))
    bestk = {}
    for k in (3, 5, 7, 9, 11, 13, 15, 17, 19):
        r = protocolA(S, gen, imp, q, 19, k, reps=120, seed=3)
        row = []
        for n in ("random", "quality", "diverse", "divqual"):
            m, s = r[n][:, 0].mean(), r[n][:, 0].std()
            row.append(f"{m:6.2f}+-{s:4.2f}")
            bestk.setdefault(n, []).append((m, k))
        print(f"  {k:3d} | " + " | ".join(row))
    print(f"  (k=19 is the whole pool, so all strategies coincide there)")

    kstar = {n: max(v)[1] for n, v in bestk.items()}
    print(f"\n  best k per strategy, chosen on protocol A: {kstar}")

    K = kstar["divqual"]
    print(f"\n  --- full protocol A at k={K}, 200 reps ---")
    rA = protocolA(S, gen, imp, q, 19, K, reps=200, seed=17)
    show("PROTOCOL A (pooled, enrol-pool 19 / probe 12, 200 random splits)", rA)

    print(f"\n  --- protocol B at the SAME k={K} (chosen on A, not on B) ---")
    rB = protocolB(S, rc, ri, imp, q, K, reps=200)
    show("PROTOCOL B (enrol right-index-cover, probe right-index)", rB)

    print("\n  protocol B, k swept (shown for completeness -- k was NOT chosen "
          "here):")
    print(f"  {'k':>3} | " + " | ".join(f"{n:>13}" for n in
                                        ("random", "quality", "diverse", "divqual")))
    for k in (5, 7, 9, 11, 13, 15, 17, 19):
        rb = protocolB(S, rc, ri, imp, q, k, reps=60)
        row = []
        for n in ("random", "quality", "diverse", "divqual"):
            v = rb[n]
            row.append(f"{v[:,0].mean():6.2f}" + (f"+-{v[:,0].std():4.2f}"
                                                  if len(v) > 1 else "      "))
        print(f"  {k:3d} | " + " | ".join(row))


if __name__ == "__main__":
    main()
