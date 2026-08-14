#!/usr/bin/env python3
"""
Stage 7: the headline table. One consistent framing for both protocols so the
numbers are directly comparable, with the baseline measured the same way.

Enrol pool = 19 captures, probe = 12 genuine + 14 impostor, in both protocols:
  A (POOLED)    -- the 19/12 split is drawn at random from the 31 pooled
                   genuine captures, 400 times. Averaged over the splits.
  B (REALISTIC) -- the split is fixed: enrol = right-index-cover (19),
                   probe = right-index (12). Bootstrap CI over probes.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                        # noqa: E402
from gate import composite_quality                           # noqa: E402
from strategy import STRATEGIES, score_set                   # noqa: E402
from final import bootstrap_B, paired_test                   # noqa: E402

CONFIGS = [
    ("baseline: all 19 templates", None, 19),
    ("random 15 of 19", "random", 15),
    ("top-15 by capture quality", "quality", 15),
    ("15 most DIVERSE of 19", "diverse", 15),
    ("11 most DIVERSE of 19", "diverse", 11),
]


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

    REPS = 400
    print("=" * 78)
    print("HEADLINE  (LCN + rotation + NCC scoring throughout -- the d'=1.54/1.94")
    print("baseline matcher. Only the ENROLMENT SELECTION changes.)")
    print("=" * 78)

    print(f"\nSCENARIO A -- POOLED, {REPS} random enrol(19)/probe(12) splits of the")
    print("             31 genuine captures; impostors = 14 right-middle.")
    print(f"  {'configuration':<28} {'d-prime':>14} {'EER %':>14} {'FAR@10%FRR %':>16}")
    store = {}
    for name, strat, k in CONFIGS:
        rng = np.random.default_rng(3)
        rows = []
        for _ in range(REPS):
            perm = rng.permutation(gen)
            pool, probes = np.sort(perm[:19]), np.sort(perm[19:])
            T = pool if strat is None else STRATEGIES[strat](S, pool, k, q, rng)
            rows.append(score_set(S, T, probes, imp))
        r = np.array(rows)
        store[name] = r
        print(f"  {name:<28} {r[:,0].mean():6.2f}+-{r[:,0].std():4.2f} "
              f"{r[:,1].mean()*100:8.1f}+-{r[:,1].std()*100:4.1f} "
              f"{r[:,2].mean()*100:9.1f}+-{r[:,2].std()*100:4.1f}")
    base = store["baseline: all 19 templates"]
    print("\n  paired vs baseline (same splits):")
    for name in store:
        if name.startswith("baseline"):
            continue
        for m, lbl in ((0, "d'"), (1, "EER"), (2, "FAR@10")):
            pass
        dm, p = paired_test(store[name][:, 0], base[:, 0])
        de, pe = paired_test(store[name][:, 1], base[:, 1])
        print(f"    {name:<28} dd'={dm:+.3f} (p={p:.4f})   "
              f"dEER={de*100:+.2f}pp (p={pe:.4f})")

    print("\nSCENARIO B -- REALISTIC. enrol = right-index-cover (19),")
    print("             probe = right-index (12), impostor = right-middle (14).")
    print("             [95% bootstrap CI over the probe sets]")
    print(f"  {'configuration':<28} {'d-prime (CI)':>22} {'EER % (CI)':>21} "
          f"{'FAR@10%FRR % (CI)':>22}")
    rng = np.random.default_rng(5)
    for name, strat, k in CONFIGS:
        T = rc if strat is None else STRATEGIES[strat](S, rc, k, q, rng)
        (d, e, f), lo, hi = bootstrap_B(S, np.asarray(T), ri, imp)
        print(f"  {name:<28} {d:6.2f} [{lo[0]:5.2f},{hi[0]:5.2f}] "
              f"{e*100:7.1f} [{lo[1]*100:4.1f},{hi[1]*100:5.1f}] "
              f"{f*100:9.1f} [{lo[2]*100:4.1f},{hi[2]*100:5.1f}]")

    print("\n  Reminder: 12 genuine probes means EER is quantised in 8.3% steps")
    print("  and FAR in 7.1% steps. Scenario B cannot resolve d' differences")
    print("  below about 1.0; scenario A (400 splits) is the number to trust.")


if __name__ == "__main__":
    main()
