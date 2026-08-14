#!/usr/bin/env python3
"""
16. The only capture-side lever that survived: HOW MANY presses to enrol, and
    WHICH KIND. No quality metric involved -- pure capture protocol.

    a) d' / EER / FAR as a function of the number of enrolled captures, under
       both protocols
    b) is a deliberately-varied ('cover') enrolment better than a habitual one?
       Matched k, cross-set only, so no probe is ever its own template.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s1_measure import metrics, score_probe, protoA_loo, bootstrapB  # noqa: E402


def main():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    S = np.load(HERE / "scores_base.npz")["S"]
    labels = c["labels"]
    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]

    print("=" * 78)
    print("16a. HOW MANY PRESSES SHOULD BE ENROLLED?")
    print("=" * 78)
    print("\n   SCENARIO A (POOLED, strict leave-one-out): every one of the 31")
    print("   genuine captures is a probe, scored against a random k-subset of")
    print("   the OTHER 30. Impostors get an independent random k-subset.")
    print("   200 repetitions.")
    print(f"   {'k templates':>12} {'d-prime':>14} {'EER %':>14} {'FAR@10%FRR %':>16}")
    for k in (1, 2, 3, 5, 8, 10, 12, 15, 19, 24, 30):
        m, s, n = protoA_loo(S, gen, imp, k, 200, 0)
        print(f"   {k:12d} {m[0]:7.2f}+-{s[0]:4.2f} {m[1]*100:8.1f}+-{s[1]*100:4.1f} "
              f"{m[2]*100:10.1f}+-{s[2]*100:4.1f}")

    print("\n   SCENARIO B (REALISTIC): templates = random k of the 19 cover")
    print("   captures, probes fixed at 12 right-index + 14 right-middle.")
    print("   200 random draws of the template subset.")
    print(f"   {'k templates':>12} {'d-prime':>14} {'EER %':>14} {'FAR@10%FRR %':>16}")
    rng = np.random.default_rng(4)
    for k in (1, 2, 3, 5, 8, 10, 12, 15, 19):
        rows = []
        reps = 1 if k == 19 else 200
        for _ in range(reps):
            T = rc if k == 19 else rng.choice(rc, k, replace=False)
            g = [score_probe(S, T, p) for p in ri]
            i = [score_probe(S, T, p) for p in imp]
            rows.append(metrics(g, i))
        a = np.array(rows)
        print(f"   {k:12d} {a[:,0].mean():7.2f}+-{a[:,0].std():4.2f} "
              f"{a[:,1].mean()*100:8.1f}+-{a[:,1].std()*100:4.1f} "
              f"{a[:,2].mean()*100:10.1f}+-{a[:,2].std()*100:4.1f}")

    print("\n" + "=" * 78)
    print("16b. IS A DELIBERATELY-VARIED ENROLMENT BETTER THAN A HABITUAL ONE?")
    print("=" * 78)
    print("   Matched k=10, cross-set in both directions so no probe can be its")
    print("   own template. 400 random template draws each.")
    rng = np.random.default_rng(6)
    for name, pool, probes in (
            ("enrol 10 of the 19 VARIED  -> probe the 12 habitual", rc, ri),
            ("enrol 10 of the 12 HABITUAL-> probe the 19 varied  ", ri, rc)):
        rows = []
        for _ in range(400):
            T = rng.choice(pool, 10, replace=False)
            g = [score_probe(S, T, p) for p in probes]
            i = [score_probe(S, T, p) for p in imp]
            rows.append(metrics(g, i))
        a = np.array(rows)
        print(f"   {name}  d'={a[:,0].mean():5.2f}+-{a[:,0].std():.2f}  "
              f"EER={a[:,1].mean()*100:5.1f}%  FAR@10={a[:,2].mean()*100:5.1f}%")

    print("\n" + "=" * 78)
    print("16c. BEST CONFIGURATION THIS INVESTIGATION FOUND, BOTH PROTOCOLS")
    print("=" * 78)
    m, s, _ = protoA_loo(S, gen, imp, 30, 400, 0)
    print(f"   A  enrol all 30 available (LOO), 400 reps:")
    print(f"      d'={m[0]:.2f}+-{s[0]:.2f}   EER={m[1]*100:.1f}%+-{s[1]*100:.1f}"
          f"   FAR@10%FRR={m[2]*100:.1f}%+-{s[2]*100:.1f}")
    (d, e, f), lo, hi = bootstrapB(S, rc, ri, imp)
    print(f"   B  enrol all 19 cover, probe 12 index:")
    print(f"      d'={d:.2f} [{lo[0]:.2f},{hi[0]:.2f}]   "
          f"EER={e*100:.1f}% [{lo[1]*100:.1f},{hi[1]*100:.1f}]   "
          f"FAR@10%FRR={f*100:.1f}% [{lo[2]*100:.1f},{hi[2]*100:.1f}]")
    print("\n   Both are the UNMODIFIED baseline matcher with the largest")
    print("   available enrolment set. No quality gate, no capture selection,")
    print("   no score normalisation improved on it.")


if __name__ == "__main__":
    main()
