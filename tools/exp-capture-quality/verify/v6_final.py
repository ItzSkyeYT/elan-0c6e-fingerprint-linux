#!/usr/bin/env python3
"""
11. Final numbers, and a proper significance test on the ONE gate configuration
    that looked promising (composite quality, drop 10%, enrol+verify).
"""
import math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v3_analysis import (S, labels, qual, qkeys, GEN, IMP, RI, RC,
                         dprime, eer, far_at_frr, composite, keep_top, score_set)

q = composite(["ridge_band", "aniso_w", "coh_w", "usable_frac"])

print("=" * 78)
print("11a. BASELINE, stated three ways (no gate, no selection, all templates)")
print("=" * 78)
# leave-one-out over all 31 genuine, template = the other 30 -- directly
# comparable to the previously published d'=1.54
g = S[np.ix_(GEN, GEN)]
loo_g = np.array([np.nanmax(np.delete(g[:, k], k)) for k in range(len(GEN))])
loo_i = S[np.ix_(GEN, IMP)].max(axis=0)
print(f"  A, leave-one-out, 30 templates : d'={dprime(loo_g,loo_i):5.2f}  "
      f"EER={eer(loo_g,loo_i)*100:5.1f}%  FAR@10%FRR={far_at_frr(loo_g,loo_i)*100:5.1f}%")
rng = np.random.default_rng(3); rows = []
for _ in range(2000):
    perm = rng.permutation(GEN)
    T, pr = np.sort(perm[:19]), np.sort(perm[19:])
    rows.append(score_set(T, pr, IMP))
A = np.array(rows)
print(f"  A, 2000 random 19/12 splits    : d'={A[:,0].mean():5.2f}+-{A[:,0].std():.2f}  "
      f"EER={A[:,1].mean()*100:5.1f}%  FAR@10%FRR={A[:,2].mean()*100:5.1f}%")
B = score_set(RC, RI, IMP)
print(f"  B, enrol 19 cover / probe 12   : d'={B[0]:5.2f}       "
      f"EER={B[1]*100:5.1f}%  FAR@10%FRR={B[2]*100:5.1f}%")

print("\n" + "=" * 78)
print("11b. THE BEST-LOOKING GATE vs ITS RANDOM-DROP CONTROL, properly tested")
print("=" * 78)
print("  composite quality, drop the worst 10%, ENROL+VERIFY. Control drops the")
print("  same number of images at random, WITHIN class, 2000 draws. If the gate")
print("  carries information, its d' must sit in the upper tail of the control.")
for f in (0.10, 0.20, 0.30):
    pool = keep_top(GEN, q, f); impk = keep_top(IMP, q, f)
    rck = keep_top(RC, q, f); rik = keep_top(RI, q, f)
    npool, nimp = len(pool), len(impk)

    def runA(pl, gp, ip, seed):
        r = np.random.default_rng(seed); o = []
        for _ in range(120):
            T = r.choice(pl, min(15, len(pl) - 1), replace=False)
            pr = np.setdiff1d(gp, T)
            if len(pr) < 3:
                continue
            o.append(score_set(T, pr, ip))
        return np.array(o).mean(axis=0)

    ag = runA(pool, pool, impk, 1)
    bg = score_set(rck, rik, impk)
    rr = np.random.default_rng(2024)
    ctlA, ctlB = [], []
    for _ in range(2000):
        pr_ = np.sort(rr.choice(GEN, npool, replace=False))
        ir_ = np.sort(rr.choice(IMP, nimp, replace=False))
        rcr = np.sort(rr.choice(RC, len(rck), replace=False))
        rir = np.sort(rr.choice(RI, len(rik), replace=False))
        ctlA.append(runA(pr_, pr_, ir_, 2))
        ctlB.append(score_set(rcr, rir, ir_))
    ctlA = np.array(ctlA); ctlB = np.array(ctlB)
    pA = float((ctlA[:, 0] >= ag[0]).mean())
    pB = float((ctlB[:, 0] >= bg[0]).mean())
    print(f"\n  drop {f*100:.0f}%  (kept {npool}/{len(GEN)} genuine, {nimp}/{len(IMP)} impostor)")
    print(f"    A  gated d'={ag[0]:5.2f}   random-drop control {ctlA[:,0].mean():5.2f}"
          f"+-{ctlA[:,0].std():.2f}   one-sided p={pA:.3f}")
    print(f"       gated EER={ag[1]*100:5.1f}%  control {ctlA[:,1].mean()*100:5.1f}%"
          f"   gated FAR@10={ag[2]*100:5.1f}%  control {ctlA[:,2].mean()*100:5.1f}%")
    print(f"    B  gated d'={bg[0]:5.2f}   random-drop control {ctlB[:,0].mean():5.2f}"
          f"+-{ctlB[:,0].std():.2f}   one-sided p={pB:.3f}")
    print(f"       gated EER={bg[1]*100:5.1f}%  control {ctlB[:,1].mean()*100:5.1f}%"
          f"   gated FAR@10={bg[2]*100:5.1f}%  control {ctlB[:,2].mean()*100:5.1f}%")

print("\n" + "=" * 78)
print("11c. BEST HONEST CONFIGURATION FOUND BY THIS INVESTIGATION")
print("=" * 78)
print("  It is the UNMODIFIED baseline. No quality gate, no enrolment selection,")
print("  no score normalisation and no wider search window beat it on scenario A,")
print("  which is the only protocol here with enough probes to resolve a")
print("  difference.")
print(f"\n  A  d'={A[:,0].mean():.2f}  EER={A[:,1].mean()*100:.1f}%  "
      f"FAR@10%FRR={A[:,2].mean()*100:.1f}%")
print(f"  B  d'={B[0]:.2f}  EER={B[1]*100:.1f}%  FAR@10%FRR={B[2]*100:.1f}%")
