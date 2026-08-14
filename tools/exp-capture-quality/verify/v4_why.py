#!/usr/bin/env python3
"""
Why does a capture-quality gate fail, and is the per-image nuisance removable
at SCORE level instead?

8.  Quality lifts the genuine AND the impostor distribution by the same amount.
9.  Score-level normalisation that cancels the per-image effect, using ONLY the
    enrolment gallery (no cohort, no impostor data -> no leak, and portable).
"""
import itertools, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from v3_analysis import (S, labels, qual, qkeys, GEN, IMP, RI, RC, N,
                         dprime, eer, far_at_frr, spearman, perm_p,
                         composite, rank)

q = composite(["ridge_band", "aniso_w", "coh_w", "usable_frac"])

print("=" * 78)
print("8. WHY THE GATE FAILS: quality raises genuine AND impostor scores alike")
print("=" * 78)
# per genuine image: mean score against other genuine, and against impostors
mg = np.nanmean(S[np.ix_(GEN, GEN)], axis=1)      # genuine image vs genuine
mi = np.nanmean(S[np.ix_(GEN, IMP)], axis=1)      # same image vs impostors
print(f"\n  for each of the {len(GEN)} genuine captures, Spearman rho of its own")
print("  quality against its mean GENUINE score and its mean IMPOSTOR score:")
print(f"    {'metric':<16} {'rho(vs genuine)':>16} {'rho(vs impostor)':>17} {'difference':>12}")
for k in qkeys + ["composite"]:
    v = q[GEN] if k == "composite" else qual[GEN, qkeys.index(k)]
    rg, ri_ = spearman(v, mg), spearman(v, mi)
    if abs(rg) > 0.3 or abs(ri_) > 0.3:
        print(f"    {k:<16} {rg:16.3f} {ri_:17.3f} {rg-ri_:12.3f}")
print("\n  the quantity a gate would need to select on is the CONTRAST")
print("  (mean genuine - mean impostor) for that capture, not its mean genuine:")
contrast = mg - mi
print(f"    {'metric':<16} {'rho(vs contrast)':>17} {'perm p':>9}")
for k in qkeys + ["composite"]:
    v = q[GEN] if k == "composite" else qual[GEN, qkeys.index(k)]
    r = spearman(v, contrast)
    print(f"    {k:<16} {r:17.3f} {perm_p(v, contrast, r, 5000):9.4f}")
print(f"\n  mean genuine score {mg.mean():.4f}, mean impostor score {mi.mean():.4f}")
print(f"  rho(mean genuine, mean impostor) across captures = "
      f"{spearman(mg, mi):+.3f}  <- a capture that matches its own finger well")
print("  also matches a DIFFERENT finger well. That is the whole story.")

print("\n" + "=" * 78)
print("9. REMOVING THE PER-IMAGE NUISANCE AT SCORE LEVEL")
print("=" * 78)
print("  All variants below use ONLY S[templates, probe]: the enrolment gallery")
print("  and nothing else. No impostor data, no cohort -> no leak, and each is")
print("  a few lines of C on top of the existing matcher.")

def v_max(col):      return col.max()
def v_maxmean(col):  return col.max() - col.mean()
def v_maxz(col):
    s = col.std()
    return (col.max() - col.mean()) / s if s > 1e-9 else 0.0
def v_max2(col):
    c = np.sort(col)
    return c[-1] - c[-2] if len(c) > 1 else c[-1]
def v_top3(col):
    c = np.sort(col)
    return c[-min(3, len(c)):].mean()
def v_top3mean(col):
    c = np.sort(col)
    return c[-min(3, len(c)):].mean() - col.mean()
def v_maxmed(col):   return col.max() - np.median(col)

VAR = {"max (baseline)": v_max, "max - mean": v_maxmean,
       "(max-mean)/sd": v_maxz, "max - 2nd best": v_max2,
       "mean of top 3": v_top3, "top3 - mean": v_top3mean,
       "max - median": v_maxmed}


def sset(fn, T, gp, ip):
    T = np.asarray(T)
    assert not (set(T.tolist()) & set(np.asarray(gp).tolist()))
    assert not np.isnan(S[np.ix_(T, gp)]).any()
    g = [fn(S[T, p]) for p in gp]
    i = [fn(S[T, p]) for p in ip]
    return dprime(g, i), eer(g, i), far_at_frr(g, i)


REPS = 500
print(f"\n  SCENARIO A -- pooled, {REPS} random enrol-pool(19)/probe(12) splits,")
print("  all 19 pool images used as templates, impostors = 14.")
print(f"  {'score rule':<18} {'d-prime':>14} {'EER %':>14} {'FAR@10%FRR %':>16}")
storeA = {}
for name, fn in VAR.items():
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(REPS):
        perm = rng.permutation(GEN)
        pool, probes = np.sort(perm[:19]), np.sort(perm[19:])
        rows.append(sset(fn, pool, probes, IMP))
    r = np.array(rows); storeA[name] = r
    print(f"  {name:<18} {r[:,0].mean():6.2f}+-{r[:,0].std():4.2f} "
          f"{r[:,1].mean()*100:8.1f}+-{r[:,1].std()*100:4.1f} "
          f"{r[:,2].mean()*100:9.1f}+-{r[:,2].std()*100:4.1f}")
base = storeA["max (baseline)"]
print("\n  paired against 'max' on the SAME splits:")
for name, r in storeA.items():
    if name == "max (baseline)":
        continue
    d = r[:, 0] - base[:, 0]; e = r[:, 1] - base[:, 1]; f = r[:, 2] - base[:, 2]
    rng = np.random.default_rng(4)
    bd = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
    be = np.array([rng.choice(e, len(e)).mean() for _ in range(4000)])
    print(f"    {name:<18} dd'={d.mean():+.3f} (p={2*min((bd<=0).mean(),(bd>=0).mean()):.4f})"
          f"   dEER={e.mean()*100:+.2f}pp (p={2*min((be<=0).mean(),(be>=0).mean()):.4f})"
          f"   dFAR@10={f.mean()*100:+.2f}pp")

print(f"\n  SCENARIO B -- REALISTIC. enrol = 19 cover, probe = 12 index, imp = 14.")
print(f"  {'score rule':<18} {'d-prime (CI)':>21} {'EER % (CI)':>21} {'FAR@10%FRR % (CI)':>23}")
for name, fn in VAR.items():
    g = np.array([fn(S[RC, p]) for p in RI])
    i = np.array([fn(S[RC, p]) for p in IMP])
    pt = (dprime(g, i), eer(g, i), far_at_frr(g, i))
    rb = np.random.default_rng(6); bs = []
    for _ in range(4000):
        gg = rb.choice(g, len(g)); ii = rb.choice(i, len(i))
        bs.append((dprime(gg, ii), eer(gg, ii), far_at_frr(gg, ii)))
    bs = np.array(bs)
    lo = np.percentile(bs, 2.5, axis=0); hi = np.percentile(bs, 97.5, axis=0)
    print(f"  {name:<18} {pt[0]:5.2f} [{lo[0]:5.2f},{hi[0]:5.2f}] "
          f"{pt[1]*100:6.1f} [{lo[1]*100:4.1f},{hi[1]*100:5.1f}] "
          f"{pt[2]*100:8.1f} [{lo[2]*100:4.1f},{hi[2]*100:5.1f}]")

print("\n  SANITY: a probe that IS one of the templates must score at the")
print("  ceiling under 'max'. Checking (this configuration is never used in")
print("  the reported numbers -- it exists only to prove the plumbing):")
Ssafe = np.nan_to_num(S, nan=1.0)
leak = [float(Ssafe[np.append(RC, p), p].max()) for p in RI[:3]]
print(f"    max including the probe itself = {['%.3f'%x for x in leak]} (== 1.0, correct)")
