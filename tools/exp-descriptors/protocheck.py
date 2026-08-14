#!/usr/bin/env python3
"""Why does the LCN+rotation NCC baseline measure d'=1.04 here when it was
reported as 1.54?  Compute ONE score matrix and read scenario A off it under
several protocol conventions."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import enh, evaluate as EV
from fastncc import ncc_all_shifts

CACHE = os.path.join(HERE, "S_lcnrot.npy")
imgs, labels, names = EV.load()
if os.path.exists(CACHE):
    S = np.load(CACHE)
else:
    P = [enh.local_contrast_norm(im) for im in imgs]
    n = len(P); S = np.zeros((n, n), np.float32)
    t0 = time.time()
    for i in range(n):
        for j in range(i, n):
            best = -1.0
            for deg in range(-12, 13, 4):          # matcher-lab m_ncc_lcn_rot
                br = enh._rotate(P[j], deg) if deg else P[j]
                c = ncc_all_shifts(P[i], br, 20, 8, 3500)
                best = max(best, c)
            S[i, j] = S[j, i] = best
    print(f"  matrix in {time.time()-t0:.0f}s")
    np.save(CACHE, S)

gidx = np.where(labels != "right-middle")[0]
iidx = np.where(labels == "right-middle")[0]

def metrics(g, i):
    g, i = np.asarray(g), np.asarray(i)
    return EV.D.dprime(g, i), EV.D.eer(g, i)[0], EV.D.far_at_frr(g, i, .10)[0]

# (1) matcher-lab evaluate_template: FIXED first-N ordering
for ne in (6, 8, 10):
    gen = [max(S[t, p] for t in [t for t in gidx if t != p][:ne]) for p in gidx]
    tm = list(gidx[:ne])
    imp = [max(S[t, p] for t in tm) for p in iidx]
    d, e, f = metrics(gen, imp)
    print(f"  FIXED first-{ne} ordering   d'={d:5.2f}  EER={e*100:5.1f}%  FAR@10={f*100:5.1f}%")

# (2) mandated: averaged over random template subsets
rng = np.random.default_rng(0)
for ne in (6, 8, 10):
    out = []
    for _ in range(32):
        perm = rng.permutation(gidx)
        gen = [max(S[t, p] for t in [t for t in perm if t != p][:ne]) for p in gidx]
        tm = list(perm[:ne])
        imp = [max(S[t, p] for t in tm) for p in iidx]
        out.append(metrics(gen, imp))
    m = np.mean(out, axis=0); sd = np.std([o[0] for o in out])
    print(f"  RANDOM subsets, {ne} tmpl   d'={m[0]:5.2f} (sd {sd:.2f})  "
          f"EER={m[1]*100:5.1f}%  FAR@10={m[2]*100:5.1f}%")

# (3) plain pairwise (no template set)
import itertools
gen = [S[i, j] for i, j in itertools.combinations(gidx, 2)]
imp = [S[i, j] for i in gidx for j in iidx]
d, e, f = metrics(gen, imp)
print(f"  PAIRWISE (no template set) d'={d:5.2f}  EER={e*100:5.1f}%  FAR@10={f*100:5.1f}%")

# (4) scenario B
enrol = np.where(labels == "right-index-cover")[0]
probe = np.where(labels == "right-index")[0]
g = [max(S[t, p] for t in enrol) for p in probe]
i = [max(S[t, p] for t in enrol) for p in iidx]
d, e, f = metrics(g, i)
print(f"  SCENARIO B, 19 templates   d'={d:5.2f}  EER={e*100:5.1f}%  FAR@10={f*100:5.1f}%")
