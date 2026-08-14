#!/usr/bin/env python3
"""Evaluation protocol A/B given a full score matrix over the dataset.

Score matrix is indexed by the manifest order produced by manifest().
Never compares an image with itself; never uses a probe as its own template.
"""
import math
import numpy as np
from pathlib import Path

DS = Path("/home/melb/.local/share/elan-fp/dataset")
LABELS = ("right-index", "right-index-cover", "right-middle")


def manifest(root=DS):
    """Ordered list of (label, path). Order is the score-matrix index order."""
    out = []
    for lbl in LABELS:
        for f in sorted((root / lbl).glob("*.pgm")):
            out.append((lbl, f))
    return out


def index_sets(man):
    idx = {l: [i for i, (lbl, _) in enumerate(man) if lbl == l] for l in LABELS}
    return idx


def dprime(gen, imp):
    g, i = np.asarray(gen, float), np.asarray(imp, float)
    denom = math.sqrt((g.var() + i.var()) / 2)
    return float((g.mean() - i.mean()) / denom) if denom > 0 else 0.0


def eer(gen, imp):
    g, i = np.asarray(gen, float), np.asarray(imp, float)
    best_gap, best_eer, best_t = math.inf, 1.0, 0.0
    for t in np.unique(np.concatenate([g, i])):
        frr = float((g < t).mean())
        far = float((i >= t).mean())
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap, best_eer, best_t = gap, (frr + far) / 2, float(t)
    return best_eer, best_t


def far_at_frr(gen, imp, target_frr=0.10):
    g, i = np.asarray(gen, float), np.asarray(imp, float)
    t = float(np.quantile(g, target_frr))
    return float((i >= t).mean()), t


def _fuse(S, templates, probe, mode="max"):
    v = [S[t, probe] for t in templates if t != probe]
    if not v:
        return 0.0
    if mode == "max":
        return float(max(v))
    if mode == "sum2":                       # sum of top 2
        return float(sum(sorted(v, reverse=True)[:2]))
    if mode == "sum3":
        return float(sum(sorted(v, reverse=True)[:3]))
    if mode == "mean":
        return float(np.mean(v))
    raise ValueError(mode)


def scenario_A(S, man, n_enroll=6, reps=20, seed=0, mode="max"):
    """POOLED. genuine = right-index + right-index-cover, impostor = right-middle.
    Random template subsets of the genuine pool; probes are the held-out
    genuine images plus all impostors."""
    idx = index_sets(man)
    gpool = idx["right-index"] + idx["right-index-cover"]
    ipool = idx["right-middle"]
    rng = np.random.default_rng(seed)
    ds, es, fs = [], [], []
    allg, alli = [], []
    for _ in range(reps):
        T = list(rng.choice(gpool, size=n_enroll, replace=False))
        probes = [p for p in gpool if p not in T]
        gen = [_fuse(S, T, p, mode) for p in probes]
        imp = [_fuse(S, T, p, mode) for p in ipool]
        ds.append(dprime(gen, imp))
        es.append(eer(gen, imp)[0])
        fs.append(far_at_frr(gen, imp)[0])
        allg += gen
        alli += imp
    return {
        "dprime": float(np.mean(ds)), "dprime_sd": float(np.std(ds)),
        "eer": float(np.mean(es)), "eer_sd": float(np.std(es)),
        "far10": float(np.mean(fs)), "far10_sd": float(np.std(fs)),
        "gen": np.array(allg), "imp": np.array(alli),
    }


def scenario_B(S, man, mode="max"):
    """REALISTIC. enrol = right-index-cover (19), probe = right-index (12),
    impostor = right-middle (14). Disjoint sets, so no leakage possible."""
    idx = index_sets(man)
    T = idx["right-index-cover"]
    gen = [_fuse(S, T, p, mode) for p in idx["right-index"]]
    imp = [_fuse(S, T, p, mode) for p in idx["right-middle"]]
    e, et = eer(gen, imp)
    f, ft = far_at_frr(gen, imp)
    return {"dprime": dprime(gen, imp), "eer": e, "eer_t": et,
            "far10": f, "far10_t": ft,
            "gen": np.array(gen), "imp": np.array(imp)}


def summarise(tag, S, man, n_enroll=6, reps=20, mode="max", verbose=True):
    A = scenario_A(S, man, n_enroll, reps, mode=mode)
    B = scenario_B(S, man, mode=mode)
    if verbose:
        print(f"\n===== {tag}  (fuse={mode}, n_enroll={n_enroll}, reps={reps}) =====")
        print(f"  A POOLED    d'={A['dprime']:6.2f} (sd {A['dprime_sd']:.2f})  "
              f"EER={A['eer']*100:5.1f}% (sd {A['eer_sd']*100:.1f})  "
              f"FAR@10%FRR={A['far10']*100:5.1f}% (sd {A['far10_sd']*100:.1f})")
        print(f"     gen mean {A['gen'].mean():8.2f} sd {A['gen'].std():6.2f}   "
              f"imp mean {A['imp'].mean():8.2f} sd {A['imp'].std():6.2f}  "
              f"imp max {A['imp'].max():.1f}  gen min {A['gen'].min():.1f}")
        print(f"  B REALISTIC d'={B['dprime']:6.2f}                EER={B['eer']*100:5.1f}%"
              f"            FAR@10%FRR={B['far10']*100:5.1f}%")
        print(f"     gen {np.sort(B['gen'])[::-1].astype(int).tolist()}")
        print(f"     imp {np.sort(B['imp'])[::-1].astype(int).tolist()}")
    return A, B
