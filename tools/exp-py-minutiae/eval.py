#!/usr/bin/env python3
"""Evaluation harness for the py-minutiae matcher.

Protocol (exactly as specified):
  A) POOLED   genuine = right-index + right-index-cover (31), impostor =
              right-middle (14).  Template-set vs probe, leave-one-out over
              genuine, averaged over >= 8 RANDOM template subsets.
  B) REALISTIC enrol = right-index-cover (19), probe = right-index (12),
              impostor = right-middle (14).

Never compares an image with itself; never uses a probe as its own template.
"""
import argparse
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
import minutiae as M
from common import load_ds, dprime, eer, far_at_frr

GEN_LABELS = ["right-index", "right-index-cover"]
IMP_LABEL = "right-middle"


def build(ds, cfg, verbose=True):
    """Returns (names, labels, templates) for all 45 images in a fixed order:
    right-index first, then right-index-cover, then right-middle."""
    names, labels, tmpl = [], [], []
    order = ["right-index", "right-index-cover", "right-middle"]
    t0 = time.time()
    for lbl in order:
        for nm, img in ds[lbl]:
            names.append(nm)
            labels.append(lbl)
            tmpl.append(M.make_template(img, cfg))
    if verbose:
        counts = np.array([t.n for t in tmpl])
        print(f"  extraction {time.time()-t0:.1f}s   minutiae/image "
              f"mean {counts.mean():.1f}  min {counts.min()}  max {counts.max()}")
        for lbl in order:
            c = np.array([t.n for t, l in zip(tmpl, labels) if l == lbl])
            print(f"    {lbl:20s} mean {c.mean():5.1f}")
    return names, labels, tmpl


def score_matrix(tmpl, mcfg, verbose=True):
    n = len(tmpl)
    S = np.zeros((n, n), np.float64)
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            s = M.match(tmpl[i], tmpl[j], **mcfg)
            S[i, j] = S[j, i] = s
    if verbose:
        el = time.time() - t0
        print(f"  matching   {el:.1f}s  ({el/(n*(n-1)/2)*1000:.2f} ms/pair)")
    np.fill_diagonal(S, np.nan)          # self-comparison is never used
    return S


def scenario_A(S, labels, n_enroll=6, trials=12, seed=0):
    gi = [k for k, l in enumerate(labels) if l in GEN_LABELS]
    ii = [k for k, l in enumerate(labels) if l == IMP_LABEL]
    rng = np.random.default_rng(seed)
    gen_all, imp_all, per_trial = [], [], []
    for _ in range(trials):
        T = list(rng.choice(gi, size=n_enroll, replace=False))
        probes = [k for k in gi if k not in T]
        g = [max(S[t, p] for t in T) for p in probes]
        m = [max(S[t, p] for t in T) for p in ii]
        gen_all += g
        imp_all += m
        per_trial.append(dprime(g, m))
    return gen_all, imp_all, per_trial


def scenario_B(S, labels):
    enrol = [k for k, l in enumerate(labels) if l == "right-index-cover"]
    probe = [k for k, l in enumerate(labels) if l == "right-index"]
    imp = [k for k, l in enumerate(labels) if l == IMP_LABEL]
    g = [max(S[t, p] for t in enrol) for p in probe]
    m = [max(S[t, p] for t in enrol) for p in imp]
    return g, m


def report(tag, g, m, extra=""):
    d = dprime(g, m)
    e, et = eer(g, m)
    f10, t10 = far_at_frr(g, m, 0.10)
    print(f"  {tag:14s} d'={d:6.2f}  EER={e*100:5.1f}%  FAR@10%FRR={f10*100:5.1f}%"
          f"   gen {np.mean(g):.3f}+-{np.std(g):.3f}  imp {np.mean(m):.3f}+-{np.std(m):.3f} {extra}")
    return dict(dprime=d, eer=e, far10=f10)


def sanity(tmpl, mcfg):
    """A matcher must score at or near its maximum comparing an image with
    itself, and near zero comparing to an empty template."""
    ok = True
    ss, cross = [], []
    for i in range(0, len(tmpl), 5):
        ss.append(M.match(tmpl[i], tmpl[i], **mcfg))
    for i in range(0, len(tmpl) - 1, 5):
        cross.append(M.match(tmpl[i], tmpl[i + 1], **mcfg))
    print(f"  SANITY self-match mean {np.mean(ss):.3f} min {np.min(ss):.3f}   "
          f"cross-match mean {np.mean(cross):.3f}")
    if np.mean(ss) <= np.mean(cross) * 1.2:
        print("  SANITY FAIL: self-match is not clearly the strongest")
        ok = False
    return ok


def run(cfg, mcfg, n_enroll=6, trials=12, tag="", verbose=True, ds=None):
    ds = ds if ds is not None else load_ds()
    print(f"\n########## {tag} ##########")
    print(f"  cfg  {cfg}")
    print(f"  mcfg {mcfg}")
    names, labels, tmpl = build(ds, cfg, verbose)
    S = score_matrix(tmpl, mcfg, verbose)
    if verbose:
        sanity(tmpl, mcfg)
    ga, ia, pt = scenario_A(S, labels, n_enroll, trials)
    ra = report(f"A pooled n={n_enroll}", ga, ia,
                f"  per-trial d' {np.mean(pt):.2f}+-{np.std(pt):.2f}")
    gb, ib = scenario_B(S, labels)
    rb = report("B realistic", gb, ib)
    return ra, rb, S, labels, tmpl


if __name__ == "__main__":
    cfg = {}
    mcfg = dict(rot_max=25.0, rot_bin=6.0, tr_bin=8.0, pos_tol=8.0,
                dir_tol=math.radians(20.0), top_k=6, min_denom=6.0, score="ov")
    run(cfg, mcfg, tag="default")
