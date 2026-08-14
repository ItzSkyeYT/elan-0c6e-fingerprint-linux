#!/usr/bin/env python3
"""Run the two mandated evaluation protocols over a full pairwise score matrix.

Protocol A (POOLED)   genuine = right-index + right-index-cover (31)
                      impostor = right-middle (14)
                      template-set vs probe, leave-one-out over genuine,
                      averaged over N_TRIALS random template subsets.
Protocol B (REALISTIC) enrol = right-index-cover (19)
                      probe = right-index (12), impostor = right-middle (14)

The score matrix is computed once (45x45, self-pairs on the diagonal are only
ever used for the sanity check, never for a reported number).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desc as D

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LBL = ["right-index", "right-index-cover", "right-middle"]


def load():
    ds = D.load_dataset(DATASET)
    imgs, labels, names = [], [], []
    for l in LBL:
        for n, im in ds[l]:
            imgs.append(im)
            labels.append(l)
            names.append(f"{l}/{n}")
    return imgs, np.array(labels), names


def score_matrix(imgs, cfg, verbose=True):
    t0 = time.time()
    tmpls = [D.make_template(im, cfg) for im in imgs]
    nkp = [t.n_kp for t in tmpls]
    n = len(imgs)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            s = D.match(tmpls[i], tmpls[j], cfg)
            S[i, j] = s
            S[j, i] = s if cfg.get("symmetric", True) else D.match(tmpls[j], tmpls[i], cfg)
    if verbose:
        print(f"  keypoints/image: mean {np.mean(nkp):.1f} "
              f"min {min(nkp)} max {max(nkp)}   ({time.time()-t0:.1f}s)")
    return S, nkp


def metrics(gen, imp):
    g, i = np.asarray(gen, float), np.asarray(imp, float)
    return dict(dprime=D.dprime(g, i), eer=D.eer(g, i)[0],
                far10=D.far_at_frr(g, i, 0.10)[0])


def protocol_a(S, labels, n_enroll=6, n_trials=16, seed=0):
    gidx = np.where(labels != "right-middle")[0]
    iidx = np.where(labels == "right-middle")[0]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        perm = rng.permutation(gidx)
        gen = []
        for p in gidx:
            tmpl = [t for t in perm if t != p][:n_enroll]
            gen.append(max(S[t, p] for t in tmpl))
        tmpl = list(perm[:n_enroll])
        imp = [max(S[t, p] for t in tmpl) for p in iidx]
        out.append(metrics(gen, imp))
    keys = ("dprime", "eer", "far10")
    return ({k: float(np.mean([o[k] for o in out])) for k in keys},
            {k: float(np.std([o[k] for o in out])) for k in keys})


def protocol_b(S, labels):
    enrol = np.where(labels == "right-index-cover")[0]
    probe = np.where(labels == "right-index")[0]
    imp = np.where(labels == "right-middle")[0]
    g = [max(S[t, p] for t in enrol) for p in probe]
    i = [max(S[t, p] for t in imp_p) for imp_p in [enrol]] if False else \
        [max(S[t, p] for t in enrol) for p in imp]
    return metrics(g, i), g, i


def sanity(S, labels, names, nkp):
    n = len(labels)
    diag = np.diag(S)
    off = S.copy()
    np.fill_diagonal(off, -np.inf)
    bad = [names[i] for i in range(n) if diag[i] < off[i].max()]
    print(f"  sanity: self-score mean {diag.mean():.2f} "
          f"(min {diag.min():.2f}); best non-self mean {off.max(axis=1).mean():.2f}")
    if bad:
        print(f"  !! {len(bad)} images score higher against another image than "
              f"against themselves: {bad[:3]}")
    else:
        print("  sanity OK: every image scores maximum against itself")


def run(cfg, tag="", n_enroll=6, n_trials=16, quiet=False):
    imgs, labels, names = load()
    S, nkp = score_matrix(imgs, cfg, verbose=not quiet)
    if not quiet:
        sanity(S, labels, names, nkp)
    a, asd = protocol_a(S, labels, n_enroll, n_trials)
    b, gb, ib = protocol_b(S, labels)
    if not quiet:
        print(f"\n  [A pooled, {n_enroll} templates, {n_trials} random subsets]")
        print(f"    d' = {a['dprime']:.2f} +/- {asd['dprime']:.2f}   "
              f"EER = {a['eer']*100:.1f}% +/- {asd['eer']*100:.1f}   "
              f"FAR@10%FRR = {a['far10']*100:.1f}% +/- {asd['far10']*100:.1f}")
        print(f"  [B realistic, enrol=19 cover, probe=12 index]")
        print(f"    d' = {b['dprime']:.2f}   EER = {b['eer']*100:.1f}%   "
              f"FAR@10%FRR = {b['far10']*100:.1f}%")
        print(f"    genuine  {np.mean(gb):.2f}+/-{np.std(gb):.2f} "
              f"[{min(gb):.2f}..{max(gb):.2f}]")
        print(f"    impostor {np.mean(ib):.2f}+/-{np.std(ib):.2f} "
              f"[{min(ib):.2f}..{max(ib):.2f}]")
    return dict(tag=tag, A=a, A_sd=asd, B=b, nkp_mean=float(np.mean(nkp)),
                S=S, labels=labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[],
                    help="cfg override k=v (python literal)")
    ap.add_argument("--n-enroll", type=int, default=6)
    ap.add_argument("--trials", type=int, default=16)
    args = ap.parse_args()
    cfg = dict(D.DEFAULT)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = eval(v)
    print("cfg:", {k: v for k, v in cfg.items()})
    run(cfg, n_enroll=args.n_enroll, n_trials=args.trials)
