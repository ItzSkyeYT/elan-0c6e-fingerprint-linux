#!/usr/bin/env python3
"""
Template-set POOLING rule sweep.

`max over templates` is the obvious rule but not obviously the best one: a
genuine probe usually resembles several enrolled captures, while an impostor
that scores high does so against one template by luck.  Averaging the top-k
scores should therefore suppress impostor flukes.  This is orthogonal to the
matcher and applies to every cached matrix.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                      # noqa
from run import load_flat, CACHE          # noqa


def topk(v, k):
    v = np.sort(np.asarray(v))[::-1]
    return float(v[:k].mean())


def eval_pooled_k(M, gen_idx, imp_idx, k, n_enroll=6, trials=32, seed=1234):
    rng = np.random.default_rng(seed)
    gen_idx = np.asarray(gen_idx); imp_idx = np.asarray(imp_idx)
    out = []
    for _ in range(trials):
        T = rng.choice(gen_idx, size=n_enroll, replace=False)
        Ts = set(T.tolist())
        probes = [p for p in gen_idx.tolist() if p not in Ts]
        gen = [topk(M[T, p], k) for p in probes]
        imp = [topk(M[T, q], k) for q in imp_idx.tolist()]
        out.append((dprime(gen, imp), eer(gen, imp)[0],
                    far_at_frr(gen, imp, 0.10)[0]))
    return np.array(out).mean(0)


def eval_real_k(M, tmpl, probe, imp, k):
    T = np.asarray(tmpl)
    gen = [topk(M[T, p], k) for p in probe]
    im = [topk(M[T, q], k) for q in imp]
    return dprime(gen, im), eer(gen, im)[0], far_at_frr(gen, im, 0.10)[0]


def main():
    imgs, labels = load_flat()
    ii = np.where(labels == "right-index")[0]
    ic = np.where(labels == "right-index-cover")[0]
    im = np.where(labels == "right-middle")[0]
    pool = np.concatenate([ii, ic])

    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(CACHE)
                                   if f.endswith(".npy"))
    print(f"{'matrix':<22}{'k':>3}{'A d':>7}{'A EER':>8}{'A FAR10':>9}"
          f"{'B d':>7}{'B EER':>8}{'B FAR10':>9}")
    for nm in names:
        M = np.load(os.path.join(CACHE, nm + ".npy"))
        if M.ndim == 3:                      # joint sweep: take the best w
            M = M[4]
            nm = nm + "[w0.4]"
        for k in (1, 2, 3, 4, 6):
            a = eval_pooled_k(M, pool, im, k)
            b = eval_real_k(M, ic, ii, im, k)
            print(f"{nm:<22}{k:3d}{a[0]:7.2f}{a[1]*100:7.1f}%{a[2]*100:8.1f}%"
                  f"{b[0]:7.2f}{b[1]*100:7.1f}%{b[2]*100:8.1f}%")
        print()


if __name__ == "__main__":
    main()
