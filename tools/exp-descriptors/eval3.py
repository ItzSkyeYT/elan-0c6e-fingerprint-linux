#!/usr/bin/env python3
"""Evaluation driver for desc3.  Computes the full asymmetric 45x45 directed
score matrix Q[q, d] = score(query=q, database=d), then reports both mandated
protocols with score(template t, probe p) = Q[p, t] (the probe is the query;
the enrolled image holds the dense database), optionally symmetrised."""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import desc3
import evaluate as EV


def matrix(cfg, imgs, verbose=True):
    t0 = time.time()
    T = [desc3.make_template(im, cfg) for im in imgs]
    n = len(T)
    Q = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            Q[i, j] = desc3._score_one(T[i], T[j], cfg)
    if verbose:
        print(f"  nq={T[0].nq} ndb={T[0].ndb}  ({time.time()-t0:.0f}s)")
    return Q, T


def report(Q, labels, sym="max", n_enroll=6, trials=16, tag="", quiet=False):
    if sym == "max":
        S = np.maximum(Q, Q.T)
    elif sym == "mean":
        S = 0.5 * (Q + Q.T)
    elif sym == "min":
        S = np.minimum(Q, Q.T)
    else:
        S = Q.T          # S[t, p] = Q[p, t]
    a, asd = EV.protocol_a(S, labels, n_enroll, trials)
    b, g, im = EV.protocol_b(S, labels)
    if not quiet:
        print(f"  {tag}[{sym}] A: d'={a['dprime']:.2f}+/-{asd['dprime']:.2f} "
              f"EER={a['eer']*100:.1f}% FAR@10={a['far10']*100:.1f}%   |   "
              f"B: d'={b['dprime']:.2f} EER={b['eer']*100:.1f}% "
              f"FAR@10={b['far10']*100:.1f}%")
    return a, b, S


def run(cfg, tag="", n_enroll=6, trials=16, sanity=True):
    imgs, labels, names = EV.load()
    Q, T = matrix(cfg, imgs)
    if sanity:
        d = np.diag(Q)
        off = Q.copy(); np.fill_diagonal(off, -np.inf)
        bad = int((d < off.max(axis=1)).sum())
        print(f"  sanity: self {d.mean():.1f} (min {d.min():.1f}), "
              f"best-other {off.max(axis=1).mean():.1f}; "
              f"{bad} image(s) beat their own self-score")
    out = {}
    for sym in ("none", "max", "mean", "min"):
        out[sym] = report(Q, labels, sym, n_enroll, trials, tag)
    return Q, out


if __name__ == "__main__":
    cfg = dict(desc3.DEFAULT)
    for kv in sys.argv[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = eval(v)
    print("cfg:", cfg)
    run(cfg)
