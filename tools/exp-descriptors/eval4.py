#!/usr/bin/env python3
"""Evaluation driver for blockdesc (v4).

Computes the full 45x45 DIRECTED score matrix for every score variant in one
pass, then reports both mandated protocols for each variant and each
symmetrisation.  The diagonal is used only for the self-match sanity check and
never enters a reported number.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import blockdesc as BD
import evaluate as EV


def matrices(cfg, imgs, verbose=True, pairs=None):
    t0 = time.time()
    T = [BD.make_template(im, cfg) for im in imgs]
    n = len(T)
    variants = list(cfg["variants"])
    Q = {v: np.zeros((n, n), np.float32) for v in variants}
    for i in range(n):
        for j in range(n):
            if pairs is not None and not pairs[i, j]:
                continue
            s = BD.score_one(T[i], T[j], cfg)
            for v in variants:
                Q[v][i, j] = s[v]
    if verbose:
        nq = [t.nq for t in T]
        print(f"  query patches/image: mean {np.mean(nq):.1f} "
              f"[{min(nq)}..{max(nq)}]   db positions {T[0].ny*T[0].nx}   "
              f"({time.time()-t0:.0f}s)")
    return Q, T


def symmetrise(Q, how):
    if how == "max":
        return np.maximum(Q, Q.T)
    if how == "mean":
        return 0.5 * (Q + Q.T)
    if how == "min":
        return np.minimum(Q, Q.T)
    return Q.T                      # S[t, p] = Q[p, t]: the probe is the query


def report(Q, labels, tag="", syms=("none", "max", "mean", "min"),
           n_enroll=6, trials=16, quiet=False):
    out = {}
    for sym in syms:
        S = symmetrise(Q, sym)
        a, asd = EV.protocol_a(S, labels, n_enroll, trials)
        b, g, i = EV.protocol_b(S, labels)
        out[sym] = (a, asd, b)
        if not quiet:
            print(f"  {tag:<8s}[{sym:<4s}] A: d'={a['dprime']:5.2f}+/-{asd['dprime']:.2f} "
                  f"EER={a['eer']*100:5.1f}% FAR@10={a['far10']*100:5.1f}%  |  "
                  f"B: d'={b['dprime']:5.2f} EER={b['eer']*100:5.1f}% "
                  f"FAR@10={b['far10']*100:5.1f}%")
    return out


def sanity(Q, tag):
    d = np.diag(Q)
    off = Q.copy()
    np.fill_diagonal(off, -np.inf)
    bad = int((d < off.max(axis=1)).sum())
    print(f"  sanity[{tag}]: self mean {d.mean():.3f} (min {d.min():.3f}); "
          f"best-other mean {off.max(axis=1).mean():.3f}; "
          f"{bad} image(s) fail to score max against themselves")


def run(cfg, n_enroll=6, trials=16, quiet=False):
    imgs, labels, names = EV.load()
    Q, T = matrices(cfg, imgs, verbose=not quiet)
    res = {}
    for v in cfg["variants"]:
        if not quiet:
            sanity(Q[v], v)
        res[v] = report(Q[v], labels, tag=v, quiet=quiet)
    return Q, res, labels


if __name__ == "__main__":
    cfg = dict(BD.DEFAULT)
    for kv in sys.argv[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = eval(v)
    print("cfg:", cfg)
    Q, res, labels = run(cfg)
    np.save(os.path.join(HERE, "Q_v4.npy"),
            np.stack([Q[v] for v in cfg["variants"]]))
