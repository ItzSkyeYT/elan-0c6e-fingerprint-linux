#!/usr/bin/env python3
"""Parallel score-matrix computation for blockdesc, plus a compact sweep driver.

fork() lets the workers inherit the templates by copy-on-write, so nothing large
is pickled per task.
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import blockdesc as BD
import evaluate as EV

_T = None
_CFG = None
_MOD = None


def _init(T, cfg, mod):
    global _T, _CFG, _MOD
    _T, _CFG, _MOD = T, cfg, mod


def _row(i):
    return i, [_MOD.score_one(_T[i], _T[j], _CFG) for j in range(len(_T))]


def matrices(cfg, imgs, nproc=8, verbose=True, mod=BD):
    t0 = time.time()
    T = [mod.make_template(im, cfg) for im in imgs]
    n = len(T)
    variants = list(cfg["variants"])
    Q = {v: np.zeros((n, n), np.float32) for v in variants}
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, initializer=_init, initargs=(T, cfg, mod)) as pool:
        for i, row in pool.imap_unordered(_row, range(n)):
            for j, s in enumerate(row):
                for v in variants:
                    Q[v][i, j] = s[v]
    if verbose:
        nq = [t.nq for t in T]
        print(f"  patches/image mean {np.mean(nq):.1f} [{min(nq)}..{max(nq)}]  "
              f"dim {T[0].qd.shape[2]}  dbpos {T[0].ny*T[0].nx}  "
              f"({time.time()-t0:.0f}s)")
    return Q, T


def summarise(Q, labels, variants, syms=("none", "max", "mean", "min"),
              n_enroll=6, trials=16):
    rows = []
    for v in variants:
        for sym in syms:
            S = (np.maximum(Q[v], Q[v].T) if sym == "max" else
                 0.5 * (Q[v] + Q[v].T) if sym == "mean" else
                 np.minimum(Q[v], Q[v].T) if sym == "min" else Q[v].T)
            a, asd = EV.protocol_a(S, labels, n_enroll, trials)
            b, _, _ = EV.protocol_b(S, labels)
            rows.append(dict(variant=v, sym=sym, A=a, Asd=asd, B=b,
                             obj=a["dprime"] + b["dprime"]))
    return rows


def show(rows, tag="", top=6):
    rows = sorted(rows, key=lambda r: -r["obj"])
    for r in rows[:top]:
        a, b = r["A"], r["B"]
        print(f"  {tag:<22s} {r['variant']:<6s}/{r['sym']:<4s}  "
              f"A d'={a['dprime']:5.2f} EER={a['eer']*100:5.1f}% "
              f"FAR10={a['far10']*100:5.1f}%  |  "
              f"B d'={b['dprime']:5.2f} EER={b['eer']*100:5.1f}% "
              f"FAR10={b['far10']*100:5.1f}%")
    return rows[0]


def sanity(Q, variants):
    msg = []
    for v in variants:
        d = np.diag(Q[v]); off = Q[v].copy(); np.fill_diagonal(off, -np.inf)
        msg.append(f"{v}:{int((d < off.max(axis=1)).sum())}")
    print("  self-match failures per variant -> " + " ".join(msg))


def run_cfg(cfg, imgs, labels, tag="", nproc=8, top=4, mod=BD):
    Q, T = matrices(cfg, imgs, nproc, mod=mod)
    sanity(Q, cfg["variants"])
    rows = summarise(Q, labels, cfg["variants"])
    best = show(rows, tag, top)
    return Q, rows, best


if __name__ == "__main__":
    imgs, labels, names = EV.load()
    cfg = dict(BD.DEFAULT)
    for kv in sys.argv[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = eval(v)
    print("cfg:", cfg)
    run_cfg(cfg, imgs, labels, "cli", top=8)
