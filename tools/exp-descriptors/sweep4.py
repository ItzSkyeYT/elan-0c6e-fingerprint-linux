#!/usr/bin/env python3
"""Sweep driver: runs a named list of blockdesc configs and prints, for each,
the best (score variant, symmetrisation) by A d' + B d'."""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blockdesc as BD
import evaluate as EV
import par4

SWEEPS = {}


def sweep(name):
    def deco(fn):
        SWEEPS[name] = fn
        return fn
    return deco


@sweep("patch")
def s_patch():
    out = []
    for r, sub in [(6, 2), (8, 2), (10, 2), (12, 3), (14, 3), (16, 4)]:
        out.append((f"r={r},sub={sub}", dict(r=r, sub=sub)))
    return out


@sweep("ratio")
def s_ratio():
    return [(f"gap={g}", dict(ratio_gap=g))
            for g in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)]


@sweep("minsim")
def s_minsim():
    return [(f"minsim={s}", dict(min_sim=s))
            for s in (0.0, 0.35, 0.5, 0.6, 0.7, 0.8)]


@sweep("amb")
def s_amb():
    return [("no filter", dict(max_amb=1.01)),
            ("amb<=0.8", dict(max_amb=0.8)),
            ("amb<=0.7", dict(max_amb=0.7)),
            ("amb<=0.6", dict(max_amb=0.6)),
            ("best30", dict(keep_best=30)),
            ("best50", dict(keep_best=50))]


@sweep("geom")
def s_geom():
    return [("tol4", dict(tol_t=4.0, bin_t=4.0)),
            ("tol6", dict(tol_t=6.0, bin_t=6.0)),
            ("tol8", dict(tol_t=8.0, bin_t=8.0)),
            ("tol10", dict(tol_t=10.0, bin_t=8.0)),
            ("rotslack0", dict(rot_slack=0)),
            ("rots3", dict(rots=(-10.0, 0.0, 10.0))),
            ("rots7", dict(rots=(-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)))]


@sweep("misc")
def s_misc():
    return [("weight=margin", dict(weight="margin")),
            ("topk2", dict(topk=2)),
            ("qstep4", dict(q_step=4)),
            ("qstep8", dict(q_step=8)),
            ("enh=lcn", dict(enh="lcn")),
            ("coh0.5", dict(min_coh=0.5)),
            ("coh0", dict(min_coh=0.0))]


def main():
    names = sys.argv[1:] or ["patch"]
    base = {}
    if os.path.exists(os.path.join(HERE, "best4.json")):
        base = json.load(open(os.path.join(HERE, "best4.json")))
        print("base overrides from best4.json:", base)
    imgs, labels, _ = EV.load()
    for name in names:
        print(f"\n===== sweep {name} =====")
        for tag, over in SWEEPS[name]():
            cfg = dict(BD.DEFAULT)
            cfg.update(base)
            cfg.update(over)
            t0 = time.time()
            try:
                par4.run_cfg(cfg, imgs, labels, tag, top=2)
            except Exception as e:
                print(f"  {tag}: FAILED {e!r}")
            print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
