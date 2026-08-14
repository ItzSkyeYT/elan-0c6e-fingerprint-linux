#!/usr/bin/env python3
"""Sweep blockncc (v5) configs. Saves every score matrix so variants can be
re-analysed offline without recomputing."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blockncc as BN, evaluate as EV, par4

GRIDS = {
 "elastic": [(f"el{e}", dict(elastic=e)) for e in (0, 1, 2, 3, 4)],
 "patch":   [("r6s2", dict(r=6, sub=2)), ("r8s2", dict(r=8, sub=2)),
             ("r10s2", dict(r=10, sub=2)), ("r12s3", dict(r=12, sub=3)),
             ("r14s3", dict(r=14, sub=3)), ("r8s1", dict(r=8, sub=1))],
 "grid":    [(f"q{q}", dict(q_step=q)) for q in (3, 4, 5, 7)],
 "rot":     [("rot3", dict(rots=(-10.0, 0.0, 10.0))),
             ("rot5", dict(rots=(-12.0, -6.0, 0.0, 6.0, 12.0))),
             ("rot7", dict(rots=(-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0))),
             ("rot9", dict(rots=(-16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0))),
             ("rot1", dict(rots=(0.0,)))],
 "enh":     [("lcn", dict(enh="lcn")), ("gabor", dict(enh="gabor")),
             ("bp", dict(enh="bp"))],
 "range":   [("wide", dict(max_tx=70, max_ty=26)),
             ("narrow", dict(max_tx=35, max_ty=16)),
             ("mid", dict(max_tx=55, max_ty=22))],
 "coh":     [("coh0", dict(min_coh=0.0)), ("coh.3", dict(min_coh=0.3)),
             ("coh.5", dict(min_coh=0.5))],
}

if __name__ == "__main__":
    base = dict(BN.DEFAULT); base.update(enh="lcn", q_step=5)
    imgs, labels, _ = EV.load()
    for name in sys.argv[1:]:
        print(f"\n===== {name} =====", flush=True)
        for tag, over in GRIDS[name]:
            cfg = dict(base); cfg.update(over)
            t0 = time.time()
            Q, rows, best = par4.run_cfg(cfg, imgs, labels, f"{name}:{tag}",
                                         nproc=15, top=3, mod=BN)
            np.savez(os.path.join(HERE, f"Q5_{name}_{tag}.npz"), labels=labels, **Q)
            print(f"  ({time.time()-t0:.0f}s)", flush=True)
