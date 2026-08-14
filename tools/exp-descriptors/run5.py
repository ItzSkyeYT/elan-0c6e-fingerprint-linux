#!/usr/bin/env python3
"""Run blockncc (v5) configs through the two mandated protocols."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blockncc as BN, evaluate as EV, par4

if __name__ == "__main__":
    imgs, labels, _ = EV.load()
    cfg = dict(BN.DEFAULT)
    tag = "v5"
    for kv in sys.argv[1:]:
        if kv.startswith("tag="):
            tag = kv[4:]; continue
        k, v = kv.split("=", 1); cfg[k] = eval(v)
    print("cfg:", {k: v for k, v in cfg.items() if k != "variants"})
    t0 = time.time()
    Q, rows, best = par4.run_cfg(cfg, imgs, labels, tag, nproc=8, top=9, mod=BN)
    np.savez(os.path.join(HERE, f"Q5_{tag}.npz"), labels=labels,
             **{k: v for k, v in Q.items()})
    print(f"  total {time.time()-t0:.0f}s")
