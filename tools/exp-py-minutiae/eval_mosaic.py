#!/usr/bin/env python3
"""Evaluate the mosaic template under both protocols.

Scenario A: 12+ random 6-image enrolment subsets from the 31 pooled genuine; a
            mosaic is built from EACH subset, and every genuine capture not in
            that subset plus all 14 impostors are probed against it.
Scenario B: one mosaic from the 19 right-index-cover captures; probes are the
            12 right-index captures and the 14 right-middle captures.

No probe is ever part of the mosaic it is scored against.
"""
import math, sys, time
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr
import quality as Q
import mosaic as MO

ORDER = ["right-index", "right-index-cover", "right-middle"]


def prep_all(ds):
    ims, wts, labels = [], [], []
    for lbl in ORDER:
        for nm, im in ds[lbl]:
            q, n = Q.quality(im)
            ims.append(np.asarray(n, np.float64))
            wts.append(np.asarray(q, np.float64))
            labels.append(lbl)
    return ims, wts, labels


def rep(tag, g, i):
    d = dprime(g, i); e, _ = eer(g, i); f, _ = far_at_frr(g, i, 0.10)
    print(f"  {tag:26s} d'={d:6.2f}  EER={e*100:5.1f}%  FAR@10%FRR={f*100:5.1f}%"
          f"   gen {np.mean(g):.3f}+-{np.std(g):.3f}  imp {np.mean(i):.3f}+-{np.std(i):.3f}")
    return d, e, f


def main(pad_x=70, pad_y=24, trials=12, n_enroll=6, seed=0):
    ds = load_ds()
    ims, wts, labels = prep_all(ds)
    gi = [k for k, l in enumerate(labels) if l != "right-middle"]
    ii = [k for k, l in enumerate(labels) if l == "right-middle"]
    cov = [k for k, l in enumerate(labels) if l == "right-index-cover"]
    idx = [k for k, l in enumerate(labels) if l == "right-index"]

    # ---------------- sanity: a mosaic must score its OWN members very high
    t0 = time.time()
    mo, order = MO.build([ims[k] for k in cov[:6]], [wts[k] for k in cov[:6]],
                         pad_x=pad_x, pad_y=pad_y)
    self_s = [MO.score_probe(mo, ims[k], wts[k]) for k in cov[:6]]
    out_s = [MO.score_probe(mo, ims[k], wts[k]) for k in ii[:6]]
    print(f"SANITY  mosaic of 6, members score {np.mean(self_s):.3f} "
          f"(min {np.min(self_s):.3f}); impostors {np.mean(out_s):.3f}   "
          f"[{time.time()-t0:.1f}s to build]")
    cov_frac = (mo.sw > 1e-6).mean()
    print(f"        canvas {mo.CH}x{mo.CW}, filled {cov_frac*100:.0f}%, "
          f"placed {mo.placed}, add order scores "
          f"{[round(o[1],2) for o in order]}")

    # ---------------- scenario B
    print("\n== scenario B: enrol = 19 cover, probe = 12 index, impostor = 14 middle ==")
    t0 = time.time()
    moB, orderB = MO.build([ims[k] for k in cov], [wts[k] for k in cov],
                           pad_x=pad_x, pad_y=pad_y)
    print(f"  mosaic built from {moB.placed}/19 in {time.time()-t0:.1f}s, "
          f"canvas filled {(moB.sw>1e-6).mean()*100:.0f}%")
    g = [MO.score_probe(moB, ims[k], wts[k]) for k in idx]
    m = [MO.score_probe(moB, ims[k], wts[k]) for k in ii]
    rB = rep("B mosaic", g, m)

    # ---------------- scenario A
    print(f"\n== scenario A: pooled, {trials} random {n_enroll}-image enrolments ==")
    rng = np.random.default_rng(seed)
    GA, IA, per = [], [], []
    t0 = time.time()
    for t in range(trials):
        T = list(rng.choice(gi, size=n_enroll, replace=False))
        probes = [k for k in gi if k not in T]
        mo, _ = MO.build([ims[k] for k in T], [wts[k] for k in T],
                         pad_x=pad_x, pad_y=pad_y)
        gg = [MO.score_probe(mo, ims[k], wts[k]) for k in probes]
        mm = [MO.score_probe(mo, ims[k], wts[k]) for k in ii]
        GA += gg; IA += mm
        per.append(dprime(gg, mm))
    print(f"  ({time.time()-t0:.1f}s)")
    rA = rep(f"A mosaic n={n_enroll}", GA, IA)
    print(f"  per-trial d' {np.mean(per):.2f} +- {np.std(per):.2f}")
    return rA, rB


if __name__ == "__main__":
    main()
