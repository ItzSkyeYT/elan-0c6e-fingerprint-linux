#!/usr/bin/env python3
"""Sweep extraction and matching parameters.  Templates are cached per
extraction config so many match configs cost almost nothing extra."""
import itertools
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
import minutiae as M
from common import load_ds, dprime, eer, far_at_frr
from eval import build, scenario_A, scenario_B, GEN_LABELS, IMP_LABEL
import stability

DS = load_ds()


def eval_cfg(cfg, mcfgs, n_enroll=6, trials=12, label=""):
    names, labels, tmpl = build(DS, cfg, verbose=False)
    counts = np.array([t.n for t in tmpl])
    rows = []
    for mc in mcfgs:
        n = len(tmpl)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                S[i, j] = S[j, i] = M.match(tmpl[i], tmpl[j], **mc)
        np.fill_diagonal(S, np.nan)
        ga, ia, pt = scenario_A(S, labels, n_enroll, trials)
        gb, ib = scenario_B(S, labels)
        rows.append(dict(cfg=cfg, mcfg=mc, nmin=counts.mean(),
                         dA=dprime(ga, ia), eA=eer(ga, ia)[0],
                         fA=far_at_frr(ga, ia)[0],
                         dB=dprime(gb, ib), eB=eer(gb, ib)[0],
                         fB=far_at_frr(gb, ib)[0],
                         label=label))
    return rows


def show(r):
    mk = {k: v for k, v in r['mcfg'].items() if k in ('score', 'pos_tol', 'min_denom', 'dir_tol')}
    mk['dir_tol'] = round(math.degrees(mk.get('dir_tol', 0)), 0)
    print(f"  {r['label']:34s} n={r['nmin']:4.1f} | A d'={r['dA']:5.2f} EER={r['eA']*100:4.1f}% "
          f"FAR10={r['fA']*100:5.1f}% | B d'={r['dB']:5.2f} EER={r['eB']*100:4.1f}% "
          f"FAR10={r['fB']*100:5.1f}% | {mk}")


BASE_M = dict(rot_max=25.0, rot_bin=6.0, tr_bin=8.0, pos_tol=8.0,
              dir_tol=math.radians(20.0), top_k=6, min_denom=6.0, score="ov")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "extract"
    all_rows = []
    if which == "extract":
        cfgs = [
            ({}, "default merge=9 border=6"),
            ({"merge_dist": 5.0}, "merge=5"),
            ({"merge_dist": 3.0}, "merge=3"),
            ({"merge_dist": 0.0}, "merge=off"),
            ({"merge_dist": 0.0, "border": 3}, "merge=off border=3"),
            ({"merge_dist": 0.0, "border": 3, "spur_len": 4, "min_ridge": 6},
             "merge=off border=3 spur=4"),
            ({"merge_dist": 0.0, "border": 3, "spur_len": 14, "min_ridge": 18},
             "merge=off border=3 spur=14"),
            ({"merge_dist": 0.0, "border": 3, "bin_k": 5}, "merge=off b3 bin_k=5"),
            ({"merge_dist": 0.0, "border": 3, "bin_k": 11}, "merge=off b3 bin_k=11"),
            ({"merge_dist": 0.0, "border": 3, "gsx": 6.0, "gsy": 4.0},
             "merge=off b3 gabor sx=6"),
            ({"merge_dist": 0.0, "border": 3, "gsx": 6.0, "gsy": 5.0, "ksize": 15},
             "merge=off b3 gabor 15/6/5"),
            ({"merge_dist": 0.0, "border": 3, "est_freq": True},
             "merge=off b3 est_freq"),
        ]
        for cfg, lbl in cfgs:
            t0 = time.time()
            rows = eval_cfg(cfg, [BASE_M], label=lbl)
            st = stability.main(cfg) if False else None
            for r in rows:
                show(r)
            all_rows += rows
    elif which == "match":
        cfg = eval(sys.argv[2]) if len(sys.argv) > 2 else {"merge_dist": 0.0, "border": 3}
        mcfgs = []
        for score in ("ov", "ov2", "count", "hyb"):
            for pos_tol in (6.0, 10.0, 14.0):
                for dt in (15.0, 25.0, 40.0):
                    m = dict(BASE_M)
                    m.update(score=score, pos_tol=pos_tol, dir_tol=math.radians(dt))
                    mcfgs.append(m)
        rows = eval_cfg(cfg, mcfgs, label="m")
        rows.sort(key=lambda r: -(r['dA'] + r['dB']))
        for r in rows[:20]:
            show(r)
        all_rows = rows
    return all_rows


if __name__ == "__main__":
    main()
