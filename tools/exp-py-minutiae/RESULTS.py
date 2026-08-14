#!/usr/bin/env python3
"""THE headline measurement. Run this to reproduce every number reported.

  python3 RESULTS.py

Reports, under the two required protocols:
  * the LCN + rotation + NCC baseline exactly as previously measured
  * the same with out-of-frame pixels invalidated after rotation (the one
    change that measurably helps)
  * the pure minutiae matcher (this experiment's assignment)
  * the quality-gated, descriptor-augmented minutiae matcher
  * the minutiae ceiling analysis

Sanity checks run first: a matcher must score at its maximum comparing an
image with itself, and the genuine/impostor split is printed so it can be
eyeballed.
"""
import math, sys, time
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
from common import load_ds, dprime, eer, far_at_frr
import minutiae as M
import quality as Q
import wncc
from fastncc import ncc_all_shifts

ORDER = ["right-index", "right-index-cover", "right-middle"]
GEN = ["right-index", "right-index-cover"]
IMP = "right-middle"
ROTS = list(range(-12, 13, 4))
W, H = 150, 52


# ------------------------------------------------------------- protocols --

def scenario_A(S, labels, n_enroll=6, trials=16, seed=0):
    gi = [k for k, l in enumerate(labels) if l in GEN]
    ii = [k for k, l in enumerate(labels) if l == IMP]
    rng = np.random.default_rng(seed)
    g, m, per = [], [], []
    for _ in range(trials):
        T = list(rng.choice(gi, size=n_enroll, replace=False))
        probes = [k for k in gi if k not in T]
        gg = [max(S[t, p] for t in T) for p in probes]
        mm = [max(S[t, p] for t in T) for p in ii]
        assert all(p not in T for p in probes)          # no probe is its own template
        g += gg; m += mm; per.append(dprime(gg, mm))
    return g, m, per


def scenario_B(S, labels):
    enrol = [k for k, l in enumerate(labels) if l == "right-index-cover"]
    probe = [k for k, l in enumerate(labels) if l == "right-index"]
    imp = [k for k, l in enumerate(labels) if l == IMP]
    return ([max(S[t, p] for t in enrol) for p in probe],
            [max(S[t, p] for t in enrol) for p in imp])


def report(tag, S, labels, n_enroll=6, trials=16):
    ga, ia, per = scenario_A(S, labels, n_enroll, trials)
    gb, ib = scenario_B(S, labels)
    print(f"  {tag}")
    print(f"    A pooled (n={n_enroll}, {trials} random subsets)  "
          f"d'={dprime(ga,ia):6.2f}  EER={eer(ga,ia)[0]*100:5.1f}%  "
          f"FAR@10%FRR={far_at_frr(ga,ia)[0]*100:5.1f}%"
          f"   [per-trial d' {np.mean(per):.2f}+-{np.std(per):.2f}]")
    print(f"    B realistic (19 enrol, 12 probe)          "
          f"d'={dprime(gb,ib):6.2f}  EER={eer(gb,ib)[0]*100:5.1f}%  "
          f"FAR@10%FRR={far_at_frr(gb,ib)[0]*100:5.1f}%")
    return dict(Ad=dprime(ga, ia), Ae=eer(ga, ia)[0], Af=far_at_frr(ga, ia)[0],
                Bd=dprime(gb, ib), Be=eer(gb, ib)[0], Bf=far_at_frr(gb, ib)[0])


# --------------------------------------------------------------- matchers --

def ncc_matrix(ims, invalidate_rotation, mdx=20, mdy=8, min_overlap=3500,
               selfcheck=True):
    """LCN + rotation search + NCC.  `invalidate_rotation` False reproduces the
    shipped/baseline behaviour (edge-clamped rotation, corners are fabricated);
    True gives the out-of-frame pixels zero weight."""
    n = len(ims)
    if invalidate_rotation:
        ones = np.ones((H, W))
        P = [wncc.prep(im, ones, ROTS, mdx, mdy) for im in ims]
        S = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i, n):
                s, _ = wncc.surface(P[i], P[j], ROTS, mdx, mdy, min_overlap)
                v = float(np.nanmax(s)) if np.isfinite(s).any() else -1.0
                if i == j:
                    if selfcheck and v < 0.999:
                        print(f"    SANITY FAIL: self-match {v:.4f} at i={i}")
                    continue
                S[i, j] = S[j, i] = v
    else:
        # mncc.rotate(fill=None) is edge-clamping bilinear, identical to
        # matcher-lab.py's _rotate that the baseline uses.
        from mncc import rotate as clamp_rotate
        prerot = [[clamp_rotate(im, d) if d else im for d in ROTS] for im in ims]
        S = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i, n):
                v = max(ncc_all_shifts(ims[i], prerot[j][k], mdx, mdy, min_overlap)
                        for k in range(len(ROTS)))
                if i == j:
                    if selfcheck and v < 0.999:
                        print(f"    SANITY FAIL: self-match {v:.4f} at i={i}")
                    continue
                S[i, j] = S[j, i] = v
    return S


def main():
    ds = load_ds()
    labels = [l for l in ORDER for _ in ds[l]]
    print("dataset")
    for l in ORDER:
        role = "GENUINE " if l in GEN else "IMPOSTOR"
        print(f"  {l:20s} {len(ds[l]):3d} captures   {role}")
    print(f"  pooled genuine {sum(len(ds[l]) for l in GEN)}, impostor {len(ds[IMP])}")

    ims = [M.local_contrast_norm(im) for l in ORDER for _, im in ds[l]]

    print("\n=== correlation matchers (self-match sanity checked inline) ===")
    t0 = time.time()
    S0 = ncc_matrix(ims, invalidate_rotation=False)
    print(f"  [{time.time()-t0:.0f}s]")
    r0 = report("BASELINE  LCN + rotation(edge-clamped) + NCC, dx+-20 dy+-8",
                S0, labels)
    t0 = time.time()
    S1 = ncc_matrix(ims, invalidate_rotation=True)
    print(f"  [{time.time()-t0:.0f}s]")
    r1 = report("THIS WORK LCN + rotation(out-of-frame invalidated) + NCC",
                S1, labels)
    np.save("R_baseline.npy", S0)
    np.save("R_rotmask.npy", S1)

    print("\n=== minutiae (the assigned approach) ===")
    import dmin
    for tag, cfg in [("plain crossing-number",
                      dict(merge_dist=0.0, min_q=0.0, qthr=0.0, border=3,
                           spur_len=4, min_ridge=6)),
                     ("quality-gated + descriptor",
                      dict(merge_dist=0.0, min_q=0.35, qthr=0.25, border=3,
                           spur_len=4, min_ridge=6))]:
        T = [dmin.make_template(im, cfg) for l in ORDER for _, im in ds[l]]
        cnts = {l: np.mean([t.n for t, ll in zip(T, labels) if ll == l]) for l in ORDER}
        print(f"  [{tag}] minutiae/image overall {np.mean([t.n for t in T]):.1f}  "
              + "  ".join(f"{k} {v:.1f}" for k, v in cnts.items()))
        use_desc = "descriptor" in tag
        mcfg = dict(rot_max=20.0, rot_bin=5.0, tr_bin=8.0, pos_tol=9.0,
                    dir_tol=math.radians(25.0),
                    desc_thr=0.75 if use_desc else -1.0,
                    top_k=6, min_denom=6.0, use_desc=use_desc)
        ss = dmin.match(T[0], T[0], **mcfg)
        print(f"    SANITY self-match score {ss:.3f}")
        n = len(T)
        S = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                S[i, j] = S[j, i] = dmin.match(T[i], T[j], **mcfg)
        report(tag, S, labels)

    print("\n=== summary ===")
    print(f"  {'matcher':52s} {'A d':>6s} {'A EER':>7s} {'B d':>6s} {'B EER':>7s}")
    for tag, r in (("baseline (edge-clamped rotation)", r0),
                   ("out-of-frame invalidated after rotation", r1)):
        print(f"  {tag:52s} {r['Ad']:6.2f} {r['Ae']*100:6.1f}% "
              f"{r['Bd']:6.2f} {r['Be']*100:6.1f}%")


if __name__ == "__main__":
    main()
