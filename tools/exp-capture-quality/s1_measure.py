#!/usr/bin/env python3
"""
Capture-quality investigation, independent re-measurement.

Everything is driven off cache.npz + scores_*.npz built by build_cache.py
(LCN + rotation + NCC == the d'=1.54/1.94 baseline matcher, verified against
the naive implementation by s0_sanity.py).

Sections
  1  per-image quality metrics, by class
  2  does quality predict match score?  (pair level and image level)
  3  what does a quality GATE buy?  (10/20/30% dropped, vs random-drop control)
  4  how much do two presses of the same finger actually OVERLAP?
  5  headline table under both protocols
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                          # noqa: E402

FRAME = 150 * 52
CLASSES = ["right-index", "right-index-cover", "right-middle"]


# ------------------------------------------------------------- statistics --

def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def perm_p(a, b, reps=2000, seed=0):
    """Two-sided permutation p-value for spearman(a, b)."""
    rng = np.random.default_rng(seed)
    obs = abs(spearman(a, b))
    b = np.asarray(b, float)
    cnt = sum(abs(spearman(a, rng.permutation(b))) >= obs for _ in range(reps))
    return (cnt + 1) / (reps + 1)


# ------------------------------------------------------------- evaluation --

def score_probe(S, T, p):
    v = S[np.asarray(T), p]
    if np.isnan(v).any():
        raise AssertionError(f"self-comparison leak: probe {p} is in its own template set")
    return float(v.max())


def metrics(g, i):
    return E.dprime(g, i), E.eer(g, i), E.far_at_frr(g, i)


def protoA(S, gen_pool, gen_probe_src, imp, k, reps, seed):
    """Pooled protocol. Each rep: draw template T (size k) at random from
    gen_pool; genuine probes are gen_probe_src \\ T; impostor probes are all of
    imp. Metrics averaged over reps."""
    rng = np.random.default_rng(seed)
    gen_pool = np.asarray(gen_pool); gen_probe_src = np.asarray(gen_probe_src)
    k = min(k, len(gen_pool) - 1)
    out = []
    for _ in range(reps):
        T = rng.choice(gen_pool, size=k, replace=False)
        pr = np.setdiff1d(gen_probe_src, T)
        if len(pr) < 3:
            continue
        g = [score_probe(S, T, p) for p in pr]
        ii = [score_probe(S, T, p) for p in imp]
        out.append(metrics(g, ii))
    a = np.array(out)
    return a.mean(axis=0), a.std(axis=0), len(a)


def protoA_loo(S, gen, imp, k, reps, seed):
    """Strict leave-one-out variant: every genuine image is a probe in every
    rep, scored against a random k-subset of the OTHER genuine captures."""
    rng = np.random.default_rng(seed)
    gen = np.asarray(gen); imp = np.asarray(imp)
    out = []
    for _ in range(reps):
        g = []
        for p in gen:
            pool = gen[gen != p]
            T = rng.choice(pool, size=min(k, len(pool)), replace=False)
            g.append(score_probe(S, T, p))
        T = rng.choice(gen, size=min(k, len(gen)), replace=False)
        ii = [score_probe(S, T, p) for p in imp]
        out.append(metrics(g, ii))
    a = np.array(out)
    return a.mean(axis=0), a.std(axis=0), len(a)


def protoB(S, enrol, probes, imp):
    enrol = np.asarray(enrol); probes = np.asarray(probes)
    assert not (set(enrol.tolist()) & set(probes.tolist()))
    g = [score_probe(S, enrol, p) for p in probes]
    i = [score_probe(S, enrol, p) for p in imp]
    return metrics(g, i)


def bootstrapB(S, enrol, probes, imp, reps=4000, seed=13):
    rng = np.random.default_rng(seed)
    g = np.array([score_probe(S, enrol, p) for p in probes])
    i = np.array([score_probe(S, enrol, p) for p in imp])
    boot = np.array([metrics(g[rng.integers(0, len(g), len(g))],
                             i[rng.integers(0, len(i), len(i))]) for _ in range(reps)])
    return metrics(g, i), np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


# ------------------------------------------------------------------ gates --

def rank01(v):
    r = rankdata(v)
    return r / (len(v) - 1)


def composite(qual, qkeys, names):
    return np.mean([rank01(qual[:, qkeys.index(n)]) for n in names], axis=0)


def keep_top(idx, q, drop_frac):
    n_keep = max(2, int(round(len(idx) * (1 - drop_frac))))
    return np.sort(idx[np.argsort(-q[idx])[:n_keep]])


# ------------------------------------------------------------------- main --

def main():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    base = np.load(HERE / "scores_base.npz")
    wide = np.load(HERE / "scores_widestrict.npz")
    S = base["S"]
    labels = c["labels"]; qual = c["qual"]; fg_px = c["fg_px"]
    qkeys = [str(k) for k in c["qkeys"]]
    names = [str(n) for n in c["names"]]

    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]
    assert len(gen) == 31 and len(imp) == 14 and len(ri) == 12 and len(rc) == 19
    assert np.isnan(np.diag(S)).all(), "diagonal must be NaN (no self-comparison)"

    # ============================================================ 1. metrics
    print("=" * 78)
    print("1. PER-IMAGE CAPTURE QUALITY, BY CLASS (mean over class)")
    print("=" * 78)
    print(f"  {'metric':<16} " + "".join(f"{cl:>20}" for cl in CLASSES))
    for mi, m in enumerate(qkeys):
        row = "".join(f"{qual[labels == li, mi].mean():20.3f}" for li in range(3))
        print(f"  {m:<16} " + row)
    print()
    for li, cl in enumerate(CLASSES):
        px = fg_px[labels == li]
        print(f"  {cl:<20} finger area median {np.median(px):6.0f} px "
              f"({np.median(px)/FRAME*100:4.1f}% of frame), "
              f"range {px.min():.0f}-{px.max():.0f}")

    QSET = ["ridge_band", "aniso_w", "coh_w", "usable_frac"]
    q = composite(qual, qkeys, QSET)
    print(f"\n  composite quality = mean of rank-normalised {QSET}")
    for li, cl in enumerate(CLASSES):
        v = q[labels == li]
        print(f"    {cl:<20} mean {v.mean():.3f}  min {v.min():.3f}  max {v.max():.3f}")

    # ==================================================== 2. quality vs score
    print("\n" + "=" * 78)
    print("2. DOES CAPTURE QUALITY PREDICT MATCH SCORE?")
    print("=" * 78)
    gp = list(itertools.combinations(gen.tolist(), 2))
    ip = [(a, b) for a in gen.tolist() for b in imp.tolist()]
    Sg = np.array([S[a, b] for a, b in gp])
    Si = np.array([S[a, b] for a, b in ip])
    print(f"\n(a) genuine PAIRS (n={len(gp)}): Spearman rho of pair score vs the")
    print("    MIN and the MEAN quality of the two images  [p by permutation]")
    print(f"    {'metric':<16} {'rho(min)':>9} {'p':>8} {'rho(mean)':>10} {'p':>8}")
    for mi, m in enumerate(qkeys + ["composite"]):
        v = q if m == "composite" else qual[:, mi]
        vmin = np.array([min(v[a], v[b]) for a, b in gp])
        vavg = np.array([(v[a] + v[b]) / 2 for a, b in gp])
        r1, r2 = spearman(vmin, Sg), spearman(vavg, Sg)
        print(f"    {m:<16} {r1:9.3f} {perm_p(vmin, Sg, seed=1):8.4f} "
              f"{r2:10.3f} {perm_p(vavg, Sg, seed=2):8.4f}")

    print(f"\n(b) per genuine IMAGE (n={len(gen)}): rho of its own quality against")
    print("    its MEAN and its MAX score versus the other 30 genuine captures")
    Sgg = S[np.ix_(gen, gen)]
    mean_s = np.nanmean(Sgg, axis=1)
    max_s = np.nanmax(Sgg, axis=1)
    print(f"    {'metric':<16} {'rho(mean)':>10} {'p':>8} {'rho(max)':>9} {'p':>8}")
    for mi, m in enumerate(qkeys + ["composite"]):
        v = (q if m == "composite" else qual[:, mi])[gen]
        print(f"    {m:<16} {spearman(v, mean_s):10.3f} {perm_p(v, mean_s, seed=3):8.4f} "
              f"{spearman(v, max_s):9.3f} {perm_p(v, max_s, seed=4):8.4f}")

    print("\n(c) are the WORST genuine pairs made of low-quality images?")
    lo = Sg <= np.percentile(Sg, 25)
    hi = Sg >= np.percentile(Sg, 75)
    qmin = np.array([min(q[a], q[b]) for a, b in gp])
    print(f"    bottom-quartile genuine pairs: mean score {Sg[lo].mean():.3f}, "
          f"mean min-quality {qmin[lo].mean():.3f}")
    print(f"    top-quartile    genuine pairs: mean score {Sg[hi].mean():.3f}, "
          f"mean min-quality {qmin[hi].mean():.3f}")

    print("\n(d) how separable are the raw pair scores at all?")
    print(f"    genuine pairs  n={len(Sg)}  mean {Sg.mean():.3f} sd {Sg.std():.3f} "
          f"min {Sg.min():.3f} max {Sg.max():.3f}")
    print(f"    impostor pairs n={len(Si)}  mean {Si.mean():.3f} sd {Si.std():.3f} "
          f"min {Si.min():.3f} max {Si.max():.3f}")
    print(f"    genuine pairs above the BEST impostor pair ({Si.max():.3f}): "
          f"{(Sg > Si.max()).sum()}/{len(Sg)} ({(Sg > Si.max()).mean()*100:.0f}%)")

    # =========================================================== 3. the gate
    print("\n" + "=" * 78)
    print("3. WHAT DOES A CAPTURE-TIME QUALITY GATE BUY?")
    print("=" * 78)
    print("   Gate = drop the worst N% within each class. CONTROL = drop the")
    print("   same number of images AT RANDOM within each class, 300 draws.")
    print("   If the gated d' does not sit in the upper tail of the control")
    print("   distribution, the metric carries no usable information.")
    K_A, REPS_A = 19, 300

    a0 = protoA(S, gen, gen, imp, K_A, REPS_A, 0)
    b0 = protoB(S, rc, ri, imp)
    print(f"\n   BASELINE, no gate")
    print(f"     A (k={K_A}, {REPS_A} reps) d'={a0[0][0]:5.2f}+-{a0[1][0]:.2f}  "
          f"EER={a0[0][1]*100:5.1f}%  FAR@10%FRR={a0[0][2]*100:5.1f}%")
    print(f"     B                    d'={b0[0]:5.2f}       EER={b0[1]*100:5.1f}%  "
          f"FAR@10%FRR={b0[2]*100:5.1f}%")

    CAND = {m: rank01(qual[:, qkeys.index(m)])
            for m in ["ridge_band", "aniso_w", "coh_w", "usable_frac",
                      "bp_energy", "fg_frac"]}
    CAND["composite"] = q

    for mode in ("ENROL-ONLY", "ENROL+VERIFY"):
        print(f"\n   --- gate mode: {mode} " + "-" * (78 - 24 - len(mode)))
        print(f"   {'metric':<12} {'drop':>5} | {'A d-prime  gated / random(sd)':>32} "
              f"{'p':>7} | {'B d-prime gated / random':>26}")
        for mname, qv in CAND.items():
            for f in (0.10, 0.20, 0.30):
                gpool = keep_top(gen, qv, f)
                impk = keep_top(imp, qv, f)
                rck = keep_top(rc, qv, f)
                rik = keep_top(ri, qv, f)
                if mode == "ENROL-ONLY":
                    gprobe, iprobe, riprobe = gen, imp, ri
                else:
                    gprobe, iprobe, riprobe = gpool, impk, rik
                ag = protoA(S, gpool, gprobe, iprobe, K_A, REPS_A, 0)[0]
                bg = protoB(S, rck, riprobe, iprobe)

                rng = np.random.default_rng(1234)
                ar, br = [], []
                for _ in range(300):
                    gpool_r = np.sort(rng.choice(gen, len(gpool), replace=False))
                    imp_r = np.sort(rng.choice(imp, len(impk), replace=False))
                    rc_r = np.sort(rng.choice(rc, len(rck), replace=False))
                    ri_r = np.sort(rng.choice(ri, len(rik), replace=False))
                    if mode == "ENROL-ONLY":
                        gp2, ip2, rip2 = gen, imp, ri
                    else:
                        gp2, ip2, rip2 = gpool_r, imp_r, ri_r
                    ar.append(protoA(S, gpool_r, gp2, ip2, K_A, 12, 7)[0])
                    br.append(protoB(S, rc_r, rip2, ip2))
                ar = np.array(ar); br = np.array(br)
                pA = (np.sum(ar[:, 0] >= ag[0]) + 1) / (len(ar) + 1)
                print(f"   {mname:<12} {f*100:4.0f}% | {ag[0]:9.2f} / "
                      f"{ar[:,0].mean():5.2f}+-{ar[:,0].std():.2f}{'':>8} {pA:7.3f} | "
                      f"{bg[0]:9.2f} / {br[:,0].mean():5.2f}+-{br[:,0].std():.2f}")

    print("\n   ORACLE CEILING -- rank images by their own mean genuine score")
    print("   (i.e. cheat, look at the answer) and drop the worst. Upper bound")
    print("   on what ANY per-image gate could possibly achieve.")
    oracle = np.full(len(labels), -1e9, float)
    oracle[gen] = np.nanmean(S[np.ix_(gen, gen)], axis=1)
    oracle[imp] = np.nanmean(S[np.ix_(imp, imp)], axis=1)
    for f in (0.10, 0.20, 0.30):
        gpool = keep_top(gen, oracle, f)
        rck = keep_top(rc, oracle, f)
        ao = protoA(S, gpool, gen, imp, K_A, REPS_A, 0)[0]
        bo = protoB(S, rck, ri, imp)
        print(f"     drop {f*100:3.0f}% (enrol-only)  A d'={ao[0]:5.2f} EER={ao[1]*100:5.1f}%"
              f"   B d'={bo[0]:5.2f} EER={bo[1]*100:5.1f}%")

    # ========================================================== 4. overlap
    print("\n" + "=" * 78)
    print("4. HOW MUCH DO TWO PRESSES ACTUALLY OVERLAP?")
    print("=" * 78)
    print("   'shared finger area' = pixels that are foreground in BOTH frames")
    print("   at the best alignment, as a fraction of the SMALLER frame's own")
    print("   finger area. 1.00 would mean one capture fully contains the other.")
    for tag, z, lim in (("matcher's own window  +-20/+-8, min_overlap 3500", base, (20, 8)),
                        ("wide search +-60/+-20, min_overlap 4500", wide, (60, 20))):
        print(f"\n   [{tag}]")
        for what, pairs in (("genuine ", gp), ("impostor", ip)):
            DX, DY, BOTH = z["DX"], z["DY"], z["BOTH"]
            dx = np.array([abs(DX[a, b]) for a, b in pairs])
            dy = np.array([abs(DY[a, b]) for a, b in pairs])
            both = np.array([BOTH[a, b] for a, b in pairs])
            small = np.array([min(fg_px[a], fg_px[b]) for a, b in pairs])
            frac = both / np.maximum(small, 1)
            p = np.percentile(frac, [5, 25, 50, 75, 95])
            print(f"     {what} n={len(pairs):4d}  shared/smaller: p5={p[0]:.2f} "
                  f"p25={p[1]:.2f} MED={p[2]:.2f} p75={p[3]:.2f} p95={p[4]:.2f}")
            print(f"              |dx| med {np.median(dx):4.0f} p90 {np.percentile(dx,90):4.0f}"
                  f"   |dy| med {np.median(dy):3.0f} p90 {np.percentile(dy,90):3.0f}"
                  f"   pinned at window edge: {((dx>=lim[0])|(dy>=lim[1])).mean()*100:3.0f}%")

    both_ws = np.array([wide["BOTH"][a, b] for a, b in gp])
    small_ws = np.array([min(fg_px[a], fg_px[b]) for a, b in gp])
    frac_ws = both_ws / np.maximum(small_ws, 1)
    print("\n   Does overlap predict the score better than quality does?")
    print(f"     rho(shared-area fraction, genuine pair score) = {spearman(frac_ws, Sg):.3f}"
          f"  p={perm_p(frac_ws, Sg, seed=5):.4f}")
    print(f"     rho(min composite quality, genuine pair score) = {spearman(qmin, Sg):.3f}"
          f"  p={perm_p(qmin, Sg, seed=6):.4f}")
    top = frac_ws >= np.percentile(frac_ws, 75)
    print(f"     top-quartile-overlap genuine pairs: mean score {Sg[top].mean():.3f}; "
          f"rest {Sg[~top].mean():.3f}; impostor mean {Si.mean():.3f}")
    print(f"     even that best quarter beats only "
          f"{(Si < Sg[top].mean()).mean()*100:.0f}% of impostor pairs.")

    # what if we could ONLY use high-overlap pairs -- pair-level ceiling
    print("\n   PAIR-LEVEL CEILING: restrict the genuine set to pairs whose")
    print("   shared area exceeds a threshold (an oracle -- unknowable at")
    print("   enrol time -- but it bounds what better PLACEMENT would buy).")
    for thr in (0.5, 0.6, 0.7, 0.8):
        sel = frac_ws >= thr
        if sel.sum() < 10:
            continue
        d, e, fa = metrics(Sg[sel], Si)
        print(f"     shared >= {thr:.1f}: {sel.sum():3d}/{len(Sg)} genuine pairs kept, "
              f"pairwise d'={d:5.2f} EER={e*100:5.1f}% FAR@10={fa*100:5.1f}%")
    d, e, fa = metrics(Sg, Si)
    print(f"     no restriction : {len(Sg):3d}/{len(Sg)} genuine pairs kept, "
          f"pairwise d'={d:5.2f} EER={e*100:5.1f}% FAR@10={fa*100:5.1f}%")

    # ========================================================= 5. headline
    print("\n" + "=" * 78)
    print("5. HEADLINE -- both protocols, best configuration this study found")
    print("=" * 78)
    best_metric, best_f, best_mode = None, None, None
    print("\n   SCENARIO A (POOLED): 31 genuine, 14 impostor. Template = random")
    print(f"   k-subset, probes = the genuine complement + all impostors, {REPS_A} reps.")
    print(f"   {'configuration':<34} {'d-prime':>13} {'EER %':>13} {'FAR@10%FRR %':>15}")
    for k in (10, 15, 19):
        m, s, n = protoA(S, gen, gen, imp, k, REPS_A, 0)
        print(f"   {'baseline, k=%d templates' % k:<34} {m[0]:6.2f}+-{s[0]:4.2f} "
              f"{m[1]*100:7.1f}+-{s[1]*100:4.1f} {m[2]*100:8.1f}+-{s[2]*100:4.1f}")
    m, s, n = protoA_loo(S, gen, imp, 19, 100, 0)
    print(f"   {'baseline, strict LOO k=19':<34} {m[0]:6.2f}+-{s[0]:4.2f} "
          f"{m[1]*100:7.1f}+-{s[1]*100:4.1f} {m[2]*100:8.1f}+-{s[2]*100:4.1f}")
    # best gate found
    gpool = keep_top(gen, q, 0.10)
    m, s, n = protoA(S, gpool, gen, imp, K_A, REPS_A, 0)
    print(f"   {'+ composite quality gate, drop 10%':<34} {m[0]:6.2f}+-{s[0]:4.2f} "
          f"{m[1]*100:7.1f}+-{s[1]*100:4.1f} {m[2]*100:8.1f}+-{s[2]*100:4.1f}")

    print("\n   SCENARIO B (REALISTIC): enrol=right-index-cover(19), "
          "probe=right-index(12),")
    print("   impostor=right-middle(14). [95% bootstrap CI over probes]")
    print(f"   {'configuration':<34} {'d-prime (CI)':>21} {'EER % (CI)':>21} "
          f"{'FAR@10 % (CI)':>21}")
    for name, T in (("baseline, all 19 templates", rc),
                    ("+ composite quality gate, drop 10%", keep_top(rc, q, 0.10)),
                    ("+ composite quality gate, drop 20%", keep_top(rc, q, 0.20))):
        (d, e, fa), lo, hi = bootstrapB(S, np.asarray(T), ri, imp)
        print(f"   {name:<34} {d:6.2f}[{lo[0]:5.2f},{hi[0]:5.2f}] "
              f"{e*100:6.1f}[{lo[1]*100:4.1f},{hi[1]*100:5.1f}] "
              f"{fa*100:6.1f}[{lo[2]*100:4.1f},{hi[2]*100:5.1f}]")
    print("\n   12 genuine probes => EER quantised in 8.3% steps, FAR in 7.1%")
    print("   steps. Scenario B cannot resolve d' differences below about 1.0.")


if __name__ == "__main__":
    main()
