#!/usr/bin/env python3
"""
Capture-quality investigation, run against the DIRECTIONAL score cache.

Sections
  1  per-image quality metrics, by class
  2  does quality predict match score?
  3  quality gate: drop worst N% (10/20/30), vs a random-drop control
  4  overlap at the best alignment -- how much do two presses actually share?
  5  variance decomposition: per-image matchability vs pair-specific placement
  6  enrolment selection: quality vs diversity vs random
  7  headline table, both protocols
"""
import itertools, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
C = np.load(EXP / "dir_cache.npz", allow_pickle=True)
S = C["S"]; labels = C["labels"]; qual = C["qual"]
qkeys = [str(k) for k in C["qkeys"]]
GEO = C["GEO"]; BOTH = C["BOTH"]; DX = C["DX"]; DY = C["DY"]
DEG = C["DEG"]; EDGE = C["EDGE"]; fg_px = C["fg_px"]
N = len(labels)
GEN = np.where(labels < 2)[0]; IMP = np.where(labels == 2)[0]
RI = np.where(labels == 0)[0]; RC = np.where(labels == 1)[0]
LBL = ["right-index", "right-index-cover", "right-middle"]
FRAME = 150 * 52


# -------------------------------------------------------------- metrics --
def dprime(g, i):
    g, i = np.asarray(g, float), np.asarray(i, float)
    den = math.sqrt((g.var() + i.var()) / 2)
    return float((g.mean() - i.mean()) / den) if den > 0 else 0.0


def eer(g, i):
    g, i = np.asarray(g, float), np.asarray(i, float)
    bg, be = math.inf, 1.0
    for t in np.unique(np.concatenate([g, i])):
        frr = float((g < t).mean()); far = float((i >= t).mean())
        if abs(frr - far) < bg:
            bg, be = abs(frr - far), (frr + far) / 2
    return be


def far_at_frr(g, i, target=0.10):
    t = float(np.quantile(np.asarray(g, float), target))
    return float((np.asarray(i, float) >= t).mean())


def rank(v):
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v)); r[o] = np.arange(len(v), dtype=float)
    # average ties
    u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(u)); np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def perm_p(a, b, obs, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    cnt = sum(1 for _ in range(n) if abs(spearman(a, rng.permutation(b))) >= abs(obs))
    return (cnt + 1) / (n + 1)


def score_set(T, gp, ip):
    """max-over-templates score. T are templates (rows), probes are columns."""
    T = np.asarray(T); gp = np.asarray(gp); ip = np.asarray(ip)
    assert not (set(T.tolist()) & set(gp.tolist())), "PROBE USED AS ITS OWN TEMPLATE"
    assert not np.isnan(S[np.ix_(T, gp)]).any(), "self-comparison leak"
    g = S[np.ix_(T, gp)].max(axis=0)
    i = S[np.ix_(T, ip)].max(axis=0)
    return dprime(g, i), eer(g, i), far_at_frr(g, i)


def protoA(pool, gprobe_universe, iprobes, n_enroll, reps, seed, select=None,
           q=None):
    """Random template subsets drawn from `pool`; genuine probes are whatever
    of gprobe_universe is not in the subset."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool)
    ne = min(n_enroll, len(pool) - 1)
    out = []
    for _ in range(reps):
        T = rng.choice(pool, size=ne, replace=False)
        if select is not None:
            T = select(T, q, rng)
        pr = np.setdiff1d(gprobe_universe, T)
        if len(pr) < 3:
            continue
        out.append(score_set(T, pr, iprobes))
    return np.array(out)


def keep_top(idx, q, drop_frac):
    idx = np.asarray(idx)
    k = max(2, int(round(len(idx) * (1 - drop_frac))))
    return np.sort(idx[np.argsort(-q[idx])[:k]])


def composite(metrics, ranks=True):
    cols = []
    for m in metrics:
        v = qual[:, qkeys.index(m)].astype(float)
        cols.append(rank(v) / (len(v) - 1) if ranks else (v - v.mean()) / (v.std() + 1e-12))
    return np.mean(cols, axis=0)


# ============================================================ 1. metrics --
def sec1():
    print("=" * 78)
    print("1. PER-IMAGE CAPTURE QUALITY, BY CLASS  (mean over the class)")
    print("=" * 78)
    print(f"  {'metric':<16}" + "".join(f"{l:>20}" for l in LBL))
    for k in qkeys:
        v = qual[:, qkeys.index(k)]
        print(f"  {k:<16}" + "".join(f"{v[labels==i].mean():20.3f}" for i in range(3)))
    print(f"\n  finger (foreground) pixels per image, median by class: " +
          ", ".join(f"{LBL[i]}={np.median(fg_px[labels==i]):.0f}"
                    f" ({np.median(fg_px[labels==i])/FRAME*100:.0f}% of frame)"
                    for i in range(3)))
    q = composite(["ridge_band", "aniso_w", "coh_w", "usable_frac"])
    print(f"\n  composite quality (mean of 4 rank-normalised metrics):")
    for i in range(3):
        v = q[labels == i]
        print(f"    {LBL[i]:<20} mean={v.mean():.3f} min={v.min():.3f} max={v.max():.3f}")
    return q


# ================================================ 2. quality vs score --
def sec2():
    print("\n" + "=" * 78)
    print("2. DOES CAPTURE QUALITY PREDICT MATCH SCORE?")
    print("=" * 78)
    pairs = [(i, j) for i, j in itertools.combinations(GEN, 2)]
    ps = np.array([max(S[i, j], S[j, i]) for i, j in pairs])
    print(f"\n(a) genuine PAIRS (n={len(pairs)}): Spearman rho of the pair score")
    print("    against the min / mean quality of the two images")
    print(f"    {'metric':<16} {'rho(min)':>9} {'p':>7} {'rho(mean)':>10} {'p':>7}")
    for k in qkeys:
        v = qual[:, qkeys.index(k)]
        mn = np.array([min(v[i], v[j]) for i, j in pairs])
        mu = np.array([(v[i] + v[j]) / 2 for i, j in pairs])
        r1, r2 = spearman(mn, ps), spearman(mu, ps)
        print(f"    {k:<16} {r1:9.3f} {perm_p(mn,ps,r1,2000):7.4f} "
              f"{r2:10.3f} {perm_p(mu,ps,r2,2000):7.4f}")

    print(f"\n(b) per genuine IMAGE (n={len(GEN)}): rho of its own quality against")
    print("    its mean, and its max, score versus the other 30 genuine captures")
    Sg = S[np.ix_(GEN, GEN)]
    mean_s = np.nanmean(Sg, axis=1); max_s = np.nanmax(Sg, axis=1)
    print(f"    {'metric':<16} {'rho(mean)':>10} {'p':>7} {'rho(max)':>9} {'p':>7}")
    best = (0, None)
    for k in qkeys:
        v = qual[GEN, qkeys.index(k)]
        r1, r2 = spearman(v, mean_s), spearman(v, max_s)
        if abs(r1) > abs(best[0]):
            best = (r1, k)
        print(f"    {k:<16} {r1:10.3f} {perm_p(v,mean_s,r1,5000):7.4f} "
              f"{r2:9.3f} {perm_p(v,max_s,r2,5000):7.4f}")
    print(f"\n    strongest single predictor of mean genuine score: "
          f"{best[1]} (rho={best[0]:+.3f}, r^2={best[0]**2*100:.0f}% of rank variance)")

    print("\n(c) are the WORST genuine pairs made of low-quality images?")
    q = composite(["ridge_band", "aniso_w", "coh_w", "usable_frac"])
    order = np.argsort(ps)
    for tag, sel in (("20 WORST", order[:20]), ("20 BEST", order[-20:])):
        cq = [min(q[pairs[k][0]], q[pairs[k][1]]) for k in sel]
        sc = ps[sel]
        print(f"    {tag} pairs: score {np.mean(sc):.3f}, "
              f"min-of-pair composite quality {np.mean(cq):.3f}")
    print(f"    (composite quality of all genuine images: {q[GEN].mean():.3f})")


# ================================================== 3. the quality gate --
def sec3(q):
    print("\n" + "=" * 78)
    print("3. WHAT DOES A CAPTURE-TIME QUALITY GATE BUY?")
    print("=" * 78)
    print("  Every gated figure is paired with a RANDOM-DROP control that removes")
    print("  the same number of images at random (60 draws). If the gate does no")
    print("  better than random, the metric carries no usable information.")
    print("  Dropping is done WITHIN class, so it cannot flatter the impostor set.")
    NE, REPS, DRAWS = 15, 300, 60

    base_A = protoA(GEN, GEN, IMP, NE, REPS, 1)
    base_B = score_set(RC, RI, IMP)
    print(f"\n  BASELINE (no gate)")
    print(f"    A pooled   (n_enroll={NE}, {REPS} random subsets):"
          f"  d'={base_A[:,0].mean():5.2f}+-{base_A[:,0].std():.2f}"
          f"  EER={base_A[:,1].mean()*100:5.1f}%  FAR@10%FRR={base_A[:,2].mean()*100:5.1f}%")
    print(f"    B realistic(enrol 19 cover, probe 12 index):"
          f"    d'={base_B[0]:5.2f}       "
          f"  EER={base_B[1]*100:5.1f}%  FAR@10%FRR={base_B[2]*100:5.1f}%")

    CAND = {m: composite([m]) for m in
            ("ridge_band", "aniso_w", "coh_w", "usable_frac", "bp_energy", "fg_frac")}
    CAND["composite"] = q
    # oracle: rank by the image's own mean genuine score (cheats; upper bound)
    orc = np.zeros(N)
    orc[GEN] = np.nanmean(S[np.ix_(GEN, GEN)], axis=1)
    orc[IMP] = np.nanmean(S[np.ix_(IMP, IMP)], axis=1)
    CAND["ORACLE(cheat)"] = orc

    for mode in ("ENROL-ONLY", "ENROL+VERIFY"):
        print("\n  " + "-" * 74)
        print(f"  MODE {mode}" + ("   (probe sets unchanged -- clean comparison)"
              if mode == "ENROL-ONLY" else "   (bad probes re-prompted; probe sets shrink)"))
        print(f"  {'gate metric':<14} {'drop':>5} | {'A d-prime':>19} | {'A EER%':>15}"
              f" | {'A FAR@10%':>15} | {'B d-prime':>19}")
        print(f"  {'':<14} {'':>5} | {'gated':>8} {'random':>10} | {'gated':>6} {'rand':>8}"
              f" | {'gated':>6} {'rand':>8} | {'gated':>8} {'random':>10}")
        for mname, qq in CAND.items():
            for f in (0.10, 0.20, 0.30):
                pool = keep_top(GEN, qq, f)
                impk = keep_top(IMP, qq, f)
                rck = keep_top(RC, qq, f); rik = keep_top(RI, qq, f)
                if mode == "ENROL-ONLY":
                    gp, ip, rip = GEN, IMP, RI
                else:
                    gp, ip, rip = pool, impk, rik
                ag = protoA(pool, gp, ip, NE, REPS, 1)
                bg = score_set(rck, rip, ip)
                rng = np.random.default_rng(99)
                ar, br = [], []
                for _ in range(DRAWS):
                    pr = np.sort(rng.choice(GEN, len(pool), replace=False))
                    ir = np.sort(rng.choice(IMP, len(impk), replace=False))
                    rcr = np.sort(rng.choice(RC, len(rck), replace=False))
                    rir = np.sort(rng.choice(RI, len(rik), replace=False))
                    if mode == "ENROL-ONLY":
                        gp2, ip2, rip2 = GEN, IMP, RI
                    else:
                        gp2, ip2, rip2 = pr, ir, rir
                    ar.append(protoA(pr, gp2, ip2, NE, 25, 2).mean(axis=0))
                    br.append(score_set(rcr, rip2, ip2))
                ar = np.array(ar); br = np.array(br)
                print(f"  {mname:<14} {f*100:4.0f}% | {ag[:,0].mean():8.2f}"
                      f" {ar[:,0].mean():6.2f}+-{ar[:,0].std():.2f} |"
                      f" {ag[:,1].mean()*100:6.1f} {ar[:,1].mean()*100:8.1f} |"
                      f" {ag[:,2].mean()*100:6.1f} {ar[:,2].mean()*100:8.1f} |"
                      f" {bg[0]:8.2f} {br[:,0].mean():6.2f}+-{br[:,0].std():.2f}")


# ======================================================== 4. overlap --
def sec4():
    print("\n" + "=" * 78)
    print("4. HOW MUCH AREA DO TWO PRESSES ACTUALLY SHARE?")
    print("=" * 78)
    print("  Measured at the alignment the baseline matcher actually chose")
    print("  (LCN+NCC, search +-20 px in x, +-8 in y, +-12 deg).")

    def block(name, idx_pairs):
        geo = np.array([GEO[i, j] for i, j in idx_pairs], float)
        both = np.array([BOTH[i, j] for i, j in idx_pairs], float)
        sm = np.array([min(fg_px[i], fg_px[j]) for i, j in idx_pairs], float)
        dx = np.array([abs(DX[i, j]) for i, j in idx_pairs], float)
        dy = np.array([abs(DY[i, j]) for i, j in idx_pairs], float)
        dg = np.array([abs(DEG[i, j]) for i, j in idx_pairs], float)
        ed = np.array([EDGE[i, j] for i, j in idx_pairs])
        pr = lambda v: "  ".join(f"p{p}={np.percentile(v,p):7.1f}" for p in (5,25,50,75,95))
        print(f"\n  {name}  (n={len(idx_pairs)} ordered pairs)")
        print(f"    geometric overlap px     {pr(geo)}   median {np.median(geo)/FRAME*100:.0f}% of frame")
        print(f"    BOTH-finger overlap px   {pr(both)}   median {np.median(both)/FRAME*100:.0f}% of frame")
        print(f"    shared / smaller finger  {pr(both/np.maximum(sm,1))}")
        print(f"    |dx| chosen              {pr(dx)}")
        print(f"    |dy| chosen              {pr(dy)}")
        print(f"    |rotation| chosen (deg)  {pr(dg)}")
        print(f"    alignment pinned to the search-window EDGE: {ed.mean()*100:.0f}%")
        return both

    gp = [(i, j) for i in GEN for j in GEN if i != j]
    ip = [(i, j) for i in GEN for j in IMP]
    bg = block("GENUINE pairs", gp)
    bi = block("IMPOSTOR pairs", ip)
    sc = np.array([S[i, j] for i, j in gp])
    r = spearman(bg, sc)
    print(f"\n  rho(shared finger area, LCN+NCC score) over genuine pairs = "
          f"{r:+.3f} (p={perm_p(bg,sc,r,2000):.4f})")
    print(f"  median shared-finger area: genuine {np.median(bg):.0f} px vs "
          f"impostor {np.median(bi):.0f} px")
    print("  -> overlap area by itself does NOT separate genuine from impostor.")

    # separability accounting
    imp_max = max(S[i, j] for i, j in ip)
    n_beat = sum(1 for i, j in gp if S[i, j] > imp_max)
    print(f"\n  best impostor pair score = {imp_max:.3f}")
    print(f"  genuine pairs above it: {n_beat}/{len(gp)} ({n_beat/len(gp)*100:.0f}%)")
    hi = [b for (i, j), b in zip(gp, bg) if S[i, j] > imp_max]
    lo = [b for (i, j), b in zip(gp, bg) if S[i, j] <= imp_max]
    if hi:
        print(f"  their median shared-finger area {np.median(hi):.0f} px vs "
              f"{np.median(lo):.0f} px for the rest")


# ============================================ 5. variance decomposition --
def sec5(q):
    print("\n" + "=" * 78)
    print("5. VARIANCE DECOMPOSITION OF THE GENUINE SCORE MATRIX")
    print("=" * 78)
    print("  Fit  s_ij = mu + a_i + a_j + e_ij  over genuine pairs.")
    print("  a_i is image i's intrinsic matchability -- the ONLY thing a")
    print("  per-image capture-quality gate can ever influence.")
    M = np.array([[max(S[i, j], S[j, i]) if i != j else np.nan for j in GEN] for i in GEN])
    n = len(GEN)
    obs = ~np.isnan(M)
    mu = float(np.nanmean(M))
    a = np.zeros(n)
    for _ in range(800):
        for i in range(n):
            a[i] = float((M[i][obs[i]] - mu - a[obs[i]]).mean())
        a -= a.mean()
    pred = mu + a[:, None] + a[None, :]
    resid = M[obs] - pred[obs]
    vtot = float(np.var(M[obs] - mu)); vres = float(np.var(resid))
    share = 1 - vres / vtot
    print(f"\n  n genuine pairs {int(obs.sum())//2}, mean score {mu:.4f}")
    print(f"  total variance                             {vtot:.6f}")
    print(f"  residual after per-image effects           {vres:.6f}")
    print(f"  -> PER-IMAGE  (a gate could attack this):  {share*100:5.1f}%")
    print(f"  -> PAIR-SPECIFIC placement interaction  :  {(1-share)*100:5.1f}%")
    print(f"  per-image effect a_i: sd={a.std():.4f}, range {a.min():+.4f}..{a.max():+.4f}")

    print("\n  how well can measurable quality predict a_i?")
    bestr, bestk = 0, None
    for k in qkeys:
        v = qual[GEN, qkeys.index(k)]
        r = spearman(v, a)
        if abs(r) > abs(bestr):
            bestr, bestk = r, k
        if abs(r) > 0.3:
            print(f"    {k:<16} rho={r:+.3f}  p={perm_p(v,a,r,5000):.4f}")
    rc_ = spearman(q[GEN], a)
    print(f"    {'composite':<16} rho={rc_:+.3f}  p={perm_p(q[GEN],a,rc_,5000):.4f}")
    print(f"\n  CEILING ON ANY PER-IMAGE GATE:")
    print(f"    best metric {bestk} explains rho^2 = {bestr**2*100:.0f}% of a_i,")
    print(f"    and a_i is only {share*100:.0f}% of the genuine-score variance,")
    print(f"    so a perfect gate on it touches {bestr**2*share*100:.1f}% of the variance.")
    return a, share


# ======================================= 6. enrolment selection strategy --
def sec6(q):
    print("\n" + "=" * 78)
    print("6. ENROLMENT SELECTION: QUALITY vs DIVERSITY vs RANDOM")
    print("=" * 78)
    print("  All three only ever look at the enrolment pool, never at a probe,")
    print("  so all three are implementable on device.")

    def sel_random(pool, k, rng):
        return rng.choice(pool, k, replace=False)

    def sel_quality(pool, k, rng):
        pool = np.asarray(pool)
        return pool[np.argsort(-q[pool])[:k]]

    def sel_diverse(pool, k, rng):
        pool = list(pool)
        start = max(pool, key=lambda p: q[p])
        sel = [start]
        while len(sel) < k:
            rest = [p for p in pool if p not in sel]
            if not rest:
                break
            sel.append(min(rest, key=lambda p: max(S[p, s] for s in sel)))
        return np.array(sel)

    SEL = {"random": sel_random, "quality": sel_quality, "diverse": sel_diverse}
    REPS = 300
    print("\n  PROTOCOL A -- pooled 31 genuine split at random into an enrol POOL")
    print(f"  of 19 and 12 probes, {REPS} splits; k templates chosen from the pool.")
    print(f"  {'k':>3} | " + " | ".join(f"{n:>14}" for n in SEL) + " |    all-19")
    for k in (3, 5, 7, 9, 11, 13, 15, 17, 19):
        rng0 = np.random.default_rng(7)
        rows = {n: [] for n in SEL}; rows["all"] = []
        for _ in range(REPS):
            perm = rng0.permutation(GEN)
            pool, probes = np.sort(perm[:19]), np.sort(perm[19:])
            for n, fn in SEL.items():
                rows[n].append(score_set(fn(pool, k, rng0), probes, IMP))
            rows["all"].append(score_set(pool, probes, IMP))
        cells = []
        for n in SEL:
            v = np.array(rows[n])
            cells.append(f"{v[:,0].mean():6.2f}+-{v[:,0].std():4.2f}")
        va = np.array(rows["all"])
        print(f"  {k:3d} | " + " | ".join(cells) + f" | {va[:,0].mean():9.2f}")

    print("\n  PROTOCOL B -- enrol pool = 19 right-index-cover, probe = 12 right-index")
    print(f"  {'k':>3} | " + " | ".join(f"{n:>14}" for n in SEL))
    rngb = np.random.default_rng(21)
    for k in (3, 5, 7, 9, 11, 13, 15, 17, 19):
        cells = []
        for n, fn in SEL.items():
            if n == "random":
                v = np.array([score_set(fn(RC, k, rngb), RI, IMP) for _ in range(300)])
                cells.append(f"{v[:,0].mean():6.2f}+-{v[:,0].std():4.2f}")
            else:
                v = score_set(fn(RC, k, rngb), RI, IMP)
                cells.append(f"{v[0]:6.2f}      ")
        print(f"  {k:3d} | " + " | ".join(cells))
    return SEL


# ================================================== 7. headline table --
def sec7(q, SEL):
    print("\n" + "=" * 78)
    print("7. HEADLINE -- both protocols, same matcher (LCN + rotation + NCC),")
    print("   only the ENROLMENT SELECTION / QUALITY GATE differs.")
    print("=" * 78)
    REPS = 500
    CFG = [("baseline: all 19 templates", None, 19),
           ("random 15 of 19", "random", 15),
           ("top-15 by capture quality", "quality", 15),
           ("15 most DIVERSE of 19", "diverse", 15),
           ("11 most DIVERSE of 19", "diverse", 11),
           ("9 most DIVERSE of 19", "diverse", 9)]
    print(f"\n  SCENARIO A -- POOLED. {REPS} random enrol-pool(19)/probe(12) splits")
    print("  of the 31 genuine captures; impostors = all 14 right-middle.")
    print(f"  {'configuration':<28} {'d-prime':>14} {'EER %':>14} {'FAR@10%FRR %':>16}")
    store = {}
    for name, s, k in CFG:
        rng = np.random.default_rng(3)
        rows = []
        for _ in range(REPS):
            perm = rng.permutation(GEN)
            pool, probes = np.sort(perm[:19]), np.sort(perm[19:])
            T = pool if s is None else SEL[s](pool, k, rng)
            rows.append(score_set(T, probes, IMP))
        r = np.array(rows); store[name] = r
        print(f"  {name:<28} {r[:,0].mean():6.2f}+-{r[:,0].std():4.2f} "
              f"{r[:,1].mean()*100:8.1f}+-{r[:,1].std()*100:4.1f} "
              f"{r[:,2].mean()*100:9.1f}+-{r[:,2].std()*100:4.1f}")
    base = store["baseline: all 19 templates"]
    print("\n  paired against the baseline on the SAME splits (mean diff, "
          "bootstrap p):")
    for name, r in store.items():
        if r is base:
            continue
        d = r[:, 0] - base[:, 0]
        e = r[:, 1] - base[:, 1]
        rng = np.random.default_rng(4)
        bd = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
        be = np.array([rng.choice(e, len(e)).mean() for _ in range(4000)])
        pd_ = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
        pe_ = 2 * min((be <= 0).mean(), (be >= 0).mean())
        print(f"    {name:<28} dd'={d.mean():+.3f} (p={pd_:.4f})   "
              f"dEER={e.mean()*100:+.2f}pp (p={pe_:.4f})")

    print(f"\n  SCENARIO B -- REALISTIC. enrol = right-index-cover (19),")
    print("  probe = right-index (12), impostor = right-middle (14).")
    print("  [95% bootstrap CI resampling the probe sets]")
    print(f"  {'configuration':<28} {'d-prime (CI)':>21} {'EER % (CI)':>21} "
          f"{'FAR@10%FRR % (CI)':>23}")
    rngb = np.random.default_rng(5)
    for name, s, k in CFG:
        T = RC if s is None else SEL[s](RC, k, rngb)
        T = np.asarray(T)
        g = S[np.ix_(T, RI)].max(axis=0)
        i = S[np.ix_(T, IMP)].max(axis=0)
        pt = (dprime(g, i), eer(g, i), far_at_frr(g, i))
        rb = np.random.default_rng(6)
        bs = []
        for _ in range(3000):
            gg = rb.choice(g, len(g)); ii = rb.choice(i, len(i))
            bs.append((dprime(gg, ii), eer(gg, ii), far_at_frr(gg, ii)))
        bs = np.array(bs)
        lo = np.percentile(bs, 2.5, axis=0); hi = np.percentile(bs, 97.5, axis=0)
        print(f"  {name:<28} {pt[0]:5.2f} [{lo[0]:5.2f},{hi[0]:5.2f}] "
              f"{pt[1]*100:6.1f} [{lo[1]*100:4.1f},{hi[1]*100:5.1f}] "
              f"{pt[2]*100:8.1f} [{lo[2]*100:4.1f},{hi[2]*100:5.1f}]")
    print("\n  NOTE: 12 genuine probes quantises EER in 8.3% steps and 14")
    print("  impostors quantise FAR in 7.1% steps. Scenario B cannot resolve")
    print("  d' differences below ~1.0; scenario A is the number to trust.")


if __name__ == "__main__":
    q = sec1()
    sec2()
    sec3(q)
    sec4()
    sec5(q)
    SEL = sec6(q)
    sec7(q, SEL)
