#!/usr/bin/env python3
"""
Stage 2: the actual measurements.

  1. per-image quality distributions
  2. does quality predict match score?
  3. what does a quality gate buy? (with a random-drop control)
  4. how much area do genuine pairs really share?
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                        # noqa: E402

LABELS = ["right-index", "right-index-cover", "right-middle"]
N_ENROLL_A = 10
REPS = 24


def spearman(x, y):
    """Rank correlation, no scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return 0.0

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, float)
        r[o] = np.arange(v.size, dtype=float)
        # average ties
        vs = v[o]
        i = 0
        while i < vs.size:
            j = i
            while j + 1 < vs.size and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def perm_p(x, y, rho, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    cnt = 0
    for _ in range(n):
        if abs(spearman(x, rng.permutation(y))) >= abs(rho):
            cnt += 1
    return (cnt + 1) / (n + 1)


def main():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    names = list(c["names"])
    labels = c["labels"]
    qual = c["qual"]
    qkeys = [str(k) for k in c["qkeys"]]
    fg_px = c["fg_px"]

    base = np.load(HERE / "scores_base.npz")
    wide = np.load(HERE / "scores_wide.npz")
    S = base["S"]

    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]
    N = len(labels)

    print("=" * 74)
    print("1. PER-IMAGE QUALITY METRICS")
    print("=" * 74)
    print(f"{'metric':<16} {'min':>8} {'p25':>8} {'median':>8} {'p75':>8} "
          f"{'max':>8} {'CV':>6}")
    for k, col in zip(qkeys, qual.T):
        p = np.percentile(col, [0, 25, 50, 75, 100])
        cv = col.std() / (abs(col.mean()) + 1e-12)
        print(f"{k:<16} {p[0]:8.3f} {p[1]:8.3f} {p[2]:8.3f} {p[3]:8.3f} "
              f"{p[4]:8.3f} {cv:6.2f}")

    print("\nper-label means:")
    print(f"{'metric':<16} " + " ".join(f"{l:>19}" for l in LABELS))
    for k, col in zip(qkeys, qual.T):
        print(f"{k:<16} " + " ".join(f"{col[labels==i].mean():19.3f}"
                                     for i in range(3)))

    # ---------------------------------------------------------------- 2 --
    print("\n" + "=" * 74)
    print("2. DOES QUALITY PREDICT MATCH SCORE?")
    print("=" * 74)

    # (a) pairwise: score of a genuine pair vs the MIN quality of the two
    gp = list(itertools.combinations(gen.tolist(), 2))
    gs = np.array([S[i, j] for i, j in gp])
    print(f"\n(a) genuine pairs (n={len(gp)}): Spearman rho of pair score vs")
    print("    min / mean quality of the two images")
    print(f"{'metric':<16} {'rho(min)':>10} {'p':>8} {'rho(mean)':>11} {'p':>8}")
    for qi, k in enumerate(qkeys):
        qmin = np.array([min(qual[i, qi], qual[j, qi]) for i, j in gp])
        qavg = np.array([(qual[i, qi] + qual[j, qi]) / 2 for i, j in gp])
        r1 = spearman(qmin, gs)
        r2 = spearman(qavg, gs)
        print(f"{k:<16} {r1:10.3f} {perm_p(qmin, gs, r1, 2000):8.3f} "
              f"{r2:11.3f} {perm_p(qavg, gs, r2, 2000):8.3f}")

    # (b) per-image: an image's mean score against all OTHER genuine images
    print("\n(b) per genuine image (n=%d): rho of its own quality vs its mean"
          % len(gen))
    print("    LCN+NCC score against the other 30 genuine captures")
    Sg = S[np.ix_(gen, gen)]
    img_mean = np.nanmean(Sg, axis=1)
    img_max = np.nanmax(np.where(np.isnan(Sg), -9, Sg), axis=1)
    print(f"{'metric':<16} {'rho(mean)':>10} {'p':>8} {'rho(max)':>10} {'p':>8}")
    best_metric, best_rho = None, 0.0
    for qi, k in enumerate(qkeys):
        q = qual[gen, qi]
        r1 = spearman(q, img_mean)
        r2 = spearman(q, img_max)
        print(f"{k:<16} {r1:10.3f} {perm_p(q, img_mean, r1, 5000):8.3f} "
              f"{r2:10.3f} {perm_p(q, img_max, r2, 5000):8.3f}")
        if abs(r1) > abs(best_rho):
            best_metric, best_rho = k, r1
    print(f"\n    strongest single predictor of mean genuine score: "
          f"{best_metric}  rho={best_rho:+.3f}")

    # (c) do low-quality images cause the failed matches?
    print("\n(c) the worst genuine pairs -- are they low-quality images?")
    order = np.argsort(gs)
    worst = [gp[i] for i in order[:10]]
    best = [gp[i] for i in order[-10:]]
    qi_ratio = qkeys.index("aniso_w") if "aniso_w" in qkeys else 0
    for tag, sel in (("10 WORST pairs", worst), ("10 BEST  pairs", best)):
        qs = np.array([[qual[i, qi_ratio], qual[j, qi_ratio]] for i, j in sel])
        print(f"    {tag}: mean {qkeys[qi_ratio]} of the two images "
              f"= {qs.mean():.3f} (min-of-pair {qs.min(axis=1).mean():.3f})")

    # ---------------------------------------------------------------- 4 --
    print("\n" + "=" * 74)
    print("4. HOW MUCH AREA DO TWO PRESSES REALLY SHARE?")
    print("=" * 74)
    frame = 150 * 52
    for tag, z in (("baseline window (+-20,+-8)", base),
                   ("wide window (+-60,+-20)", wide)):
        GEO, BOTH, DX, DY = z["GEO"], z["BOTH"], z["DX"], z["DY"]
        gg = np.array([BOTH[i, j] for i, j in gp])
        ggeo = np.array([GEO[i, j] for i, j in gp])
        dx = np.array([abs(DX[i, j]) for i, j in gp])
        dy = np.array([abs(DY[i, j]) for i, j in gp])
        fa = np.array([min(fg_px[i], fg_px[j]) for i, j in gp])
        frac = gg / np.maximum(fa, 1)
        print(f"\n  {tag}, genuine pairs (n={len(gp)}):")
        print(f"    finger pixels per image: median {np.median(fg_px):.0f} "
              f"({np.median(fg_px)/frame*100:.0f}% of the frame)")
        for nm, v, pct in (("geometric overlap px", ggeo, True),
                           ("BOTH-finger overlap px", gg, True),
                           ("as frac of smaller finger area", frac, False),
                           ("|dx| at best align", dx, False),
                           ("|dy| at best align", dy, False)):
            p = np.percentile(v, [5, 25, 50, 75, 95])
            extra = f"  (median {p[2]/frame*100:.0f}% of frame)" if pct else ""
            print(f"    {nm:<32} p5={p[0]:7.1f} p25={p[1]:7.1f} "
                  f"med={p[2]:7.1f} p75={p[3]:7.1f} p95={p[4]:7.1f}{extra}")
        lim_x, lim_y = (20, 8) if tag.startswith("baseline") else (60, 20)
        atedge = float(((dx >= lim_x) | (dy >= lim_y)).mean())
        print(f"    genuine pairs whose best alignment sits ON the search-window "
              f"edge: {atedge*100:.0f}%")

    # overlap vs score
    GEO, BOTH = wide["GEO"], wide["BOTH"]
    gg = np.array([BOTH[i, j] for i, j in gp])
    r = spearman(gg, gs)
    print(f"\n  rho(shared finger area, LCN+NCC score) over genuine pairs = "
          f"{r:+.3f}  (p={perm_p(gg, gs, r, 5000):.4f})")
    ip = [(i, j) for i in gen.tolist() for j in imp.tolist()]
    igg = np.array([BOTH[i, j] for i, j in ip])
    print(f"  shared-area median: genuine {np.median(gg):.0f} px vs "
          f"impostor {np.median(igg):.0f} px  "
          f"(so overlap alone does NOT separate them)")


if __name__ == "__main__":
    main()
