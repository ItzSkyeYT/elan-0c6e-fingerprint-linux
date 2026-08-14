#!/usr/bin/env python3
"""
Stage 4: how much of the genuine-score variance is a per-IMAGE property (which
a capture-time quality gate could in principle fix) and how much is
PAIR-specific (placement -- which it cannot)?

Additive two-way model on the genuine score matrix:

    s_ij = mu + a_i + a_j + e_ij

a_i is image i's own 'matchability' -- everything intrinsic to that capture.
e_ij is what is left: the interaction between two particular placements.
Fitted by least squares (symmetric design, so it is a small normal system;
solved with a plain Gauss-Seidel sweep so nothing here needs a linear-algebra
library either).

The variance split is the headline number for this whole investigation. A
quality gate can only ever attack var(a); if var(e) dominates, no gate helps.

Then: enrolment QUALITY versus enrolment DIVERSITY, measured head to head.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                        # noqa: E402
from gate import composite_quality, run_B, run_A             # noqa: E402


def fit_additive(S, idx, iters=500):
    """Fit s_ij = mu + a_i + a_j over the off-diagonal entries of S[idx,idx]."""
    M = S[np.ix_(idx, idx)].astype(float)
    n = len(idx)
    obs = ~np.isnan(M)
    np.fill_diagonal(obs, False)
    mu = np.nanmean(M[obs])
    a = np.zeros(n)
    for _ in range(iters):
        for i in range(n):
            r = M[i][obs[i]] - mu - a[obs[i]]
            a[i] = r.mean()
        a -= a.mean()
    pred = mu + a[:, None] + a[None, :]
    resid = M[obs] - pred[obs]
    total = M[obs] - mu
    return a, mu, float(np.var(total)), float(np.var(resid)), obs, M, pred


def greedy_diverse(S, pool, k):
    """Pick k templates that are as DISSIMILAR to each other as possible --
    i.e. that between them cover the most distinct placements. Greedy
    farthest-point on the score matrix (low score == little shared area).
    """
    pool = list(pool)
    M = S.copy()
    # seed: the pair with the lowest mutual score
    best = None
    for i, j in itertools.combinations(pool, 2):
        if best is None or M[i, j] < best[0]:
            best = (M[i, j], i, j)
    sel = [best[1], best[2]]
    while len(sel) < k:
        rest = [p for p in pool if p not in sel]
        # add whichever candidate is least similar to its closest chosen one
        scores = [(max(M[p, s] for s in sel), p) for p in rest]
        scores.sort()
        sel.append(scores[0][1])
    return np.sort(np.array(sel))


def greedy_quality(pool, q, k):
    pool = np.asarray(pool)
    return np.sort(pool[np.argsort(-q[pool])[:k]])


def main():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    S = np.load(HERE / "scores_base.npz")["S"]
    labels = c["labels"]
    qual = c["qual"]
    qkeys = [str(k) for k in c["qkeys"]]

    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]

    print("=" * 78)
    print("5. VARIANCE DECOMPOSITION OF THE GENUINE SCORE MATRIX")
    print("=" * 78)
    a, mu, vtot, vres, obs, M, pred = fit_additive(S, gen)
    print(f"  genuine pairs n={int(obs.sum())//2}   mean score {mu:.4f}")
    print(f"  total variance of genuine pair scores      {vtot:.6f}")
    print(f"  variance left after per-image effects      {vres:.6f}")
    print(f"  -> explained by PER-IMAGE quality/matchability : "
          f"{(1-vres/vtot)*100:5.1f}%")
    print(f"  -> PAIR-SPECIFIC (placement interaction)       : "
          f"{vres/vtot*100:5.1f}%")
    print(f"  spread of the per-image effect a_i: sd={a.std():.4f}, "
          f"range {a.min():+.4f}..{a.max():+.4f}")

    print("\n  how well do the measurable quality metrics predict a_i?")
    from analyze import spearman, perm_p
    for k in qkeys:
        q = qual[gen, qkeys.index(k)]
        r = spearman(q, a)
        if abs(r) > 0.25:
            print(f"    {k:<16} rho={r:+.3f}  p={perm_p(q, a, r, 5000):.4f}")

    best = max(qkeys, key=lambda k: abs(spearman(qual[gen, qkeys.index(k)], a)))
    rb = spearman(qual[gen, qkeys.index(best)], a)
    frac = (1 - vres / vtot) * rb ** 2
    print(f"\n  BEST measurable metric: {best} (rho={rb:+.3f})")
    print(f"  A gate using it could address at most rho^2 x (per-image share)")
    print(f"  = {rb**2*100:.0f}% x {(1-vres/vtot)*100:.0f}% = "
          f"{frac*100:.1f}% of the total genuine-score variance.")

    # same for impostors, as a sanity check
    ai, mui, vt_i, vr_i, _, _, _ = fit_additive(S, imp)
    print(f"\n  (impostor matrix, for contrast: per-image share "
          f"{(1-vr_i/vt_i)*100:.1f}%)")

    print("\n" + "=" * 78)
    print("6. ENROLMENT QUALITY vs ENROLMENT DIVERSITY (protocol B)")
    print("=" * 78)
    print("  enrol k of the 19 right-index-cover captures, probe with the 12")
    print("  right-index captures, impostors = 14 right-middle.")
    print(f"  {'k':>3} | {'top-k by quality':>17} | {'k random (mean+-sd)':>21} "
          f"| {'k most DIVERSE':>15}")
    q = composite_quality(qual, qkeys, ["ridge_band", "aniso_w", "coh_w",
                                        "usable_frac"])
    rng = np.random.default_rng(11)
    for k in (3, 5, 7, 9, 11, 13, 15, 17, 19):
        eq = greedy_quality(rc, q, k)
        ed = greedy_diverse(S, rc, k) if k >= 2 else eq
        dq = run_B(S, eq, ri, imp)[0]
        dd = run_B(S, ed, ri, imp)[0]
        rr = [run_B(S, np.sort(rng.choice(rc, k, replace=False)), ri, imp)[0]
              for _ in range(60)]
        print(f"  {k:3d} | {dq:17.2f} | {np.mean(rr):13.2f}+-{np.std(rr):5.2f} "
              f"| {dd:15.2f}")

    print("\n  (the diverse selection PEEKS at the score matrix among the")
    print("   enrol images only -- it never touches a probe -- so it is a")
    print("   legitimate enrol-time strategy, implementable on device.)")


if __name__ == "__main__":
    main()
