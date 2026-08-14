#!/usr/bin/env python3
"""
Follow-ups that test the interpretation of s1.

  6  Are the alignments we measure REAL? Transitivity test on genuine triples:
     if dx(i,j) is a true registration then dx(i,j)+dx(j,k) must equal dx(i,k).
     Impostor triples give the null.  Without this, every overlap number in
     section 4 is meaningless.
  7  Is a LOW-quality capture a bad TEMPLATE? (which images actually win the
     max over the template set, and are they the high-quality ones?)
  8  VERIFY-SIDE-ONLY gate: keep every template, but let the device refuse a
     poor PROBE and re-prompt. Does capture quality predict which probe fails?
  9  Best-case re-prompt: an oracle that re-presses whenever the first probe
     would be rejected -- the ceiling on what any verify-time gate can buy.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                          # noqa: E402
from s1_measure import (composite, rank01, spearman, perm_p, metrics,   # noqa: E402
                        score_probe, protoA, protoB, keep_top)

CLASSES = ["right-index", "right-index-cover", "right-middle"]


def main():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    base = np.load(HERE / "scores_base.npz")
    wide = np.load(HERE / "scores_widestrict.npz")
    S = base["S"]
    labels = c["labels"]; qual = c["qual"]; fg_px = c["fg_px"]
    qkeys = [str(k) for k in c["qkeys"]]
    q = composite(qual, qkeys, ["ridge_band", "aniso_w", "coh_w", "usable_frac"])

    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]

    # ================================================== 6. transitivity test
    print("=" * 78)
    print("6. ARE THE MEASURED ALIGNMENTS REAL? (transitivity of registration)")
    print("=" * 78)
    print("   For a triple i,j,k the shifts must compose: d(i,j)+d(j,k)=d(i,k).")
    print("   Random/meaningless alignments do not compose. Impostor triples,")
    print("   which cannot have a true registration, are the null distribution.")
    for tag, z, lim in (("matcher window +-20/+-8", base, 20),
                        ("wide +-60/+-20 strict", wide, 60)):
        DX, DY = z["DX"], z["DY"]
        for what, pool in (("genuine ", gen), ("impostor", imp)):
            trip = list(itertools.combinations(pool.tolist(), 3))
            rng = np.random.default_rng(0)
            if len(trip) > 4000:
                trip = [trip[t] for t in rng.choice(len(trip), 4000, replace=False)]
            err = []
            for i, j, k in trip:
                ex = DX[i, j] + DX[j, k] - DX[i, k]
                ey = DY[i, j] + DY[j, k] - DY[i, k]
                err.append(abs(ex) + abs(ey))
            err = np.array(err)
            # null: shuffle the shift assignments
            allx = DX[np.ix_(pool, pool)][np.triu_indices(len(pool), 1)]
            ally = DY[np.ix_(pool, pool)][np.triu_indices(len(pool), 1)]
            nerr = []
            for _ in range(len(trip)):
                a, b, d = rng.integers(0, len(allx), 3)
                nerr.append(abs(allx[a] + allx[b] - allx[d]) +
                            abs(ally[a] + ally[b] - ally[d]))
            nerr = np.array(nerr)
            print(f"   [{tag:<24}] {what} n={len(trip):5d}  "
                  f"|closure error| med {np.median(err):5.1f} px  "
                  f"(shuffled null {np.median(nerr):5.1f})  "
                  f"exact-ish (<=4px): {(err<=4).mean()*100:4.1f}% "
                  f"(null {(nerr<=4).mean()*100:4.1f}%)")

    # =============================================== 7. template usefulness
    print("\n" + "=" * 78)
    print("7. IS A LOW-QUALITY CAPTURE A BAD TEMPLATE?")
    print("=" * 78)
    print("   For every genuine probe, which enrolled capture wins the max?")
    print("   'win rate' = fraction of the other 30 genuine probes for which")
    print("   this image is the single best template.")
    Sg = S[np.ix_(gen, gen)]
    wins = np.zeros(len(gen))
    for pi in range(len(gen)):
        col = Sg[:, pi].copy()
        col[pi] = -np.inf
        wins[int(np.argmax(col))] += 1
    wins /= len(gen)
    mean_s = np.nanmean(Sg, axis=1)
    print(f"   rho(composite quality, template win rate) = {spearman(q[gen], wins):+.3f}"
          f"  p={perm_p(q[gen], wins, seed=11):.4f}")
    print(f"   rho(composite quality, own mean genuine score) = "
          f"{spearman(q[gen], mean_s):+.3f}")
    order = np.argsort(-wins)
    print(f"\n   {'rank':>4} {'image':<30} {'quality':>8} {'win rate':>9} {'mean score':>11}")
    for r in list(order[:5]) + list(order[-5:]):
        nm = str(c["names"][gen[r]])
        cls = CLASSES[labels[gen[r]]]
        print(f"   {list(order).index(r)+1:4d} {cls[-12:] + '/' + nm[-9:]:<30} "
              f"{q[gen[r]]:8.3f} {wins[r]:9.2f} {mean_s[r]:11.3f}")

    lowq = q[gen] <= np.percentile(q[gen], 33)
    print(f"\n   bottom-third-quality genuine captures win the max for "
          f"{wins[lowq].sum()*len(gen):.0f} of {len(gen)} probes "
          f"({wins[lowq].sum()*100:.0f}%); they are {lowq.mean()*100:.0f}% of the set.")

    # ============================================ 8. verify-side-only gate
    print("\n" + "=" * 78)
    print("8. VERIFY-SIDE-ONLY GATE: does quality predict which PROBE fails?")
    print("=" * 78)
    print("   All templates kept. Drop the worst N% of PROBES (genuine AND")
    print("   impostor, so the impostor side is not quietly advantaged).")
    print("   Control: drop the same number of probes at random, 500 draws.")
    for f in (0.10, 0.20, 0.30):
        rik = keep_top(ri, q, f); impk = keep_top(imp, q, f)
        d, e, fa = protoB(S, rc, rik, impk)
        rng = np.random.default_rng(77)
        ctl = np.array([protoB(S, rc,
                               np.sort(rng.choice(ri, len(rik), replace=False)),
                               np.sort(rng.choice(imp, len(impk), replace=False)))
                        for _ in range(500)])
        p = (np.sum(ctl[:, 0] >= d) + 1) / 501
        print(f"   B drop {f*100:3.0f}%  gated d'={d:5.2f} EER={e*100:5.1f}% "
              f"FAR@10={fa*100:5.1f}%   random d'={ctl[:,0].mean():5.2f}"
              f"+-{ctl[:,0].std():.2f}  one-sided p={p:.3f}")

        gpool = gen
        gp_keep = keep_top(gen, q, f)
        ag = protoA(S, gpool, gp_keep, impk, 19, 300, 0)[0]
        ac = []
        rng = np.random.default_rng(78)
        for _ in range(200):
            ac.append(protoA(S, gpool,
                             np.sort(rng.choice(gen, len(gp_keep), replace=False)),
                             np.sort(rng.choice(imp, len(impk), replace=False)),
                             19, 12, 7)[0])
        ac = np.array(ac)
        pa = (np.sum(ac[:, 0] >= ag[0]) + 1) / 201
        print(f"   A drop {f*100:3.0f}%  gated d'={ag[0]:5.2f} EER={ag[1]*100:5.1f}% "
              f"FAR@10={ag[2]*100:5.1f}%   random d'={ac[:,0].mean():5.2f}"
              f"+-{ac[:,0].std():.2f}  one-sided p={pa:.3f}")

    # ================================================= 9. re-prompt ceiling
    print("\n" + "=" * 78)
    print("9. RE-PRESS CEILING: what if the device simply asked for MORE presses?")
    print("=" * 78)
    print("   Score a genuine 'attempt' as the max over m independent probe")
    print("   captures (the user presses up to m times, any accept wins). The")
    print("   impostor gets the same m attempts, so FAR rises too. This is the")
    print("   honest version of 'just retry', and it needs no quality metric.")
    rng = np.random.default_rng(21)
    gsc = np.array([score_probe(S, rc, p) for p in ri])
    isc = np.array([score_probe(S, rc, p) for p in imp])
    for m in (1, 2, 3):
        gg, ii = [], []
        for _ in range(2000):
            gg.append(max(rng.choice(gsc, m)))
            ii.append(max(rng.choice(isc, m)))
        d, e, fa = metrics(gg, ii)
        print(f"   m={m} attempts:  d'={d:5.2f}  EER={e*100:5.1f}%  "
              f"FAR@10%FRR={fa*100:5.1f}%")
    print("   (d' is roughly flat: retrying moves genuine and impostor")
    print("    distributions together, so it buys convenience, not security.)")


if __name__ == "__main__":
    main()
