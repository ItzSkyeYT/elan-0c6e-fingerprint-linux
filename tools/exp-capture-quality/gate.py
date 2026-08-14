#!/usr/bin/env python3
"""
Stage 3: what would a capture-time quality gate actually buy?

Two ways to apply a gate, measured separately because they mean different
things operationally:

  ENROL-ONLY  -- refuse to enrol a poor capture, but accept whatever the user
                 presents at verification time. Probe sets are UNCHANGED, so
                 this is a clean like-for-like comparison against the baseline.

  ENROL+VERIFY -- also refuse a poor capture at verification and re-prompt.
                 Genuine and impostor probe sets both shrink. This costs the
                 user retries and it shrinks the evaluation set, so it needs a
                 control.

THE CONTROL that makes this honest: every gated result is compared against
dropping the SAME NUMBER of images AT RANDOM (averaged over many draws). If a
quality-ordered drop does no better than a random drop, the metric carries no
information and the "gain" is just set shrinkage / luck.

Dropping is done WITHIN each class. A global drop would preferentially delete
right-middle captures (they are measurably lower quality than the index
captures), which would flatter the impostor distribution for a reason that has
nothing to do with quality gating.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalproto as E                                        # noqa: E402

N_ENROLL_A = 10
REPS_A = 24
RAND_DRAWS = 40


def load():
    c = np.load(HERE / "cache.npz", allow_pickle=True)
    S = np.load(HERE / "scores_base.npz")["S"]
    return c, S


def composite_quality(qual, qkeys, metrics, ranks=True):
    """Average of per-metric z-scores (or ranks) across the chosen metrics."""
    cols = []
    for m in metrics:
        v = qual[:, qkeys.index(m)].astype(float)
        if ranks:
            o = np.argsort(np.argsort(v)).astype(float)
            cols.append(o / (len(v) - 1))
        else:
            cols.append((v - v.mean()) / (v.std() + 1e-12))
    return np.mean(cols, axis=0)


def keep_by_quality(idx, q, drop_frac):
    """Keep the top (1-drop_frac) of idx by q. Always keeps at least 2."""
    n_keep = max(2, int(round(len(idx) * (1 - drop_frac))))
    order = np.argsort(-q[idx])
    return np.sort(idx[order[:n_keep]])


def keep_random(idx, n_keep, rng):
    return np.sort(rng.choice(idx, size=min(n_keep, len(idx)), replace=False))


def run_A(S, gen_pool, gen_probes, imp_probes, n_enroll, reps, seed):
    """Protocol A with an explicitly separated enrol POOL and probe set."""
    rng = np.random.default_rng(seed)
    n_enroll = min(n_enroll, len(gen_pool) - 1)
    ds, es, fs = [], [], []
    for _ in range(reps):
        T = rng.choice(gen_pool, size=n_enroll, replace=False)
        pr = np.setdiff1d(gen_probes, T)
        if len(pr) < 3:
            continue
        g = [float(np.nanmax(S[T, p])) for p in pr]
        i = [float(np.nanmax(S[T, p])) for p in imp_probes]
        assert not np.isnan(S[np.ix_(T, pr)]).any(), "self-comparison leak"
        ds.append(E.dprime(g, i))
        es.append(E.eer(g, i))
        fs.append(E.far_at_frr(g, i))
    return (float(np.mean(ds)), float(np.mean(es)), float(np.mean(fs)),
            float(np.std(ds)))


def run_B(S, enrol, probes, imp):
    assert not (set(enrol.tolist()) & set(probes.tolist()))
    g = [float(np.nanmax(S[enrol, p])) for p in probes]
    i = [float(np.nanmax(S[enrol, p])) for p in imp]
    return E.dprime(g, i), E.eer(g, i), E.far_at_frr(g, i), 0.0


def main():
    c, S = load()
    labels = c["labels"]
    qual = c["qual"]
    qkeys = [str(k) for k in c["qkeys"]]

    gen = np.where(labels < 2)[0]
    imp = np.where(labels == 2)[0]
    ri = np.where(labels == 0)[0]
    rc = np.where(labels == 1)[0]

    # candidate quality rankings to gate on
    CANDIDATES = {
        "ridge_band": composite_quality(qual, qkeys, ["ridge_band"]),
        "aniso_w": composite_quality(qual, qkeys, ["aniso_w"]),
        "coh_w": composite_quality(qual, qkeys, ["coh_w"]),
        "usable_frac": composite_quality(qual, qkeys, ["usable_frac"]),
        "bp_energy": composite_quality(qual, qkeys, ["bp_energy"]),
        "composite": composite_quality(
            qual, qkeys, ["ridge_band", "aniso_w", "coh_w", "usable_frac"]),
    }

    print("=" * 78)
    print("3. WHAT DOES A QUALITY GATE BUY?")
    print(f"   protocol A: n_enroll={N_ENROLL_A}, {REPS_A} random template "
          f"subsets; random-drop control averaged over {RAND_DRAWS} draws")
    print("=" * 78)

    # ---------------------------------------------------------- baseline --
    a0 = run_A(S, gen, gen, imp, N_ENROLL_A, REPS_A, 0)
    b0 = run_B(S, rc, ri, imp)
    print(f"\nBASELINE (no gate)")
    print(f"  A  d'={a0[0]:5.2f} (sd {a0[3]:.2f})  EER={a0[1]*100:5.1f}%  "
          f"FAR@10%FRR={a0[2]*100:5.1f}%")
    print(f"  B  d'={b0[0]:5.2f}              EER={b0[1]*100:5.1f}%  "
          f"FAR@10%FRR={b0[2]*100:5.1f}%")

    for mode in ("ENROL-ONLY", "ENROL+VERIFY"):
        print("\n" + "-" * 78)
        print(f"GATE MODE: {mode}")
        print("-" * 78)
        print(f"{'metric':<13} {'drop':>5} | {'A d-prime':>20} | "
              f"{'A EER%':>16} | {'B d-prime':>20}")
        print(f"{'':<13} {'':>5} | {'gated':>9} {'random':>10} | "
              f"{'gated':>7} {'rand':>8} | {'gated':>9} {'random':>10}")
        for mname, q in CANDIDATES.items():
            for f in (0.10, 0.20, 0.30):
                # ---- gated
                gen_pool = keep_by_quality(gen, q, f)
                imp_keep = keep_by_quality(imp, q, f)
                rc_keep = keep_by_quality(rc, q, f)
                ri_keep = keep_by_quality(ri, q, f)
                if mode == "ENROL-ONLY":
                    gp, ip, rip = gen, imp, ri
                else:
                    gp, ip, rip = gen_pool, imp_keep, ri_keep
                ag = run_A(S, gen_pool, gp, ip, N_ENROLL_A, REPS_A, 0)
                bg = run_B(S, rc_keep, rip, ip)

                # ---- random-drop control, same sizes
                rng = np.random.default_rng(1234)
                ar, br = [], []
                for _ in range(RAND_DRAWS):
                    gpool_r = keep_random(gen, len(gen_pool), rng)
                    imp_r = keep_random(imp, len(imp_keep), rng)
                    rc_r = keep_random(rc, len(rc_keep), rng)
                    ri_r = keep_random(ri, len(ri_keep), rng)
                    if mode == "ENROL-ONLY":
                        gp2, ip2, rip2 = gen, imp, ri
                    else:
                        gp2, ip2, rip2 = gpool_r, imp_r, ri_r
                    ar.append(run_A(S, gpool_r, gp2, ip2, N_ENROLL_A, 8, 7))
                    br.append(run_B(S, rc_r, rip2, ip2))
                ar = np.array([x[:3] for x in ar])
                br = np.array([x[:3] for x in br])

                print(f"{mname:<13} {f*100:4.0f}% | {ag[0]:9.2f} "
                      f"{ar[:,0].mean():6.2f}+-{ar[:,0].std():.2f} | "
                      f"{ag[1]*100:6.1f} {ar[:,1].mean()*100:7.1f} | "
                      f"{bg[0]:9.2f} {br[:,0].mean():6.2f}+-{br[:,0].std():.2f}")

    # ------------------------------------------------ the oracle ceiling --
    print("\n" + "=" * 78)
    print("ORACLE CEILING: gate on the thing a gate cannot see")
    print("=" * 78)
    print("Rank images by their MEAN genuine score (i.e. cheat -- use the")
    print("match outcome itself) and drop the worst. This is the absolute")
    print("upper bound on what ANY per-image gate could achieve, and it is")
    print("optimistically biased because the ranking peeked at the test data.")
    Sg = S[np.ix_(gen, gen)]
    oracle = np.full(len(labels), -1e9)
    oracle[gen] = np.nanmean(Sg, axis=1)
    Si = S[np.ix_(imp, imp)]
    oracle[imp] = np.nanmean(Si, axis=1)
    for f in (0.10, 0.20, 0.30):
        gen_pool = keep_by_quality(gen, oracle, f)
        rc_keep = keep_by_quality(rc, oracle, f)
        ri_keep = keep_by_quality(ri, oracle, f)
        ao = run_A(S, gen_pool, gen, imp, N_ENROLL_A, REPS_A, 0)
        bo = run_B(S, rc_keep, ri, imp)
        print(f"  drop {f*100:3.0f}% (enrol-only)  A d'={ao[0]:5.2f} "
              f"EER={ao[1]*100:5.1f}%   B d'={bo[0]:5.2f} EER={bo[1]*100:5.1f}%")


if __name__ == "__main__":
    main()
