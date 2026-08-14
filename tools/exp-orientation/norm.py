#!/usr/bin/env python3
"""
Template-set SCORE-NORMALISATION rules, evaluated on the cached pair matrices.

`max over templates` throws away information that the driver already has: the
whole vector of scores of one probe against every enrolled template.  A genuine
probe tends to resemble a FEW templates strongly and the rest weakly (placements
overlap or they do not); an impostor that scores high does so uniformly, because
what it matches is the generic ridge flow every finger shares.  So the SHAPE of
the score vector carries information the maximum discards.

Rules implemented (all use only enrolled data + the probe -- no labels, no test
statistics, so none of them can leak):

    max          max_t s_t                              (baseline)
    peak         max_t s_t - mean_t s_t
    zpeak        (max_t s_t - mean_t s_t) / std_t s_t
    top2gap      mean of top-2  - mean of the rest
    znorm        max_t (s_t - mu_t) / sd_t, where mu_t, sd_t come from the
                 template-vs-template scores of template t (Z-norm: discounts
                 a "promiscuous" template that scores high against everything)
    znorm_peak   znorm, then subtract the mean of the normalised vector

Run:  python3 norm.py [matrix ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oflib import *                      # noqa: F401,F403  (dprime/eer/far_at_frr)
from run import load_flat, CACHE          # noqa


# ------------------------------------------------------------------ rules --

def _stats_of(v):
    v = np.asarray(v, float)
    return v.max(), v.mean(), v.std()


def rule_max(v, tt):
    return float(np.max(v))


def rule_peak(v, tt):
    return float(np.max(v) - np.mean(v))


def rule_zpeak(v, tt):
    s = np.std(v)
    return float((np.max(v) - np.mean(v)) / s) if s > 1e-9 else 0.0


def rule_top2gap(v, tt):
    o = np.sort(np.asarray(v, float))[::-1]
    if o.size < 4:
        return float(o[0])
    return float(o[:2].mean() - o[2:].mean())


def rule_znorm(v, tt):
    """tt[i] = (mu_i, sd_i) from template i's scores against the OTHER templates."""
    mu, sd = tt
    return float(np.max((np.asarray(v, float) - mu) / np.maximum(sd, 1e-6)))


def rule_znorm_peak(v, tt):
    mu, sd = tt
    z = (np.asarray(v, float) - mu) / np.maximum(sd, 1e-6)
    return float(z.max() - z.mean())


RULES = {
    "max": rule_max,
    "peak": rule_peak,
    "zpeak": rule_zpeak,
    "top2gap": rule_top2gap,
    "znorm": rule_znorm,
    "znorm_peak": rule_znorm_peak,
}


def template_stats(M, T):
    """Per-template mean/sd over its scores against the other templates.
    Uses only enrolled images."""
    T = np.asarray(T)
    sub = M[np.ix_(T, T)].astype(float).copy()
    n = len(T)
    mu = np.empty(n)
    sd = np.empty(n)
    for i in range(n):
        other = np.delete(sub[i], i)
        mu[i] = other.mean()
        sd[i] = other.std()
    return mu, sd


def score_vec(M, T, p):
    return np.asarray(M[np.asarray(T), p], float)


# ------------------------------------------------------------ evaluation --

def _m(gen, imp):
    return dprime(gen, imp), eer(gen, imp)[0], far_at_frr(gen, imp, 0.10)[0]


def eval_A(M, gen_idx, imp_idx, rule, n_enroll=6, trials=16, seed=1234):
    rng = np.random.default_rng(seed)
    gen_idx = np.asarray(gen_idx)
    imp_idx = np.asarray(imp_idx)
    out = []
    for _ in range(trials):
        T = rng.choice(gen_idx, size=n_enroll, replace=False)
        Ts = set(T.tolist())
        probes = [p for p in gen_idx.tolist() if p not in Ts]
        tt = template_stats(M, T)
        g = [rule(score_vec(M, T, p), tt) for p in probes]
        i = [rule(score_vec(M, T, q), tt) for q in imp_idx.tolist()]
        out.append(_m(g, i))
    a = np.array(out)
    return a.mean(0), a.std(0)


def eval_B(M, T, probes, imps, rule):
    tt = template_stats(M, T)
    g = [rule(score_vec(M, T, p), tt) for p in probes]
    i = [rule(score_vec(M, T, q), tt) for q in imps]
    return _m(g, i)


def main():
    imgs, labels = load_flat()
    ii = np.where(labels == "right-index")[0]
    ic = np.where(labels == "right-index-cover")[0]
    im = np.where(labels == "right-middle")[0]
    pool = np.concatenate([ii, ic])

    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(CACHE)
                                   if f.endswith(".npy"))
    print(f"{'matrix':<20}{'rule':<12}{'A d':>7}{'A EER':>8}{'A FAR10':>9}"
          f"{'B d':>7}{'B EER':>8}{'B FAR10':>9}")
    best = []
    for nm in names:
        M = np.load(os.path.join(CACHE, nm + ".npy"))
        if M.ndim == 3:
            M = M[4]
            nm = nm + "[w.4]"
        for rn, rule in RULES.items():
            (a, _), b = eval_A(M, pool, im, rule), eval_B(M, ic, ii, im, rule)
            print(f"{nm:<20}{rn:<12}{a[0]:7.2f}{a[1]*100:7.1f}%{a[2]*100:8.1f}%"
                  f"{b[0]:7.2f}{b[1]*100:7.1f}%{b[2]*100:8.1f}%")
            best.append((b[0], a[0], nm, rn))
        print()
    print("top by scenario-B d':")
    for b, a, nm, rn in sorted(best, reverse=True)[:12]:
        print(f"   {nm:<20}{rn:<12} B d'={b:5.2f}   A d'={a:5.2f}")


if __name__ == "__main__":
    main()
