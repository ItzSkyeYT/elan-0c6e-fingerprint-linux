"""
Leak / sanity controls for the v5 matcher.

  1. self-match           an image against itself must score at or above any
                          other image (already asserted in every run)
  2. reversed roles       treat right-middle as the genuine finger and
                          right-index(+cover) as the impostor.  A real matcher
                          separates that pairing too; a matcher that has latched
                          onto something about the *right-index* folders would
                          not.
  3. NULL control         split right-index-cover at random into two halves and
                          call one of them "impostor".  Both halves are the same
                          finger, so d-prime must come out near zero.  If this
                          control produces a healthy d-prime, the pipeline is
                          leaking and every other number is meaningless.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto import load_all, _metrics, N_ENROLL_A, N_TRIALS, SEED


def generic_scenario(M, gen_pool, imp_pool, n_enroll=N_ENROLL_A,
                     n_trials=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_trials):
        tmpl = list(rng.choice(gen_pool, size=min(n_enroll, len(gen_pool) - 1),
                               replace=False))
        gen, imp = [], []
        for p in gen_pool:
            sub = [t for t in tmpl if t != p]
            if not sub:
                continue
            gen.append(max(M[t, p] for t in sub))
        for p in imp_pool:
            imp.append(max(M[t, p] for t in tmpl))
        ds.append(_metrics(gen, imp)["dprime"])
    return float(np.mean(ds)), float(np.std(ds))


if __name__ == "__main__":
    imgs, _, idx = load_all()
    f = sys.argv[1] if len(sys.argv) > 1 else "V5_v5_t9_p3.npz"
    key = sys.argv[2] if len(sys.argv) > 2 else "top16"
    M = np.load(f)[key]
    n = M.shape[0]

    print(f"controls for {f}:{key}")
    bad = sum(1 for i in range(n)
              if M[i, i] < max(M[j, i] for j in range(n) if j != i))
    print(f"  1. self-match           {n - bad}/{n} images score highest against "
          f"themselves")

    idxpool = idx["right-index"] + idx["right-index-cover"]
    d, s = generic_scenario(M, idxpool, idx["right-middle"])
    print(f"  2a. index  vs middle    d' = {d:5.2f} +/- {s:.2f}")
    d, s = generic_scenario(M, idx["right-middle"], idxpool)
    print(f"  2b. middle vs index     d' = {d:5.2f} +/- {s:.2f}   "
          f"(roles reversed; must also separate)")

    rng = np.random.default_rng(7)
    cov = np.array(idx["right-index-cover"])
    nulls = []
    for _ in range(20):
        perm = rng.permutation(cov)
        a, b = list(perm[:10]), list(perm[10:])
        d, _s = generic_scenario(M, a, b, n_enroll=5, n_trials=4)
        nulls.append(d)
    print(f"  3. NULL (cover split)   d' = {np.mean(nulls):5.2f} +/- "
          f"{np.std(nulls):.2f} over 20 random splits   (must be ~0)")

    # score distributions, for eyeballing
    gp = idxpool
    ip = idx["right-middle"]
    gg = [M[a, b] for a in gp for b in gp if a != b]
    gi = [M[a, b] for a in gp for b in ip]
    print(f"  genuine  pairwise mean {np.mean(gg):.3f} sd {np.std(gg):.3f} "
          f"max {np.max(gg):.3f}")
    print(f"  impostor pairwise mean {np.mean(gi):.3f} sd {np.std(gi):.3f} "
          f"max {np.max(gi):.3f}")
    print(f"  genuine pairs above the best impostor pair: "
          f"{sum(1 for v in gg if v > max(gi))}/{len(gg)}")
