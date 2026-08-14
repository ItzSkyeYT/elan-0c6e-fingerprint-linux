"""
Shared evaluation protocol for the ELAN 0c6e descriptor experiments.

One file, one definition of the two scenarios, so every matcher in this
directory is measured identically.  Nothing here selects a threshold or a
parameter using the impostor set.

  (A) POOLED   genuine = right-index + right-index-cover (31)
               impostor = right-middle (14)
               template-set vs probe, leave-one-out over the genuine pool,
               averaged over N_TRIALS *random* template subsets.
  (B) REALISTIC enrol = right-index-cover (19)
               probe = right-index (12), impostor = right-middle (14)

A matcher is a function score(i, j) -> float over dataset image INDICES, so
descriptors can be precomputed once and reused across the ~3400 comparisons
the protocol requires.  Index i is always the TEMPLATE, j always the PROBE.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_LAB = Path(__file__).resolve().parent.parent / "matcher-lab.py"
exec(_LAB.read_text().split("def main(")[0])          # noqa: F401  (load_pgm, dprime, ...)

DATASET = Path.home() / ".local/share/elan-fp/dataset"
N_TRIALS = 12          # >= 8 required by the protocol
N_ENROLL_A = 8         # template-subset size for scenario A
SEED = 20260809


def load_all(root=DATASET):
    """Return (images, labels, index_by_label).  Images are float32 52x150."""
    imgs, labels = [], []
    idx = {}
    for lbl in ("right-index", "right-index-cover", "right-middle"):
        d = Path(root) / lbl
        ids = []
        for f in sorted(d.glob("*.pgm")):
            ids.append(len(imgs))
            imgs.append(load_pgm(f))
            labels.append(lbl)
        idx[lbl] = ids
    return imgs, labels, idx


# --------------------------------------------------------------- scenarios --

def scenario_A(score, idx, n_enroll=N_ENROLL_A, n_trials=N_TRIALS, seed=SEED):
    """Pooled.  Returns (list of per-trial metric dicts, pooled gen, pooled imp).

    For each trial a fresh random template subset is drawn.  A genuine probe is
    scored against a subset drawn from the OTHER genuine images (true
    leave-one-out: the probe can never be its own template).  Impostor probes
    are scored against a subset of the full genuine pool.
    """
    gen_pool = idx["right-index"] + idx["right-index-cover"]
    imp_pool = idx["right-middle"]
    rng = np.random.default_rng(seed)

    trials, all_gen, all_imp = [], [], []
    for _ in range(n_trials):
        # one template subset per trial, used for the impostors
        tmpl = list(rng.choice(gen_pool, size=n_enroll, replace=False))
        gen, imp = [], []
        for p in gen_pool:
            others = [g for g in gen_pool if g != p]
            # keep the trial's subset where possible, top up if the probe was in it
            sub = [t for t in tmpl if t != p]
            if len(sub) < n_enroll:
                spare = [g for g in others if g not in sub]
                sub = sub + list(rng.choice(spare, size=n_enroll - len(sub),
                                            replace=False))
            gen.append(max(score(t, p) for t in sub))
        for p in imp_pool:
            imp.append(max(score(t, p) for t in tmpl))
        trials.append(_metrics(gen, imp))
        all_gen += gen
        all_imp += imp
    return trials, all_gen, all_imp


def scenario_B(score, idx):
    """Enrol on the 19 coverage captures, verify with the 12 habitual presses."""
    tmpl = idx["right-index-cover"]
    gen = [max(score(t, p) for t in tmpl) for p in idx["right-index"]]
    imp = [max(score(t, p) for t in tmpl) for p in idx["right-middle"]]
    return _metrics(gen, imp), gen, imp


# ----------------------------------------------------------------- metrics --

def _metrics(gen, imp):
    g, i = np.asarray(gen, float), np.asarray(imp, float)
    e, _ = eer(g, i)
    f, _ = far_at_frr(g, i, 0.10)
    return {"dprime": dprime(g, i), "eer": e, "far10": f,
            "gmean": float(g.mean()), "imean": float(i.mean())}


def summarise(name, trials_A, poolA_g, poolA_i, mB, genB, impB, extra=""):
    d = np.array([t["dprime"] for t in trials_A])
    e = np.array([t["eer"] for t in trials_A])
    f = np.array([t["far10"] for t in trials_A])
    pooled = _metrics(poolA_g, poolA_i)
    print(f"\n=== {name} === {extra}")
    print(f"  A (pooled, {len(trials_A)} random subsets of {N_ENROLL_A}):")
    print(f"      d' = {d.mean():.2f} +/- {d.std():.2f}   "
          f"EER = {e.mean()*100:.1f}%   FAR@10%FRR = {f.mean()*100:.1f}%")
    print(f"      [scores pooled over trials: d' = {pooled['dprime']:.2f}  "
          f"EER = {pooled['eer']*100:.1f}%  FAR@10 = {pooled['far10']*100:.1f}%]")
    print(f"  B (enrol 19 cover -> probe 12 index):")
    print(f"      d' = {mB['dprime']:.2f}   EER = {mB['eer']*100:.1f}%   "
          f"FAR@10%FRR = {mB['far10']*100:.1f}%")
    return {"name": name,
            "A": {"dprime": float(d.mean()), "dsd": float(d.std()),
                  "eer": float(e.mean()), "far10": float(f.mean())},
            "B": mB}


def selfcheck(score, n, tag=""):
    """A matcher must score an image against itself at or above anything else.
    Returns the number of images for which that fails."""
    bad = 0
    for i in range(n):
        s_self = score(i, i)
        s_other = max(score(j, i) for j in range(n) if j != i)
        if s_self < s_other:
            bad += 1
    if bad:
        print(f"  !! self-match check {tag}: {bad}/{n} images beaten by another image")
    else:
        print(f"  self-match check {tag}: OK ({n}/{n})")
    return bad
