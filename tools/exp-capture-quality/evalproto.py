"""
The two evaluation protocols, driven off a precomputed score matrix.

Protocol A (POOLED): genuine = right-index + right-index-cover (31),
    impostor = right-middle (14). One random template subset of size
    n_enroll per repetition; genuine probes are every genuine image NOT in
    that subset; impostor probes are all 14. Metrics averaged over >= 8
    repetitions.  No image is ever its own template (guaranteed: probes are
    drawn from the complement of the template set, and S has NaN on the
    diagonal so a self-comparison would blow up loudly rather than silently).

Protocol B (REALISTIC): enrol = all right-index-cover, probe = right-index,
    impostor = right-middle. Deterministic.
"""
import math
import numpy as np


def dprime(g, i):
    g, i = np.asarray(g, float), np.asarray(i, float)
    den = math.sqrt((g.var() + i.var()) / 2)
    return float((g.mean() - i.mean()) / den) if den > 0 else 0.0


def eer(g, i):
    g, i = np.asarray(g, float), np.asarray(i, float)
    best_gap, best = math.inf, 1.0
    for t in np.unique(np.concatenate([g, i])):
        frr = float((g < t).mean())
        far = float((i >= t).mean())
        if abs(frr - far) < best_gap:
            best_gap, best = abs(frr - far), (frr + far) / 2
    return best


def far_at_frr(g, i, target=0.10):
    g, i = np.asarray(g, float), np.asarray(i, float)
    t = float(np.quantile(g, target))
    return float((i >= t).mean())


def _maxscore(S, templates, probe):
    v = S[np.asarray(templates), probe]
    if np.isnan(v).any():
        raise AssertionError(f"self-comparison leak: probe {probe} in template set")
    return float(v.max())


def protocol_A(S, gen_idx, imp_idx, n_enroll=6, reps=16, seed=0):
    """Returns dict of mean metrics plus the per-rep arrays."""
    rng = np.random.default_rng(seed)
    gen_idx = np.asarray(gen_idx)
    imp_idx = np.asarray(imp_idx)
    n_enroll = min(n_enroll, len(gen_idx) - 1)
    ds, es, fs = [], [], []
    for _ in range(reps):
        T = rng.choice(gen_idx, size=n_enroll, replace=False)
        probes = np.setdiff1d(gen_idx, T)
        g = [_maxscore(S, T, p) for p in probes]
        i = [_maxscore(S, T, p) for p in imp_idx]
        ds.append(dprime(g, i))
        es.append(eer(g, i))
        fs.append(far_at_frr(g, i))
    return {"dprime": float(np.mean(ds)), "eer": float(np.mean(es)),
            "far10": float(np.mean(fs)),
            "dprime_sd": float(np.std(ds)), "eer_sd": float(np.std(es)),
            "far10_sd": float(np.std(fs)),
            "n_gen_probes": len(gen_idx) - n_enroll, "n_enroll": n_enroll,
            "reps": reps}


def protocol_B(S, enrol_idx, probe_idx, imp_idx):
    enrol_idx = np.asarray(enrol_idx)
    assert not set(enrol_idx.tolist()) & set(np.asarray(probe_idx).tolist())
    g = [_maxscore(S, enrol_idx, p) for p in probe_idx]
    i = [_maxscore(S, enrol_idx, p) for p in imp_idx]
    return {"dprime": dprime(g, i), "eer": eer(g, i),
            "far10": far_at_frr(g, i), "n_enroll": len(enrol_idx),
            "n_gen_probes": len(probe_idx), "gen": np.array(g),
            "imp": np.array(i)}
