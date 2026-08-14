"""
Score-level fusion of the descriptor matcher and the correlation baseline.

The two make different errors: NCC needs broad overlap, the descriptor vote
needs a handful of distinctive patches to agree geometrically.  If the errors
are even partly independent, the sum of the two calibrated scores should beat
either.

Calibration must not touch the impostor set.  Both scores are divided by the
standard deviation of that matcher over GENUINE-POOL pairs only (the 31x31
within-pool comparisons, diagonal excluded) -- statistics an enrolled device
actually has.  No impostor image contributes to any constant used here.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto import load_all, scenario_A, scenario_B, summarise, _metrics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastncc import ncc_all_shifts
from proto import local_contrast_norm, _rotate                     # noqa

CACHE = Path(__file__).resolve().parent


def ncc_matrix(rots=(-12, -8, -4, 0, 4, 8, 12), max_dx=20, max_dy=8,
               min_overlap=3500, tag="lcnrot"):
    f = CACHE / f"M_ncc_{tag}.npy"
    if f.exists():
        return np.load(f)
    imgs, _, idx = load_all()
    E = [local_contrast_norm(im) for im in imgs]
    R = [[_rotate(e, d) if d else e for d in rots] for e in E]
    n = len(imgs)
    M = np.zeros((n, n), np.float32)
    t0 = time.time()
    for i in range(n):
        for j in range(n):
            M[i, j] = max(ncc_all_shifts(E[i], R[j][r], max_dx, max_dy, min_overlap)
                          for r in range(len(rots)))
    print(f"  NCC matrix built ({time.time()-t0:.0f}s)")
    np.save(f, M)
    return M


def calib_scale(M, gen_pool):
    """Spread of this matcher over genuine-pool pairs only (no impostor data)."""
    sub = M[np.ix_(gen_pool, gen_pool)].astype(np.float64)
    off = sub[~np.eye(len(gen_pool), dtype=bool)]
    return float(off.std()) or 1.0


def evaluate_matrix(M, idx, name):
    sc = lambda i, j: float(M[i, j])
    n = M.shape[0]
    bad = sum(1 for i in range(n)
              if M[i, i] < max(M[j, i] for j in range(n) if j != i))
    tA, gA, iA = scenario_A(sc, idx)
    mB, gB, iB = scenario_B(sc, idx)
    r = summarise(name, tA, gA, iA, mB, gB, iB, extra=f"(selffail {bad})")
    return r


if __name__ == "__main__":
    imgs, _, idx = load_all()
    gen_pool = idx["right-index"] + idx["right-index-cover"]
    Mn = ncc_matrix()
    sn = calib_scale(Mn, gen_pool)

    for f in sorted(CACHE.glob(sys.argv[1] if len(sys.argv) > 1 else "v3_*.npz")):
        z = np.load(f)
        for key in z.files:
            Md = z[key]
            sd = calib_scale(Md, gen_pool)
            if sd <= 0:
                continue
            for w in (0.5, 1.0, 2.0):
                F = Mn / sn + w * (Md / sd)
                evaluate_matrix(F, idx, f"FUSE ncc + {w}*{f.stem}:{key}")
