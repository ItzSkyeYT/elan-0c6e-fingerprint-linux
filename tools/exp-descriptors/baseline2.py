"""Re-measure the correlation baselines under proto.py, so the descriptor
numbers are compared against something computed the same way."""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto import (load_all, scenario_A, scenario_B, summarise, selfcheck,
                   local_contrast_norm, gabor_enhance, _rotate)          # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastncc import ncc_all_shifts

ROTS = (-12, -8, -4, 0, 4, 8, 12)


def build(enh, rots=ROTS, max_dx=20, max_dy=8, min_overlap=3500):
    imgs, labels, idx = load_all()
    E = [enh(im) for im in imgs]
    R = [[_rotate(e, d) if d else e for d in rots] for e in E]
    memo = {}

    def score(i, j):
        k = (i, j)
        v = memo.get(k)
        if v is None:
            v = max(ncc_all_shifts(E[i], R[j][r], max_dx, max_dy, min_overlap)
                    for r in range(len(rots)))
            memo[k] = v
        return v
    return score, idx, len(imgs)


if __name__ == "__main__":
    for name, enh in (("LCN + rot + NCC", local_contrast_norm),
                      ("Gabor + rot + NCC", gabor_enhance)):
        t0 = time.time()
        score, idx, n = build(enh)
        selfcheck(score, n, name)
        tA, gA, iA = scenario_A(score, idx)
        mB, gB, iB = scenario_B(score, idx)
        summarise(name, tA, gA, iA, mB, gB, iB, extra=f"({time.time()-t0:.0f}s)")
