"""
FINAL matcher: one configuration, both protocols, all controls, no variant menu.

    ./final.py            evaluate the chosen configuration
    ./final.py --fuse     also evaluate the sum with the LCN+rotation NCC score

Chosen configuration (see README block at the bottom of desc_v5.py for how it
was arrived at):

    enhancement     local contrast normalisation, sigma 6
    keypoints       dense grid, step 4, margin 8, inside a ridge-validity mask
                    -> ~304 keypoints per 150x52 image
    descriptor      Gaussian-windowed 8-radius patch of the enhanced image,
                    sampled every 2 px inside the disc (61 taps), mean-removed
                    and L2-normalised, so a dot product is a correlation
    rotation        global: the probe is re-described at -12/-6/0/6/12 degrees
    alignment       NN + ratio test (0.92) with a 12 px spatial exclusion,
                    then a Hough vote over translation (5 px bins, 3x3 pooled),
                    top 3 peaks kept
    score           at each candidate alignment, every probe keypoint takes the
                    best descriptor similarity among template keypoints within
                    9 px of its predicted position (elastic snap); the score is
                    the mean of the 16 best such values, maximised over
                    alignments and rotations
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desc_v5 import build, DEF5
from proto import load_all, scenario_A, scenario_B, summarise
from controls import generic_scenario

CFG = dict(DEF5, ltol=9, npeak=3)
KEY = "top16"


def main():
    imgs, _, idx = load_all()
    t0 = time.time()
    Ms, _, n, counts = build(CFG, imgs, idx)
    M = Ms[KEY]
    np.save(Path(__file__).parent / "M_final.npy", M)

    print(f"\nkeypoints per image: mean {np.mean(counts):.0f} "
          f"[{min(counts)}..{max(counts)}]")
    bad = sum(1 for i in range(n)
              if M[i, i] < max(M[j, i] for j in range(n) if j != i))
    print(f"self-match control: {n-bad}/{n} images score highest against themselves")

    sc = lambda i, j: float(M[i, j])
    tA, gA, iA = scenario_A(sc, idx)
    mB, gB, iB = scenario_B(sc, idx)
    summarise(f"descriptor v5 {KEY}", tA, gA, iA, mB, gB, iB,
              extra=f"({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(7)
    cov = np.array(idx["right-index-cover"])
    nulls = []
    for _ in range(20):
        p = rng.permutation(cov)
        d, _s = generic_scenario(M, list(p[:10]), list(p[10:]), n_enroll=5, n_trials=4)
        nulls.append(d)
    print(f"  NULL control (cover split in half, same finger both sides): "
          f"d' = {np.mean(nulls):.2f} +/- {np.std(nulls):.2f}   (must be ~0)")

    if "--fuse" in sys.argv:
        from fuse import ncc_matrix, calib_scale
        Mn = ncc_matrix()
        gen_pool = idx["right-index"] + idx["right-index-cover"]
        F = Mn / calib_scale(Mn, gen_pool) + M / calib_scale(M, gen_pool)
        scf = lambda i, j: float(F[i, j])
        tA, gA, iA = scenario_A(scf, idx)
        mB, gB, iB = scenario_B(scf, idx)
        summarise("FUSION: descriptor v5 + LCN/rot NCC", tA, gA, iA, mB, gB, iB)
        np.save(Path(__file__).parent / "M_fusion.npy", F)


if __name__ == "__main__":
    main()
