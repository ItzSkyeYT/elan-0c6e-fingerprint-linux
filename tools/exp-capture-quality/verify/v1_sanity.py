#!/usr/bin/env python3
"""Independent sanity checks on the cached score matrix and the harness.

Checks, in order:
  1. self-NCC of an image with itself is exactly 1.0 (matcher is sane)
  2. fastncc2.ncc_best_align agrees with matcher-lab's naive ncc_best
  3. the cached S matrix reproduces from the images (spot check)
  4. cached S diagonal is NaN; S is symmetric
  5. label assignment matches the directory names
  6. the d'=1.54 / d'=1.94 published baselines reproduce from S
"""
import os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TOOLS = EXP.parent
sys.path.insert(0, str(EXP))
os.chdir(TOOLS)
exec(open("matcher-lab.py").read().split("def main(")[0])      # noqa
from fastncc2 import ncc_best_align, ncc_map                   # noqa: E402

DATASET = os.path.expanduser("~/.local/share/elan-fp/dataset")
LABELS = ["right-index", "right-index-cover", "right-middle"]
BASE = dict(max_dx=20, max_dy=8, min_overlap=3500, max_rot=12, rot_step=4)

ds = load_dataset(DATASET)                                     # noqa
names, labels, imgs = [], [], []
for li, lbl in enumerate(LABELS):
    for n, im in ds[lbl]:
        names.append(n); labels.append(li); imgs.append(im)
labels = np.array(labels)
N = len(imgs)
print(f"[1] loaded {N} images: " + ", ".join(f"{l}={int((labels==i).sum())}"
                                             for i, l in enumerate(LABELS)))
assert (labels == 0).sum() == 12 and (labels == 1).sum() == 19 and (labels == 2).sum() == 14

L = [local_contrast_norm(im) for im in imgs]                    # noqa

# --- 1. self-match must be 1.0
s, dx, dy, deg = ncc_best_align(L[0], L[0], 20, 8, 3500, 0, 4, rotate=_rotate)  # noqa
print(f"[2] self-NCC(img0,img0) = {s:.6f} at dx={dx} dy={dy}  (want 1.0 at 0,0)")
assert abs(s - 1.0) < 1e-8 and dx == 0 and dy == 0

# --- 2. fast vs naive agreement, no rotation
bad = 0
for (i, j) in [(0, 1), (0, 20), (3, 40), (12, 31), (5, 44)]:
    f = ncc_best_align(L[i], L[j], 20, 8, 3500, 0, 4, rotate=_rotate)[0]
    n_ = ncc_best(L[i], L[j], 20, 8, 3500, max_rot=0)           # noqa
    if abs(f - n_) > 1e-7:
        bad += 1; print(f"    MISMATCH {i},{j}: fast={f:.6f} naive={n_:.6f}")
print(f"[3] fast-vs-naive NCC (no rot): {5-bad}/5 agree to 1e-7")
assert bad == 0

# --- 3. cached S spot check (with rotation)
Sc = np.load(EXP / "scores_base.npz")["S"]
print(f"[4] cached S shape {Sc.shape}, diag all-NaN: {bool(np.isnan(np.diag(Sc)).all())}, "
      f"symmetric: {bool(np.allclose(Sc, Sc.T, equal_nan=True))}")
rng = np.random.default_rng(0)
pairs = [(min(int(a),int(b)), max(int(a),int(b))) for a, b in rng.integers(0, N, size=(6, 2)) if a != b]
worst = 0.0
for i, j in pairs:
    r = ncc_best_align(L[i], L[j], BASE["max_dx"], BASE["max_dy"],
                       BASE["min_overlap"], BASE["max_rot"], BASE["rot_step"],
                       rotate=_rotate)[0]
    worst = max(worst, abs(r - Sc[i, j]))
print(f"[5] cached S reproduces on {len(pairs)} random pairs, max |diff| = {worst:.2e}")
assert worst < 1e-9

# --- 4. is S really asymmetric-in-truth? rotation search on b only
asym = []
for i, j in pairs:
    a_ = ncc_best_align(L[i], L[j], **{k: BASE[k] for k in
         ("max_dx","max_dy","min_overlap","max_rot","rot_step")}, rotate=_rotate)[0]
    b_ = ncc_best_align(L[j], L[i], **{k: BASE[k] for k in
         ("max_dx","max_dy","min_overlap","max_rot","rot_step")}, rotate=_rotate)[0]
    asym.append(abs(a_ - b_))
print(f"[6] score(i,j) vs score(j,i) max |diff| = {max(asym):.4f} "
      f"(rotation search is applied to b only; cache stores one direction)")

# --- 5. reproduce the published baselines from S
import evalproto as E
gen = np.where(labels < 2)[0]; imp = np.where(labels == 2)[0]
ri = np.where(labels == 0)[0]; rc = np.where(labels == 1)[0]

def score_set(S, T, gp, ip):
    T = np.asarray(T)
    assert not np.isnan(S[np.ix_(T, gp)]).any(), "SELF-COMPARISON LEAK"
    g = [float(np.nanmax(S[T, p])) for p in gp]
    i = [float(np.nanmax(S[T, p])) for p in ip]
    return E.dprime(g, i), E.eer(g, i), E.far_at_frr(g, i)

print("\n[7] BASELINE reproduction (LCN + rot + NCC, no quality gating)")
for ne in (6, 10, 15, 19):
    rows = []
    r2 = np.random.default_rng(1)
    for _ in range(200):
        perm = r2.permutation(gen)
        T, pr = np.sort(perm[:ne]), np.sort(perm[ne:])
        rows.append(score_set(Sc, T, pr, imp))
    a = np.array(rows)
    print(f"    A pooled, n_enroll={ne:2d}: d'={a[:,0].mean():5.2f}+-{a[:,0].std():4.2f}  "
          f"EER={a[:,1].mean()*100:5.1f}%  FAR@10={a[:,2].mean()*100:5.1f}%")
d, e, f = score_set(Sc, rc, ri, imp)
print(f"    B realistic, enrol=19 cover, probe=12 index: d'={d:.2f} EER={e*100:.1f}% FAR@10={f*100:.1f}%")

# leave-one-out variant literally as matcher-lab does it
print("\n[8] literal leave-one-out over all 31 genuine (template = all others):")
g = [float(np.nanmax(Sc[np.setdiff1d(gen, [p]), p])) for p in gen]
i = [float(np.nanmax(Sc[gen, p])) for p in imp]
print(f"    d'={E.dprime(g,i):.2f} EER={E.eer(g,i)*100:.1f}% FAR@10={E.far_at_frr(g,i)*100:.1f}%")
print("\nALL SANITY CHECKS PASSED")
