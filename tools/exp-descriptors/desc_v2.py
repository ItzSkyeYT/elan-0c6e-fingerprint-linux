"""
Descriptor matching v2.

The one idea v1 was missing: OVERLAP NORMALISATION.

If two presses share only 25% of their area, at most 25% of the probe's
keypoints CAN match, so a raw or sqrt(nt*np)-normalised vote count is small even
for a true genuine pair -- which is precisely the failure mode that makes
correlation plateau on this sensor.  Once the Hough stage has proposed a
translation, the number of probe keypoints that actually land inside the
template's valid area is known, so the score can be

        inlier mass / (number of keypoints that COULD have matched)

i.e. "what fraction of the shared region agreed", which is comparable across
pairs with wildly different overlap.  A floor on the denominator stops a
two-keypoint overlap scoring 1.0.

Also new:
  * mutual nearest neighbour (a match must be best in both directions)
  * a second Hough pass that re-collects inliers from ALL matches, not only the
    ratio-test survivors
  * top-k template fusion instead of plain max
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desc_v1 import build_views, H, W, DEFAULT
from proto import load_all, scenario_A, scenario_B, summarise, _metrics


def match_pair(vt, vp, DT2, cfg):
    """Return a dict of score variants for one template view vs one probe view."""
    nt, npb = vt.n, vp.n
    out = {}
    if nt < 5 or npb < 5:
        return None
    S = (vp.D @ vt.D.T)                                    # (np, nt)

    best = np.argmax(S, axis=1)
    sbest = S[np.arange(npb), best]

    # ratio test with spatial exclusion (adjacent grid patches overlap, so the
    # nearest rival must be taken from somewhere else on the finger)
    S2 = np.where(DT2[best] < cfg["excl"] ** 2, -2.0, S)
    s2 = S2.max(axis=1)
    keep = (sbest > cfg["simmin"]) & (s2 < cfg["ratio"] * sbest)

    if cfg["mutual"]:
        back = np.argmax(S, axis=0)                        # best probe per template
        keep &= (back[best] == np.arange(npb))

    if keep.sum() < 3:
        return None

    bt = vt.P[best]
    ty = bt[:, 0] - vp.P[:, 0]
    tx = bt[:, 1] - vp.P[:, 1]

    b = cfg["bin"]
    ny, nx = (2 * H) // b + 3, (2 * W) // b + 3
    acc = np.zeros((ny, nx), np.float32)
    iy = np.clip((ty[keep] + H) // b, 0, ny - 1).astype(np.int32)
    ix = np.clip((tx[keep] + W) // b, 0, nx - 1).astype(np.int32)
    wts = (sbest[keep] - cfg["simmin"]).astype(np.float32)
    np.add.at(acc, (iy, ix), wts)
    pooled = (acc[:-2, :-2] + acc[:-2, 1:-1] + acc[:-2, 2:] +
              acc[1:-1, :-2] + acc[1:-1, 1:-1] + acc[1:-1, 2:] +
              acc[2:, :-2] + acc[2:, 1:-1] + acc[2:, 2:])
    k = int(np.argmax(pooled))
    by, bx = divmod(k, pooled.shape[1])
    mass = float(pooled[by, bx])

    # centre of the winning translation cluster, from the votes it contains
    sel = (iy >= by) & (iy <= by + 2) & (ix >= bx) & (ix <= bx + 2)
    cnt = int(sel.sum())
    if cnt == 0:
        return None
    kidx = np.nonzero(keep)[0][sel]
    tyc = float(np.mean(ty[kidx]))
    txc = float(np.mean(tx[kidx]))

    # --- second pass: every probe keypoint whose winner agrees with (tyc,txc)
    tol = cfg["tol"]
    agree = (np.abs(ty - tyc) <= tol) & (np.abs(tx - txc) <= tol) & \
            (sbest > cfg["simmin"])
    mass2 = float((sbest[agree] - cfg["simmin"]).sum())
    cnt2 = int(agree.sum())

    # --- how many probe keypoints COULD have matched at this translation?
    py = vp.P[:, 0] + tyc
    px = vp.P[:, 1] + txc
    lo_y, hi_y = vt.P[:, 0].min(), vt.P[:, 0].max()
    lo_x, hi_x = vt.P[:, 1].min(), vt.P[:, 1].max()
    inside = (py >= lo_y) & (py <= hi_y) & (px >= lo_x) & (px <= hi_x)
    n_ov = int(inside.sum())

    fl = cfg["ovfloor"]
    den = max(n_ov, fl)
    geo = math.sqrt(nt * npb)

    out["sqrt"] = mass / geo
    out["cnt"] = cnt / geo
    out["ov_mass"] = mass / den
    out["ov_cnt"] = cnt / den
    out["ov2_mass"] = mass2 / den
    out["ov2_cnt"] = cnt2 / den
    # geometric compromise: rewards fraction matched but still prefers evidence
    out["mix"] = math.sqrt(max(mass, 0.0)) * math.sqrt(mass / den)
    out["mix2"] = math.sqrt(max(mass2, 0.0)) * math.sqrt(mass2 / den)
    out["nov"] = float(n_ov)
    return out


VARIANTS = ("sqrt", "cnt", "ov_mass", "ov_cnt", "ov2_mass", "ov2_cnt",
            "mix", "mix2")

DEF2 = dict(DEFAULT, mutual=False, tol=6.0, ovfloor=40)


def build(cfg, imgs=None, idx=None, verbose=True):
    if imgs is None:
        imgs, _, idx = load_all()
    t0 = time.time()
    rots = cfg["rots"]
    views = [build_views(im, cfg, rots) for im in imgs]
    r0 = len(rots) // 2
    counts = [v[r0].n for v in views]
    if verbose:
        print(f"  keypoints/image mean {np.mean(counts):.0f} "
              f"[{min(counts)}..{max(counts)}]  dim {views[0][0].D.shape[1]}  "
              f"({time.time()-t0:.0f}s extract)")
    DT2 = [((v[r0].P[None, :, :] - v[r0].P[:, None, :]) ** 2).sum(-1).astype(np.float32)
           for v in views]
    n = len(imgs)
    Ms = {v: np.zeros((n, n), np.float32) for v in VARIANTS}
    for i in range(n):
        vt = views[i][r0]
        for j in range(n):
            bestv = {v: 0.0 for v in VARIANTS}
            for r in range(len(rots)):
                res = match_pair(vt, views[j][r], DT2[i], cfg)
                if res is None:
                    continue
                for v in VARIANTS:
                    if res[v] > bestv[v]:
                        bestv[v] = res[v]
            for v in VARIANTS:
                Ms[v][i, j] = bestv[v]
    if verbose:
        print(f"  {n*n} pairs ({time.time()-t0:.0f}s)")
    return Ms, idx, n, counts


def report(Ms, idx, n, tag, top=99, quiet=False):
    rows = []
    for v, M in Ms.items():
        sc = lambda i, j: float(M[i, j])
        bad = sum(1 for i in range(n)
                  if M[i, i] < max(M[j, i] for j in range(n) if j != i))
        tA, _, _ = scenario_A(sc, idx)
        mB, _, _ = scenario_B(sc, idx)
        rows.append(dict(v=v, bad=bad,
                         dA=float(np.mean([t["dprime"] for t in tA])),
                         sdA=float(np.std([t["dprime"] for t in tA])),
                         eA=float(np.mean([t["eer"] for t in tA])),
                         fA=float(np.mean([t["far10"] for t in tA])),
                         dB=mB["dprime"], eB=mB["eer"], fB=mB["far10"]))
    rows.sort(key=lambda r: -(r["dA"] + r["dB"]))
    if not quiet:
        for r in rows[:top]:
            print(f"  {tag:20s} {r['v']:10s} selffail {r['bad']:2d}  "
                  f"A d'={r['dA']:5.2f}+/-{r['sdA']:.2f} EER={r['eA']*100:5.1f}% "
                  f"FAR10={r['fA']*100:5.1f}%  |  B d'={r['dB']:5.2f} "
                  f"EER={r['eB']*100:5.1f}% FAR10={r['fB']*100:5.1f}%", flush=True)
    return rows


if __name__ == "__main__":
    imgs, _, idx = load_all()
    cfgs = [
        ("v2 lcn r8 s4", dict(DEF2)),
        ("v2 lcn r8 s3", dict(DEF2, step=3)),
        ("v2 lcn r12 s4", dict(DEF2, r=12, sub=3, step=4)),
        ("v2 mutual", dict(DEF2, mutual=True)),
        ("v2 ovfloor 80", dict(DEF2, ovfloor=80)),
    ]
    for tag, cfg in cfgs:
        print(f"\n===== {tag} =====", flush=True)
        Ms, idx2, n, counts = build(cfg, imgs, idx)
        report(Ms, idx, n, tag)
