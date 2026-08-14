#!/usr/bin/env python3
"""Diagnostics for blockdesc: (1) does it recover a KNOWN transform, and
(2) what do genuine vs impostor pairs actually look like inside the matcher."""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import blockdesc as BD
import evaluate as EV
import enh

W, H = 150, 52


def shift_img(img, dx, dy):
    out = np.zeros_like(img)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    out[ys0:ys1, xs0:xs1] = img[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def detail(ta, tb, cfg):
    """Re-run score_one but return the winning alignment as well."""
    rots = np.asarray(cfg["rots"], np.float32)
    nrot, nq = len(rots), ta.nq
    S = (ta.qd.reshape(nrot * nq, -1) @ tb.db.T).reshape(nrot * nq, tb.ny, tb.nx)
    nrows = nrot * nq
    rows = np.arange(nrows)[:, None, None]
    dd = np.arange(-cfg["nms"], cfg["nms"] + 1)
    cp, cs = [], []
    flat = S.reshape(nrows, -1)
    for _ in range(cfg["topk"]):
        k = flat.argmax(axis=1)
        cp.append(k); cs.append(flat[np.arange(nrows), k])
        py, px = np.divmod(k, tb.nx)
        S[rows, np.clip(py[:, None, None] + dd[None, :, None], 0, tb.ny - 1),
          np.clip(px[:, None, None] + dd[None, None, :], 0, tb.nx - 1)] = -2.0
    src = np.concatenate([np.tile(np.arange(nq), nrot)] * cfg["topk"])
    rid = np.concatenate([np.repeat(np.arange(nrot), nq)] * cfg["topk"])
    pos = np.concatenate(cp); sim = np.concatenate(cs)
    keep = sim >= cfg["min_sim"]
    src, rid, pos, sim = src[keep], rid[keep], pos[keep], sim[keep]
    py, px = np.divmod(pos, tb.nx)
    th = np.radians(rots[rid]); ct, st = np.cos(th), np.sin(th)
    ax, ay = ta.qc[src, 0], ta.qc[src, 1]
    tx = (px + tb.r) - (ct * ax + st * ay)
    ty = (py + tb.r) - (-st * ax + ct * ay)
    wt = np.maximum(sim - cfg["sim_floor"], 1e-3)
    bt = cfg["bin_t"]
    nbx, nby = int(2 * W / bt) + 3, int(2 * H / bt) + 3
    fx, fy = (tx + W) / bt, (ty + H) / bt
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    gx, gy = fx - ix, fy - iy
    acc = np.zeros(nbx * nby * nrot, np.float32)
    for dx in (0, 1):
        wx = gx if dx else 1 - gx
        jx = np.clip(ix + dx, 0, nbx - 1)
        for dy in (0, 1):
            wy = gy if dy else 1 - gy
            jy = np.clip(iy + dy, 0, nby - 1)
            np.add.at(acc, (jx * nby + jy) * nrot + rid, wt * wx * wy)
    best = int(np.argmax(acc))
    cx = -W + (best // (nrot * nby) + 0.5) * bt
    cy = -H + ((best // nrot) % nby + 0.5) * bt
    cr = best % nrot
    inl = ((np.abs(tx - cx) <= cfg["tol_t"]) & (np.abs(ty - cy) <= cfg["tol_t"]) &
           (np.abs(rid - cr) <= cfg["rot_slack"]))
    bw = np.zeros(nq); np.maximum.at(bw, src[inl], wt[inl])
    th0 = math.radians(float(rots[cr])); c0, s0 = math.cos(th0), math.sin(th0)
    mx = c0 * ta.qc[:, 0] + s0 * ta.qc[:, 1] + cx
    my = -s0 * ta.qc[:, 0] + c0 * ta.qc[:, 1] + cy
    ov = int(((mx >= tb.r) & (mx <= W - 1 - tb.r) &
              (my >= tb.r) & (my <= H - 1 - tb.r)).sum())
    return dict(cx=cx, cy=cy, rot=float(rots[cr]), cnt=int((bw > 0).sum()),
                nq=nq, ov=ov, meansim=float(sim.mean()) if len(sim) else 0.0)


def main():
    cfg = dict(BD.DEFAULT)
    for kv in sys.argv[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = eval(v)
    imgs, labels, names = EV.load()

    print("--- known-transform recovery (image 0) ---")
    a = imgs[0]
    ta = BD.make_template(a, cfg)
    for dx, dy in [(0, 0), (10, 3), (-14, -4), (25, 0)]:
        tb = BD.make_template(shift_img(a, dx, dy), cfg)
        d = detail(ta, tb, cfg)
        print(f"  true t=({dx:4d},{dy:3d}) rot=0   -> recovered "
              f"t=({d['cx']:6.1f},{d['cy']:5.1f}) rot={d['rot']:5.1f} "
              f"cnt={d['cnt']:3d}/{d['nq']:3d} ov={d['ov']:3d}")
    for deg in (-6, 6, 12):
        tb = BD.make_template(enh._rotate(a, deg), cfg)
        d = detail(ta, tb, cfg)
        print(f"  true rot={deg:4d} (about centre)  -> recovered "
              f"t=({d['cx']:6.1f},{d['cy']:5.1f}) rot={d['rot']:5.1f} "
              f"cnt={d['cnt']:3d}/{d['nq']:3d} ov={d['ov']:3d}")

    print("\n--- real pairs ---")
    T = [BD.make_template(im, cfg) for im in imgs]
    gi = [i for i, l in enumerate(labels) if l != "right-middle"]
    ii = [i for i, l in enumerate(labels) if l == "right-middle"]
    rng = np.random.default_rng(0)
    print("  genuine:")
    for _ in range(8):
        i, j = rng.choice(gi, 2, replace=False)
        d = detail(T[i], T[j], cfg)
        print(f"    {i:2d}->{j:2d}  t=({d['cx']:6.1f},{d['cy']:5.1f}) rot={d['rot']:5.1f} "
              f"cnt={d['cnt']:3d}/{d['nq']:3d} ov={d['ov']:3d} sim={d['meansim']:.2f}")
    print("  impostor:")
    for _ in range(8):
        i = int(rng.choice(gi)); j = int(rng.choice(ii))
        d = detail(T[i], T[j], cfg)
        print(f"    {i:2d}->{j:2d}  t=({d['cx']:6.1f},{d['cy']:5.1f}) rot={d['rot']:5.1f} "
              f"cnt={d['cnt']:3d}/{d['nq']:3d} ov={d['ov']:3d} sim={d['meansim']:.2f}")


if __name__ == "__main__":
    main()
