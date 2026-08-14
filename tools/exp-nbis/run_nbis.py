#!/usr/bin/env python3
"""Run MINDTCT+BOZORTH3 over a preprocessed dataset variant and evaluate."""
import subprocess, sys, os, json
from pathlib import Path
import numpy as np
import evalproto as ep

HERE = Path(__file__).resolve().parent
BIN = HERE / "nbisbatch"


def run_variant(root, ppmm=19.685, rmperim=0, binary=None, tag=None):
    """root: directory containing right-index/ right-index-cover/ right-middle/"""
    root = Path(root)
    man = ep.manifest(root)
    mf = HERE / "manifest.txt"
    mf.write_text("\n".join(str(p) for _, p in man) + "\n")
    b = str(binary or BIN)
    out = subprocess.run([b, str(mf), str(ppmm), str(rmperim)],
                         capture_output=True, text=True, check=True).stdout
    n = len(man)
    S = np.zeros((n, n))
    counts = np.zeros(n, int)
    rel = np.zeros(n)
    for line in out.splitlines():
        p = line.split()
        if p[0] == "M":
            counts[int(p[1])] = int(p[2]); rel[int(p[1])] = float(p[3])
        elif p[0] == "S":
            S[int(p[1]), int(p[2])] = int(p[3])
    return S, counts, rel, man


def count_report(tag, counts, man):
    idx = ep.index_sets(man)
    parts = []
    for lbl, ii in idx.items():
        c = counts[ii]
        parts.append(f"{lbl}: mean {c.mean():5.1f} med {np.median(c):4.1f} "
                     f"min {c.min():3d} max {c.max():3d} ok>=10 {int((c>=10).sum())}/{len(c)}")
    print(f"[{tag}] minutiae  " + " | ".join(parts))
    return counts.mean(), float((counts >= 10).mean())


if __name__ == "__main__":
    variants = sys.argv[1:] or ["prep/s1nn/raw"]
    for v in variants:
        root = HERE / v if not os.path.isabs(v) else Path(v)
        ppmm = 19.685
        if "/s2" in str(root):
            ppmm = 39.37
        S, counts, rel, man = run_variant(root, ppmm=ppmm)
        count_report(v, counts, man)
        # sanity: self-comparison must be near-maximal
        diag = np.diag(S)
        off = S[~np.eye(len(S), dtype=bool)]
        print(f"    sanity: self-score mean {diag.mean():.0f} max {diag.max():.0f}; "
              f"off-diag max {off.max():.0f} -> self should dominate")
        ep.summarise(v, S, man)
