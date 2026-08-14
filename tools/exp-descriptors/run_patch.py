"""Patch-descriptor matcher over several support radii; matrices saved for fusion.

v3's describe_patch masks out-of-image samples, so a keypoint may sit close to
the border (margin 4 rather than a full radius).  On a 52 px tall image that is
the difference between describing the whole finger and only its middle band.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desc_v3 import build, report, DEF3
from proto import load_all

imgs, _, idx = load_all()
CFGS = [
    ("patchR8",  dict(DEF3, desc="patch", R=8,  sstep=1, excl=10, simmin=0.30, step=4)),
    ("patchR12", dict(DEF3, desc="patch", R=12, sstep=2, excl=14, simmin=0.30, step=4)),
    ("patchR16", dict(DEF3, desc="patch", R=16, sstep=2, excl=18, simmin=0.30, step=4)),
    ("patchR20", dict(DEF3, desc="patch", R=20, sstep=3, excl=22, simmin=0.30, step=4)),
    ("bothR12",  dict(DEF3, desc="both",  R=12, sstep=2, excl=14, simmin=0.20, step=4)),
]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, cfg in CFGS:
        if only and only not in tag:
            continue
        print(f"\n===== {tag} =====", flush=True)
        Ms, _, n, counts = build(cfg, imgs, idx)
        report(Ms, idx, n, tag)
        np.savez(f"P_{tag}.npz", **Ms)
