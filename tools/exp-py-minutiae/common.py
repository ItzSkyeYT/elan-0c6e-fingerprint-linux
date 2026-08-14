import sys, math
import numpy as np
from pathlib import Path

def load_pgm(path):
    data = Path(path).read_bytes(); fields, idx = [], 2
    while len(fields) < 3:
        while data[idx:idx+1].isspace(): idx += 1
        s = idx
        while not data[idx:idx+1].isspace(): idx += 1
        fields.append(int(data[s:idx]))
    idx += 1; w,h,_ = fields
    return np.frombuffer(data[idx:idx+w*h], dtype=np.uint8).reshape(h,w).astype(np.float32)

ROOT = Path('/home/melb/.local/share/elan-fp/dataset')

def load_ds(root=ROOT):
    return {d.name: [(f.name, load_pgm(f)) for f in sorted(d.glob('*.pgm'))]
            for d in sorted(Path(root).iterdir()) if d.is_dir()}

def dprime(g,i):
    g,i = np.asarray(g,float), np.asarray(i,float)
    den = math.sqrt((g.var()+i.var())/2)
    return float((g.mean()-i.mean())/den) if den>0 else 0.0

def eer(g,i):
    g,i = np.asarray(g,float), np.asarray(i,float)
    bg, be, bt = math.inf, 1.0, 0.0
    for t in np.unique(np.concatenate([g,i])):
        frr = float((g<t).mean()); far = float((i>=t).mean())
        if abs(frr-far) < bg: bg, be, bt = abs(frr-far), (frr+far)/2, float(t)
    return be, bt

def far_at_frr(g,i,target=0.10):
    g,i = np.asarray(g,float), np.asarray(i,float)
    t = float(np.quantile(g,target))
    return float((i>=t).mean()), t
