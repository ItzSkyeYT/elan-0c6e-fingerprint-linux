import sys, math, time
sys.path.insert(0,'.')
import numpy as np
from pathlib import Path
import minutiae as M

def load_pgm(path):
    data = Path(path).read_bytes(); fields, idx = [], 2
    while len(fields) < 3:
        while data[idx:idx+1].isspace(): idx += 1
        s = idx
        while not data[idx:idx+1].isspace(): idx += 1
        fields.append(int(data[s:idx]))
    idx += 1; w,h,_ = fields
    return np.frombuffer(data[idx:idx+w*h], dtype=np.uint8).reshape(h,w).astype(np.float32)

root = Path('/home/melb/.local/share/elan-fp/dataset')
ds = {d.name: [(f.name, load_pgm(f)) for f in sorted(d.glob('*.pgm'))] for d in sorted(root.iterdir()) if d.is_dir()}

for lbl in ds:
    tot=[]
    for name,img in ds[lbl]:
        mask = M.segment(img); n = M.local_contrast_norm(img)
        th,coh = M.orientation(n); enh = M.gabor(n,th,0.110)
        b = M.binarise(enh,mask); sk = M.thin(b)
        code = M._nbcode(sk); cn = np.zeros(sk.shape,np.int32)
        for i in range(8): cn += (code>>i)&1
        inner = M._erode(mask,6)
        inner[:6,:]=inner[-6:,:]=False; inner[:,:6]=inner[:,-6:]=False
        e = int((sk&inner&(cn==1)).sum()); bif = int((sk&inner&(cn==3)).sum())
        tot.append((e,bif))
    a=np.array(tot)
    print(f"{lbl:20s} endings {a[:,0].mean():5.1f}  bifs {a[:,1].mean():5.1f}  total {a.sum(1).mean():5.1f}  min {a.sum(1).min()} max {a.sum(1).max()}")
