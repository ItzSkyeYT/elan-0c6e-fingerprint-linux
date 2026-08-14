import sys, math, time
sys.path.insert(0,'/home/melb/projects/elan-0c6e-linux/tools')
sys.path.insert(0,'.')
import numpy as np
from pathlib import Path
import minutiae as M

def load_pgm(path):
    data = Path(path).read_bytes()
    fields, idx = [], 2
    while len(fields) < 3:
        while data[idx:idx+1].isspace(): idx += 1
        if data[idx:idx+1] == b"#":
            while data[idx] != 0x0A: idx += 1
            continue
        s = idx
        while not data[idx:idx+1].isspace(): idx += 1
        fields.append(int(data[s:idx]))
    idx += 1
    w,h,_ = fields
    return np.frombuffer(data[idx:idx+w*h], dtype=np.uint8).reshape(h,w).astype(np.float32)

root = Path('/home/melb/.local/share/elan-fp/dataset')
ds = {d.name: [(f.name, load_pgm(f)) for f in sorted(d.glob('*.pgm'))] for d in sorted(root.iterdir()) if d.is_dir()}

img = ds['right-index'][0][1]
t0=time.time()
mask = M.segment(img)
print('mask coverage', mask.mean())
n = M.local_contrast_norm(img)
th, coh = M.orientation(n)
fr = M.ridge_frequency(n, th)
print('freq map: mean %.4f min %.4f max %.4f'%(fr.mean(), fr.min(), fr.max()))
enh = M.gabor(n, th, 0.110)
print('enh', enh.min(), enh.max(), enh.std())
b = M.binarise(enh, mask)
print('binary ridge fraction', b[mask].mean())
sk = M.thin(b)
print('skel px', sk.sum(), 'time', time.time()-t0)
mm = M.extract_minutiae(sk, th, mask, ridge_period=9.0)
print('minutiae', len(mm), 'time', time.time()-t0)

# ascii of skeleton
for r in range(0,52):
    print(''.join('#' if sk[r,c] else ('.' if mask[r,c] else ' ') for c in range(150)))
