import sys, time; sys.path.insert(0,'.')
import numpy as np, minutiae as M
from common import load_ds
ds = load_ds()
img = ds['right-index'][0][1]
mask = M.segment(img); n = M.local_contrast_norm(img)
th,coh = M.orientation(n); enh = M.gabor(n,th,0.110)
b = M.binarise(enh,mask); sk = M.thin(b)
print('skel px', sk.sum())
for sl,mr in [(4,6),(8,10),(12,16),(20,25)]:
    s = M.prune_spurs(sk, spur_len=sl, rounds=3, min_ridge=mr)
    cn = M._crossing_number(s)
    inner = M._erode(mask,6); inner[:6,:]=inner[-6:,:]=False; inner[:,:6]=inner[:,-6:]=False
    e=int((s&inner&(cn==1)).sum()); bf=int((s&inner&(cn==3)).sum())
    print(f'spur_len={sl} min_ridge={mr}: px {s.sum()}  end {e} bif {bf}')
s = M.prune_spurs(sk, spur_len=8, rounds=3, min_ridge=10)
for r in range(52):
    print(''.join('#' if s[r,c] else '.' for c in range(150)))
