import sys, time; sys.path.insert(0,'.')
import numpy as np, minutiae as M
from common import load_ds
ds = load_ds()
t0=time.time()
for lbl in ds:
    tot=[]
    for name,img in ds[lbl]:
        t = M.make_template(img)
        tot.append(t.n)
    a=np.array(tot)
    print(f"{lbl:20s} minutiae mean {a.mean():5.1f}  min {a.min()}  max {a.max()}")
print('time', time.time()-t0)
