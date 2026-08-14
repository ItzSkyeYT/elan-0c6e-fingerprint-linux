#!/usr/bin/env python3
"""Ceiling analysis: over ALL rigid alignments, what is the MAXIMUM number of
corresponding minutiae between two captures?  No matcher can beat this.  If
genuine and impostor pairs reach the same ceiling, minutiae carry no signal at
this sensor size."""
import math, sys, itertools
import numpy as np
sys.path.insert(0, '/home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae')
import minutiae as M
from common import load_ds

W,H = 150,52
CFG = {"merge_dist":0.0, "border":3, "spur_len":4, "min_ridge":6}

def best_count(ta, tb, pos_tol=10.0, use_dir=True, dir_tol=math.radians(30)):
    """Exhaustive: every (rot, translation) implied by a minutia pair is a
    candidate; the true optimum alignment is always one of them (up to tol)."""
    A,B = ta.m, tb.m
    if len(A)==0 or len(B)==0: return 0,0,0
    best = 0; bestov=(0,0)
    for rdeg in range(-24, 25, 3):
        t = math.radians(rdeg); ct,st = math.cos(t), math.sin(t)
        rax = ct*A[:,0]-st*A[:,1]; ray = st*A[:,0]+ct*A[:,1]
        adp = A[:,2]+t
        for i in range(len(A)):
            for j in range(len(B)):
                tx = B[j,0]-rax[i]; ty = B[j,1]-ray[i]
                ax = rax+tx; ay = ray+ty
                d = np.hypot(ax[:,None]-B[None,:,0], ay[:,None]-B[None,:,1])
                ok = d<=pos_tol
                if use_dir:
                    dd = (adp[:,None]-B[None,:,2])%math.pi
                    dd = np.minimum(dd, math.pi-dd)
                    ok &= dd<=dir_tol
                # greedy
                used=np.zeros(len(B),bool); m=0
                cost=np.where(ok,d,np.inf)
                order=np.argsort(cost,axis=None)
                ua=np.zeros(len(A),bool)
                for k in order:
                    if not np.isfinite(cost.flat[k]): break
                    p,q = divmod(int(k), len(B))
                    if ua[p] or used[q]: continue
                    ua[p]=used[q]=True; m+=1
                if m>best:
                    best=m
                    inA = (ax>=0)&(ax<W)&(ay>=0)&(ay<H)
                    bx = ct*(B[:,0]-tx)+st*(B[:,1]-ty); by=-st*(B[:,0]-tx)+ct*(B[:,1]-ty)
                    inB = (bx>=0)&(bx<W)&(by>=0)&(by<H)
                    bestov=(int(inA.sum()),int(inB.sum()))
    return best, bestov[0], bestov[1]

ds = load_ds()
gen = [im for _,im in ds['right-index']] + [im for _,im in ds['right-index-cover']]
imp = [im for _,im in ds['right-middle']]
Tg = [M.make_template(x,CFG) for x in gen]
Ti = [M.make_template(x,CFG) for x in imp]
print('minutiae/image: genuine %.1f  impostor %.1f'%(np.mean([t.n for t in Tg]), np.mean([t.n for t in Ti])))
rng = np.random.default_rng(3)
gp = [(i,j) for i,j in itertools.combinations(range(len(Tg)),2)]
ip = [(i,j) for i in range(len(Tg)) for j in range(len(Ti))]
gs = rng.choice(len(gp), 80, replace=False); iss = rng.choice(len(ip), 80, replace=False)
G = np.array([best_count(Tg[gp[k][0]], Tg[gp[k][1]]) for k in gs], float)
I = np.array([best_count(Tg[ip[k][0]], Ti[ip[k][1]]) for k in iss], float)
print('ORACLE max corresponding minutiae over all alignments')
print('  genuine  mean %.2f  median %.0f  max %.0f'%(G[:,0].mean(), np.median(G[:,0]), G[:,0].max()))
print('  impostor mean %.2f  median %.0f  max %.0f'%(I[:,0].mean(), np.median(I[:,0]), I[:,0].max()))
d = (G[:,0].mean()-I[:,0].mean())/math.sqrt((G[:,0].var()+I[:,0].var())/2)
print('  d-prime of the ORACLE matched count: %.2f'%d)
print('  genuine pairs reaching >=6 matches: %.0f%%   impostor: %.0f%%'%((G[:,0]>=6).mean()*100,(I[:,0]>=6).mean()*100))
