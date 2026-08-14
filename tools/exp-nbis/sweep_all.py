#!/usr/bin/env python3
"""Every preprocessing variant x MINDTCT x BOZORTH3, under protocols A and B."""
import numpy as np
import run_nbis as R
import evalproto as ep

TAGS = [("s1nn", 1), ("s2nn", 2), ("s2bilin", 2), ("s3nn", 3)]
VARS = ["raw", "lcn", "lcn_eq", "bp_eq", "gabor", "gabor_eq"]

rows = []
for tag, sc in TAGS:
    for v in VARS:
        p = f"prep/{tag}/{v}"
        root = R.HERE / p
        if not root.exists():
            continue
        S, c, rel, man = R.run_variant(root, ppmm=19.685 * sc)
        A, B = ep.summarise(p, S, man, verbose=False)
        idx = ep.index_sets(man)
        g = idx["right-index"] + idx["right-index-cover"]
        i = idx["right-middle"]
        gp = np.array([S[a, b] for ai, a in enumerate(g) for b in g[ai + 1:]])
        ip = np.array([S[a, b] for a in g for b in i])
        rows.append((p, c.mean(), float((c >= 10).mean()), np.diag(S).mean(),
                     gp.mean(), gp.max(), ip.mean(), ip.max(),
                     A["dprime"], A["eer"], A["far10"],
                     B["dprime"], B["eer"], B["far10"]))

hdr = ("variant", "minu", ">=10", "self", "gpair", "gmax", "ipair", "imax",
       "A_d", "A_eer", "A_far", "B_d", "B_eer", "B_far")
print(f"{hdr[0]:22s} {hdr[1]:>5s} {hdr[2]:>5s} {hdr[3]:>6s} {hdr[4]:>6s} {hdr[5]:>5s} "
      f"{hdr[6]:>6s} {hdr[7]:>5s} | {hdr[8]:>6s} {hdr[9]:>6s} {hdr[10]:>6s} "
      f"| {hdr[11]:>6s} {hdr[12]:>6s} {hdr[13]:>6s}")
for r in rows:
    print(f"{r[0]:22s} {r[1]:5.1f} {r[2]*100:4.0f}% {r[3]:6.1f} {r[4]:6.2f} {r[5]:5.0f} "
          f"{r[6]:6.2f} {r[7]:5.0f} | {r[8]:6.2f} {r[9]*100:5.1f}% {r[10]*100:5.1f}% "
          f"| {r[11]:6.2f} {r[12]*100:5.1f}% {r[13]*100:5.1f}%")
print("\nself = bozorth3(image, itself); must dominate off-diagonal or the harness is wrong.")
