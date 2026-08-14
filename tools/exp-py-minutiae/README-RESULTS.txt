py-minutiae: measured results (see final.py / oracle.py / repeat.py / stability.py)

Reproduce:
  cd /home/melb/projects/elan-0c6e-linux/tools/exp-py-minutiae
  python3 baseline.py     # caches S_baseline.npy, validates the harness
  python3 final.py        # the headline table
  python3 oracle.py       # the ceiling analysis (the decisive number)
