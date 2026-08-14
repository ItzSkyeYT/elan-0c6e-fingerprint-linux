#!/bin/sh
# Full reproduction of the capture-quality investigation. ~15 min.
set -e
cd "$(dirname "$0")"
python v1_sanity.py            # harness sanity: self-NCC==1, fast==naive, no leaks
python v2_build.py             # directional 45x45 score cache + quality metrics
python v3_analysis.py          # sections 1-7: metrics, correlation, gate, overlap
python v4_why.py               # sections 8-9: why the gate fails; score normalisation
python v5_window.py            # section 10: alignment search-window sweep
python v6_final.py             # section 11: final numbers + significance test
