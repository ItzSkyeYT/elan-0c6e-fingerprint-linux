#!/bin/sh
# Full reproduction. ~4 minutes total on this machine.
set -e
cd "$(dirname "$0")"
python build_cache.py      # quality metrics + 3 score matrices (45x45)
python analyze.py          # quality distributions, quality-vs-score correlation
python gate.py             # does a quality gate help? (with random-drop control)
python decompose.py        # variance decomposition; quality vs diversity
python strategy.py         # enrolment strategies over random splits
python final.py            # paired significance, bootstrap CIs, overlap stats
python headline.py         # the headline table
