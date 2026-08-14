#!/bin/sh
# Full capture-quality investigation, from the raw dataset. ~4 minutes.
#   s0  sanity checks (fast NCC == naive NCC, self-match == 1.0, no leaks)
#   build_cache  quality metrics + 45x45 score/alignment matrices
#   s1  quality metrics, quality-vs-score, the gate, overlap, headline
#   s2  transitivity, template usefulness, verify-side gate, re-press ceiling
#   s3  positive control for the transitivity test + orientation-field registration
#   s4  montage PNGs (look at the data)
#   s5  ridge period, alias floor, orientation spread, minutia budget
#   s6  enrolment size and enrolment kind
set -e
cd "$(dirname "$0")"
python3 s0_sanity.py
python3 build_cache.py
python3 s1_measure.py     | tee out_s1.txt
python3 s2_deeper.py      | tee out_s2.txt
python3 s3_alignment.py   | tee out_s3.txt
python3 s4_render.py
python3 s5_information.py | tee out_s5.txt
python3 s6_enrolment.py   | tee out_s6.txt
