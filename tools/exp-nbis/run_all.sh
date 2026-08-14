#!/bin/sh
# Reproduce the whole NBIS experiment from scratch.
set -e
cd "$(dirname "$0")"
R=${R:-/home/melb/projects/libfprint-elanpress/libfprint}

echo "== build standalone MINDTCT+BOZORTH3 from libfprint's bundled NBIS =="
gcc -O2 -o nbisbatch nbisbatch.c $R/nbis/mindtct/*.c $R/nbis/bozorth3/*.c \
    -I$R/nbis/include -I$R/nbis/libfprint-include -I$R \
    $(pkg-config --cflags --libs glib-2.0) -lm

echo "== generate preprocessed dataset variants =="
python3 prep.py 1 nn 0
python3 prep.py 2 nn 0
python3 prep.py 2 bilin 0
python3 prep.py 3 nn 0

echo "== CONTROL: does the evaluator reproduce the known LCN+rot+NCC baseline? =="
python3 control_ncc.py

echo "== full sweep: minutia counts + bozorth3 under protocols A and B =="
python3 sweep_all.py

echo "== diagnostics =="
python3 diag_oracle.py prep/s2nn/gabor_eq 8
python3 diag_oracle.py prep/s2nn/gabor_eq 12 rot
python3 diag_repro.py prep/s2bilin/gabor_eq 10
python3 diag_crop.py 2 gabor_eq
python3 diag_synth.py
