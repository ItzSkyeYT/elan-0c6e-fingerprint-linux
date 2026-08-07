#!/usr/bin/env python3
"""
Analyse a PGM capture from an ELAN 04f3:0c6e sensor.

Reports the statistics that actually distinguish a usable capture from a
degraded one, and optionally writes a contrast-stretched copy so you can look
at it. Prints numbers only by default -- please share numbers rather than
images, since fingerprint captures are biometric data.

Usage:
    ./analyze-pgm.py capture.pgm [--stretch out.pgm]

Reference figures from a GV301QH (firmware 0x0161, 150x52 sensor):

    good capture      span ~60, coherent directional ridges
    degraded capture  span ~35, speckled, ridges broken up

A low span means the sensor is returning a compressed slice of its 14-bit
range. On this device that has correlated with the "wedged after one capture
per power-up" state rather than with anything in the image pipeline -- power
cycle before concluding your processing code is at fault.
"""

import sys
import statistics


def read_pgm(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    if not data.startswith(b'P5'):
        raise ValueError('not a binary PGM (P5)')
    # header: P5 <ws> W H <ws> MAXVAL <single ws> then raw bytes
    fields, idx = [], 2
    while len(fields) < 3:
        while idx < len(data) and data[idx:idx + 1].isspace():
            idx += 1
        if data[idx:idx + 1] == b'#':                       # comment line
            while idx < len(data) and data[idx] != 0x0A:
                idx += 1
            continue
        start = idx
        while idx < len(data) and not data[idx:idx + 1].isspace():
            idx += 1
        fields.append(int(data[start:idx]))
    idx += 1                                                # single whitespace
    w, h, _maxval = fields
    return w, h, list(data[idx:idx + w * h])


def ridge_score(px, w, h):
    """Crude directional-coherence check.

    A real fingerprint has strong periodicity along the axis crossing the
    ridges and weak periodicity along the axis running with them. Noise is
    roughly isotropic, so the ratio separates the two.
    """
    rows = [statistics.pstdev(px[y * w:(y + 1) * w]) for y in range(h)]
    cols = [statistics.pstdev([px[y * w + x] for y in range(h)]) for x in range(w)]
    mr, mc = statistics.mean(rows), statistics.mean(cols)
    return mr, mc, (max(mr, mc) / min(mr, mc)) if min(mr, mc) else 0.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    path = sys.argv[1]
    w, h, px = read_pgm(path)
    lo, hi = min(px), max(px)
    mr, mc, aniso = ridge_score(px, w, h)

    print(f'file        {path}')
    print(f'dimensions  {w} x {h}  ({len(px)} px)')
    print(f'range       min={lo}  max={hi}  span={hi - lo}')
    print(f'mean        {statistics.mean(px):.1f}')
    print(f'stdev       {statistics.pstdev(px):.1f}')
    print(f'row/col sd  {mr:.1f} / {mc:.1f}   anisotropy={aniso:.2f}')
    print()
    if hi - lo < 45:
        print('  ! low dynamic range - suspect a wedged sensor, not bad processing.')
        print('    Power cycle the machine and re-capture before drawing conclusions.')
    if aniso < 1.15:
        print('  ! near-isotropic texture - little directional ridge flow.')
    else:
        print('  directional structure present, consistent with real ridges.')

    if '--stretch' in sys.argv:
        out = sys.argv[sys.argv.index('--stretch') + 1]
        span = hi - lo
        body = bytes(((v - lo) * 255 // span) if span else 0 for v in px)
        with open(out, 'wb') as fh:
            fh.write(b'P5\n%d %d\n255\n' % (w, h) + body)
        print(f'\nwrote contrast-stretched copy to {out}')
        print('(keep it local - it is biometric data)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
