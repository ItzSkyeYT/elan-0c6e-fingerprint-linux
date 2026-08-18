#!/usr/bin/env python3
"""
Verification for the ELAN 04f3:0c6e: match ONE probe swipe against an enrolled
finger map.

swipe_assemble.py turns one swipe into a strip.  swipe_map.py registers several
strips into one map.  This file is the third step and the one the driver
actually runs at unlock time: the user gives you a single swipe, you assemble it
into a strip, and you have to decide whether that strip belongs on the enrolled
map -- with no second attempt, under about a second, and without ever having
seen this particular swipe before.

Everything below is measured on the six real strips on this machine.  Read the
section marked WHAT THIS CANNOT SHOW before believing any of it, because the
single most important fact about this file is that there is exactly one finger
in the data, so nothing here is a false-accept rate.

-------------------------------------------------------------------------------
1. WHAT THE TEMPLATE IS, AND WHY THERE ARE TWO ANSWERS
-------------------------------------------------------------------------------
swipe_map.register_to_map found that registering a candidate against the BLENDED
canvas is worse than registering it against the individual enrolled strips, and
that finding is confirmed here.  Leave-one-out, over the four accepted strips
(each held out, the map rebuilt from the other three, the held-out strip used as
a probe):

    held out              canvas route      best gallery link
    20260814-165428#2     NCC 0.652         0.784
    20260814-165428#1     NCC 0.602         0.763
    20260814-170350       NCC 0.637         0.753
    20260814-165457       NCC 0.637         0.775

The canvas route loses about 0.13 of NCC, every time, on every strip.  The
reason is in swipe_map's own report: the canvas is a mosaic whose contributors
are still ~4.3 px out of register with each other, so correlating against it
correlates against a slightly smeared thing.  `--overlap-sweep` shows the same
effect from another angle: cropping the canvas RAISES the correlation, because
cropping removes contributors that disagree with each other.

The gallery column above is the best link BY NCC, and picking that way is a
mistake -- see `match_gallery`, which picks by `explained` instead after this
very measurement showed that maximising NCC across enrolled strips selects the
smallest overlap on offer.

The canvas route costs ONE search.  The gallery route costs one search per
enrolled strip, which for a realistic 6-8 swipe enrolment is 6-8 searches, and
the whole verify budget is about a second.  So:

    ROUTE_CANVAS   default.  One search.  Lower score, but the margin over the
                   measured null is still 1.3x and all four held-out strips
                   are accepted.
    ROUTE_GALLERY  higher score and a pose-agreement cross-check that the
                   canvas cannot give, at N times the cost.  Use it when the
                   canvas route lands near threshold, not instead of it.
    ROUTE_HYBRID   canvas search for the pose, then re-score against the single
                   enrolled strip with the most overlap, over a small local
                   window only.  Gets most of the gallery score for about 1.1
                   searches.  Measured in `--loo`.

-------------------------------------------------------------------------------
2. THE SCORE, AND WHY IT IS NOT NORMALISED THE WAY YOU WOULD EXPECT
-------------------------------------------------------------------------------
The obvious worry about partial overlap is statistical: a correlation measured
over fewer samples is noisier, so a non-matching probe that only grazes the map
should be able to reach a higher NCC by chance, and the threshold ought to rise
as the overlap falls -- classically as 1/sqrt(N).

That is measured here, and it is false over the range that matters.  Cropping
the map's coverage and re-registering a phase-randomised probe at each crop
level (`--overlap-sweep`), the null NCC over overlaps from 12.9 to 28.9 mm^2:

    overlap mm^2   12.9   13.5   16.1   20.4   26.1   28.9
    null mean      0.324  0.331  0.337  0.354  0.329  0.342

Log-log slope against overlap: -0.019.  Flat.  Not -0.5.  The reason is that
the null score is not one correlation over N samples, it is the MAXIMUM over
tens of thousands of shifts of a heavily autocorrelated field: the band-passed
map has an autocorrelation area of 54.7 px, so a 20 mm^2 overlap is only ~140
independent cells, and the extreme-value term dominates the sample-size term
across the whole range.  Cropping the map also shrinks the shift set, and the
two effects very nearly cancel.

So NCC is NOT rescaled by overlap here.  What partial overlap does instead is
worse and less obvious, and section 3 is about it.

Three numbers are reported for every match, because no one of them is enough:

    ncc          masked NCC at the winning pose.  The discriminator.
    psr          peak-to-sidelobe ratio.  Nearly useless on its own -- the
                 measured null reaches 13.06 against a worst genuine 12.3 --
                 and kept only because it fails independently.
    explained    ncc * (overlap / probe covered area).  The number that
                 responds to a map too SMALL, because it asks how much of the
                 PROBE the map accounts for, not how well the part that matched
                 matched.  See section 3.  It does NOT respond to a probe too
                 SHORT -- measured slope -0.002, flat -- so it is half a test,
                 not a whole one.  See section 3b, which was measured after
                 section 3 and corrects it.

-------------------------------------------------------------------------------
3. THE FAILURE MODE PARTIAL OVERLAP ACTUALLY CAUSES
-------------------------------------------------------------------------------
Cropping the enrolled map and re-registering the SAME genuine probe:

    map coverage   67.3   60.4   52.4   46.4   39.5   33.3   27.2 mm^2
    overlap        28.9   26.1   20.4   16.1   13.5   12.9   12.9 mm^2
    NCC            0.637  0.729  0.737  0.794  0.645  0.523  0.321
    pose error      0.00   3.18   7.99  16.71  26.58  37.77 105.09 px

The NCC RISES as the map is cropped, from 0.637 to 0.794, while the answer
becomes wrong by 16.7 px.  This is not noise and it is not a bug: a smaller map
gives the search fewer constraints, so it finds a placement that explains the
surviving patch better than the true placement explains the whole thing.  A
score that goes UP as the answer goes WRONG is the most dangerous shape a
biometric score can have, and NCC alone has it.

Two things catch it and both are kept:

  * `explained` = ncc * overlap/probe_area falls monotonically through the
    sweep (0.45, 0.42, 0.32, 0.27, 0.19, 0.14, 0.08) because the overlap
    fraction falls faster than the NCC rises.
  * the hard overlap gate.  MIN_TRUSTED_OVERLAP_MM2 = 16 mm^2 was set in
    swipe_map from a ground-truth sweep; this file's sweep agrees with it from
    the other direction, and raises it, because 16.1 mm^2 here was already
    16.7 px wrong.

-------------------------------------------------------------------------------
3b. BUT `explained` DOES NOT CATCH A SHORT SWIPE  (--probe-sweep)
-------------------------------------------------------------------------------
The claim above -- that `explained` is the one statistic whose shape is right
for partial overlap -- was measured by cropping the MAP, and it does not
survive the other experiment.  `--probe-sweep` truncates the PROBE instead,
re-assembles it from a contiguous run of the full probe's frames, and registers
it against a fixed 70.0 mm^2 map.  Ground truth is exact (the sub-strip's own
track reproduces the full track to 0.000 px, so the offset is a pure constant).

    keep-head  frac   1.00  0.85  0.70  0.55  0.45  0.35  0.25
    probe mm^2         44.7  41.8  36.5  32.2  30.3  26.6  24.9
    overlap mm^2       33.5  31.8  28.6  25.6  22.8  19.7  18.5
    overlap FRACTION   0.75  0.76  0.78  0.79  0.75  0.74  0.74
    NCC               0.637 0.556 0.622 0.612 0.723 0.771 0.754
    explained         0.476 0.423 0.488 0.486 0.544 0.569 0.561
    pose error px      0.00 11.88 12.61 18.86  3.87  1.69  1.41

`explained` does not fall.  Its slope against the truncation fraction is
-0.002 -- flat -- and it RISES to 0.569 while NCC rises to 0.771.  The reason
is structural: cropping the map shrinks the overlap while the probe area stays
fixed, so the ratio falls; shortening the probe shrinks BOTH, and a shorter
probe that still lands on the map keeps its overlap fraction at 0.74-0.79
throughout.  `explained` is a ratio, and truncation divides its numerator and
denominator by nearly the same factor.

So the two sweeps stress different tests and the honest summary is:

    map too small (enrolment)   -> caught by `explained`
    probe too short (verify)    -> caught ONLY by the hard mm^2 overlap floor

That matters because a short swipe is the routine case, not the exotic one:
the device wedges after 420-563 ms, so the user has no way to swipe longer and
every failure to swipe far enough lands here.  The floor is doing the work
alone, and it is the constant fitted to the fewest measurements.

The acceptance rule's behaviour over this sweep is also not monotone in
accuracy, and this is the sharpest evidence in the file that the rule is fitted
rather than principled: 11 of 14 truncations were ACCEPTED, including one wrong
by 18.86 px (0.96 mm, more than two ridge pitches), while the single MOST
accurate truncation in the whole sweep -- keep-tail 0.85, wrong by 0.71 px --
was REJECTED, by the split-half excess test.  Worst accepted genuine NCC 0.519
against a null max of 0.421 over the same sweep is a margin of 1.23x.

For VERIFICATION specifically there is a distinction worth stating plainly,
because it is easy to get backwards: a genuine probe accepted at a wrong pose
is still a correct ACCEPT.  Pose error is not a security failure at verify time.
It becomes one the moment the template is updated from the probe, and it is
evidence that the score is not measuring what it is assumed to measure -- which
is why the gate is enforced anyway.

-------------------------------------------------------------------------------
4. WHAT THIS CANNOT SHOW
-------------------------------------------------------------------------------
There is ONE finger in this dataset, from one person, in two sessions.  There is
therefore no impostor distribution, no d-prime, no EER, and no FAR.  Nothing in
this file should be quoted as one.

The nulls measured here -- phase-randomised, flipped and 180-rotated strips --
bound how high NON-CORRESPONDING CONTENT can score through this search.  They do
not bound how high a different person's finger can score, and a different
person's finger is a far harder negative than a phase-randomised strip, because
it has real ridges with real minutiae and the same broad ridge flow.  swipe_map
already measured the size of that effect from the other side: a strip turned end
for end still correlates at 0.474 with another strip of the same finger, purely
on broad ridge-flow similarity.

The press dataset is not a substitute.  `--press-check` runs the identical
matcher over press-to-press pairs, where image quality is matched on both sides,
and finds no usable finger discrimination at all:

    same finger      right-index         NCC 0.275 +- 0.128
                     right-middle        NCC 0.199 +- 0.084
                     right-index-cover   NCC 0.224 +- 0.101
    DIFFERENT finger index vs middle     NCC 0.210 +- 0.073
                     index-cover vs mid  NCC 0.181 +- 0.067

right-middle against ITSELF (0.199) scores lower than right-index against
right-middle (0.210).  The ordering is not even consistent, so this is not a
weak signal, it is no signal.  Two explanations are available and this data
cannot separate them: the presses are averaged captures from the old driver and
are too blurred to carry identity, and two presses of one finger need not image
the same part of that finger.  Either way, the 45 presses in swipe_map's null
distribution were bounding non-matching content, not impostors.

See `protocol()` for the exact collection that would fix this.

Usage:
    swipe_verify.py --enrol DIR [DIR ...] --template T.npz
    swipe_verify.py --template T.npz --verify DIR
    swipe_verify.py --loo              leave-one-out on the real captures
    swipe_verify.py --split-truth      ground-truth validation from one swipe
    swipe_verify.py --overlap-sweep    score and accuracy vs overlap (crops map)
    swipe_verify.py --probe-sweep      score and accuracy vs probe length
    swipe_verify.py --press-check      does this matcher see finger identity?
    swipe_verify.py --coverage-model   how many enrolment swipes
    swipe_verify.py --budget           what a coarser angle grid costs
    swipe_verify.py --protocol         the collection protocol to run next
    swipe_verify.py --selftest
    swipe_verify.py --all              everything above, in order
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from swipe_assemble import (                                    # noqa: E402
    FRAME_W, FRAME_H, MM_PER_PX,
    DX_MIN, DX_MAX, DY_ABS,
    ridge_dog, to_u8, write_pgm,
)
import swipe_map as SM                                          # noqa: E402
from swipe_map import (                                         # noqa: E402
    ROT_MAX, ROT_STEP,
    MIN_OVERLAP_MAP, MIN_TRUSTED_OVERLAP_MM2,
    ACCEPT_SPLIT_SUPPORT, ACCEPT_SPLIT_EXCESS,
    similarity, translation, transform_corners, apply_to_points,
    warp, erode, decompose,
    prep_band, warped_source, register_strip, judge, split_half_check,
    masked_ncc, _fft_pack, surface_peak, pose_distance,
    Strip, assemble_one, strips_from_capture, segment_capture,
    build_finger_map, link_graph, refine_poses, blend_map,
    phase_randomise, read_pgm, load_swipe, _mean_offset,
)

# ---------------------------------------------------------------------------
# tunables, with the measurement each one rests on
# ---------------------------------------------------------------------------

# Along-axis scale search at VERIFY time is wider than at enrolment.  swipe_map
# searches 0.94..1.06 because that is wide enough for one hop between two
# strips.  A probe is registered against a map whose poses have already COMPOSED
# several such hops: the four accepted strips ended at scale_x 1.000, 0.9703,
# 0.9415 and 0.9604 relative to the seed, so a probe resembling 20260814-170350
# needs 0.94 against the canvas -- exactly swipe_map's rail.  Measured: with the
# 0.94 rail in place, the held-out 170350 probe is REJECTED for sitting on it.
# Widening to 0.90..1.10 fixes that and, per `--overlap-sweep`, does not lift
# the nulls (they were not scale-limited).
VERIFY_SCALE_GRID = np.arange(0.90, 1.1001, 0.01)

# Verification acceptance threshold on NCC against the CANVAS.
#
# This is deliberately NOT swipe_map's ACCEPT_NCC of 0.55, because the two
# numbers are answering different questions on different inputs.  Enrolment
# compares two single-swipe strips; verification compares a strip against a
# blended canvas, which costs about 0.13 of NCC (section 1).  Measured on this
# machine, leave-one-out:
#
#     worst held-out genuine, canvas route   0.602
#     best null, canvas route, wide scale    0.467  (phase-randomised probe)
#
# 0.52 sits between them, closer to the null than to the genuine, because a
# false reject costs a retry and a false accept costs everything.  The margin
# it leaves is 1.29x on the genuine side and 1.11x on the null side, and 1.11x
# is thin.  It is thin because the null here is a bound on non-corresponding
# CONTENT and a real impostor finger would score higher; do not read the gap as
# comfort.  This threshold cannot be finalised without other fingers.
ACCEPT_NCC_CANVAS = 0.52

# The same threshold for the gallery route, where the probe meets an individual
# enrolled strip rather than the canvas.  Held-out genuine best-link measured
# 0.753-0.784; swipe_map's own null bound on strip-vs-strip was 0.474.
ACCEPT_NCC_GALLERY = 0.55

# PSR floor.  Kept low on purpose.  swipe_map measured a null PSR of 13.06
# against a worst genuine link of 13.75, and the canvas route here measures
# genuine PSR down to 12.3, BELOW that null.  So PSR cannot carry a decision on
# this data at all; it is retained only to catch a surface with no peak in it.
ACCEPT_PSR_VERIFY = 6.0

# Minimum overlap for a verification to be believed, in mm^2 of finger.
# Inherited unchanged from swipe_map, where it was set by a ground-truth sweep
# on strip-vs-strip, and independently corroborated here from the other side:
# over the 56 map-crop registrations in `--overlap-sweep`, the least overlap
# that survived the whole acceptance rule was 19.6 mm^2, and every registration
# that came back more than 15 px wrong had 26.1 mm^2 or less.  Nothing between
# 16.0 and 19.6 mm^2 was accepted, so raising the constant to 20 would change
# no decision on this data while pretending to a precision the data does not
# support.  It stays at 16.
MIN_VERIFY_OVERLAP_MM2 = MIN_TRUSTED_OVERLAP_MM2      # 16.0

# Minimum `explained` = ncc * (overlap / probe area).
#
# This is the test with the right SHAPE for partial overlap, and it is the only
# one of the four candidates that has it.  Over the 56 map-crop registrations,
# split by how far the recovered pose actually was from the truth:
#
#                  accurate (<=5 px, n=24)      badly wrong (>15 px, n=15)
#     ncc          0.682 +- 0.065, min 0.602    0.548 +- 0.102, MAX 0.794
#     psr          14.06 +- 1.30,  min 12.27    10.67 +- 3.61,  MAX 19.23
#     overlap mm2  28.8  +- 4.6,   min 15.95    15.21 +- 3.28,  MAX 26.05
#     explained    0.418 +- 0.054, min 0.249    0.175 +- 0.054, MAX 0.310
#
# Read the min/MAX columns, not the means.  NCC's worst wrong answer (0.794)
# beats EVERY accurate answer, so NCC cannot separate these at any threshold at
# all.  PSR and raw overlap overlap heavily.  `explained` leaves an ambiguous
# band of only 0.249-0.310, and 0.28 splits it.
#
# 0.28 is therefore fitted to 56 registrations of one finger, and the honest
# reading is "somewhere in 0.25-0.31, and this data cannot say where".  It
# rejects nothing that the rest of the rule accepts today.
MIN_EXPLAINED = 0.28

# Angle grid for verification.  Enrolment can afford 51 angles at 1 deg plus a
# 0.25 deg refinement; a verify has about a second.  `--budget` measures the
# cost of a coarser grid directly.
VERIFY_ROT_STEP = 1.0

# Reject when the split-half test cannot be run at all?
#
# swipe_map's `judge` does, and says so.  This file does NOT, by default, and
# the difference is deliberate rather than an oversight, so here is the measured
# consequence of each choice.  Over the 56 map-crop registrations:
#
#   REQUIRE_SPLIT_HALF = True   nothing accepted is worse than about 11 px, but
#                               several correct answers are thrown away for the
#                               unrelated reason that the strip was too small
#                               to cut in half.
#   REQUIRE_SPLIT_HALF = False  one accepted answer is 16.71 px (0.85 mm) wrong.
#                               That answer is a GENUINE probe placed badly.
#
# For ENROLMENT, True is right: a badly placed strip smears the map permanently.
# For VERIFICATION it is not, because a genuine probe accepted at a wrong pose
# is still a correct accept, and the only thing the rule buys is false
# rejections.  It flips back to True the moment the template is updated from
# verification probes, which is why it is a constant and not a hard-coded
# `if`.
REQUIRE_SPLIT_HALF = False

DEFAULT_TEMPLATE = Path.home() / ".local/share/elan-fp/template.npz"
DEFAULT_OUT = Path.home() / ".local/share/elan-fp/verify"

CAPTURES = [
    Path.home() / ".local/share/elan-fp/swipe/20260814-165428",
    Path.home() / ".local/share/elan-fp/swipe/20260814-165457",
    Path.home() / ".local/share/elan-fp/swipe/20260814-170350",
]
DATASET = Path.home() / ".local/share/elan-fp/dataset"


# ---------------------------------------------------------------------------
# 1. the enrolled template
# ---------------------------------------------------------------------------
class Template:
    """What enrolment stores and verification loads.

    Both representations are kept, because they are used for different things
    and the expensive one is only 4x the size of the cheap one:

      canvas / canvas_cov   the blended map.  What the default route
                            correlates against.  One search per verify.
      strips                the enrolled strips and their canvas poses.  What
                            the gallery route correlates against, one search
                            each, for a better score and a pose cross-check.

    `null` is the calibration measured at enrolment time against THIS map: a
    set of phase-randomised probes pushed through the identical search.  Doing
    it per template rather than shipping a global constant is the whole point --
    a small map, a blurred map and a well-registered map do not have the same
    null, and enrolment is allowed to be expensive.
    """

    def __init__(self, canvas, canvas_cov, strips=(), origin=(0, 0),
                 null=None, meta=None):
        self.canvas = np.asarray(canvas, np.float64)
        self.canvas_cov = np.asarray(canvas_cov, bool)
        self.strips = list(strips)          # Strip objects, .pose set
        self.origin = tuple(origin)
        self.null = null or {}
        self.meta = meta or {}

    @property
    def coverage_mm2(self):
        return float(self.canvas_cov.sum()) * MM_PER_PX ** 2

    def canvas_pose(self, strip):
        """Strip -> canvas-with-origin-removed."""
        y0, x0 = self.origin
        return translation(-y0, -x0) @ strip.pose

    # -- persistence --------------------------------------------------------
    def save(self, path):
        """One npz.  The strips are stored as their blended images and masks
        plus their poses; the frames are NOT stored, because verification never
        needs to re-blend the map and the frames are 15.6 kB each."""
        d = {"canvas": self.canvas.astype(np.float32),
             "canvas_cov": self.canvas_cov,
             "origin": np.asarray(self.origin, np.int32),
             "n_strips": np.asarray(len(self.strips), np.int32)}
        for i, s in enumerate(self.strips):
            d[f"s{i}_img"] = s.img.astype(np.float32)
            d[f"s{i}_cov"] = s.cov
            d[f"s{i}_pose"] = np.asarray(s.pose, np.float64)
            d[f"s{i}_gain"] = np.asarray(s.gain, np.float64)
        d["meta"] = np.frombuffer(
            json.dumps({"meta": self.meta, "null": self.null,
                        "names": [s.name for s in self.strips]}).encode(),
            dtype=np.uint8)
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        blob = json.loads(bytes(z["meta"]).decode())
        n = int(z["n_strips"])
        strips = []
        for i in range(n):
            img = z[f"s{i}_img"].astype(np.float64)
            cov = z[f"s{i}_cov"]
            s = Strip(blob["names"][i], np.zeros((0, FRAME_H, FRAME_W)),
                      np.zeros(0), np.zeros(0), img, cov, {})
            s.pose = z[f"s{i}_pose"]
            s.gain = tuple(z[f"s{i}_gain"])
            strips.append(s)
        return Template(z["canvas"].astype(np.float64), z["canvas_cov"],
                        strips, tuple(z["origin"]),
                        blob.get("null"), blob.get("meta"))


def enrol(capture_dirs, verbose=True, calibrate=True, n_null=12):
    """Assemble every swipe in every capture, build the map, calibrate the null."""
    strips = []
    for c in capture_dirs:
        strips += strips_from_capture(c, verbose=verbose)
    if not strips:
        raise RuntimeError("no usable swipe in any capture")
    return enrol_from_strips(strips, verbose=verbose, calibrate=calibrate,
                             n_null=n_null)


def enrol_from_strips(strips, verbose=True, calibrate=True, n_null=12):
    for s in strips:
        s.pose = None
    accepted, rejected, trials = build_finger_map(strips, verbose=verbose)
    links = link_graph(accepted, verbose=verbose)
    if links:
        refine_poses(accepted, links, verbose=verbose)
    res = blend_map(accepted, modes=("sharp_win",))
    img, cov = res["blends"]["sharp_win"]
    t = Template(img, cov, accepted, res["origin"], meta={
        "n_offered": len(strips),
        "n_accepted": len(accepted),
        "rejected": [s.name for s in rejected],
        "accepted": [s.name for s in accepted],
        "coverage_mm2": round(float(cov.sum()) * MM_PER_PX ** 2, 2),
        "canvas_px": [int(img.shape[1]), int(img.shape[0])],
    })
    if calibrate:
        t.null = calibrate_null(t, n=n_null, verbose=verbose)
    return t, rejected, trials


def calibrate_null(template, n=12, rng=None, verbose=True):
    """Measure what NON-CORRESPONDING content scores against THIS map.

    Phase randomisation is the null of choice, for the reason swipe_map gives:
    it preserves the magnitude spectrum exactly, so the null probe has the same
    ridge pitch, the same orientation energy and the same band fraction as a
    real strip, and only the correspondence is gone.  Flips and rotations change
    the local geometry as well, so a low score against them is partly the
    matcher noticing that the ridges stopped looking like ridges.

    This is a bound on non-corresponding CONTENT.  It is not an impostor
    distribution and no false-accept rate follows from it.
    """
    rng = rng or np.random.default_rng(20260814)
    src = template.strips or []
    if not src:
        return {"n": 0, "note": "no strips stored, cannot calibrate"}
    rows = []
    t0 = time.time()
    for k in range(n):
        s = src[k % len(src)]
        kind = ("phase", "flipud", "rot180")[k % 3] if k >= len(src) else "phase"
        if kind == "phase":
            bi, bc = phase_randomise(s.img, s.cov, rng), s.cov
        elif kind == "flipud":
            bi, bc = np.flipud(s.img), np.flipud(s.cov)
        else:
            bi, bc = s.img[::-1, ::-1], s.cov[::-1, ::-1]
        r = register_strip(template.canvas, template.canvas_cov, bi, bc,
                           scales=VERIFY_SCALE_GRID, split_half=False)
        if r is None:
            continue
        rows.append({"kind": kind, "ncc": r["ncc"], "psr": r["psr"],
                     "overlap_px": r["overlap_px"],
                     "explained": round(r["overlap_px"] / max(bc.sum(), 1)
                                        * r["ncc"], 4)})
    if not rows:
        return {"n": 0}
    ncc = np.array([r["ncc"] for r in rows])
    psr = np.array([r["psr"] for r in rows])
    exp = np.array([r["explained"] for r in rows])
    out = {
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "ncc_mean": round(float(ncc.mean()), 4),
        "ncc_sd": round(float(ncc.std(ddof=1)) if len(ncc) > 1 else 0.0, 4),
        "ncc_max": round(float(ncc.max()), 4),
        "psr_max": round(float(psr.max()), 2),
        "explained_max": round(float(exp.max()), 4),
        "rows": rows,
        "note": ("bound on NON-CORRESPONDING CONTENT, measured against this "
                 "map. NOT an impostor distribution and NOT a false-accept "
                 "rate: every capture on this machine is one finger."),
    }
    if verbose:
        print(f"null calibration: {out['n']} non-matching registrations, "
              f"NCC mean {out['ncc_mean']:.3f} sd {out['ncc_sd']:.4f} "
              f"max {out['ncc_max']:.3f}, PSR max {out['psr_max']:.1f} "
              f"({out['seconds']:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# 2. matching a probe strip against the template
# ---------------------------------------------------------------------------
ROUTE_CANVAS = "canvas"
ROUTE_GALLERY = "gallery"
ROUTE_HYBRID = "hybrid"


def _angles(step=VERIFY_ROT_STEP, lim=ROT_MAX):
    return np.arange(-lim, lim + 1e-9, step)


def _explained(res, probe_cov):
    """ncc weighted by the fraction of the probe the map accounts for.

    The only one of the three reported numbers whose shape is right for partial
    overlap: through the map-crop sweep NCC rises while this falls, because the
    overlap fraction falls faster than the correlation rises.
    """
    a = float(probe_cov.sum())
    f = res["overlap_px"] / a if a > 0 else 0.0
    return round(f * res["ncc"], 4), round(f, 4)


def match_canvas(template, probe, angles=None, scales=VERIFY_SCALE_GRID,
                 refine=True):
    """One masked-NCC search of the probe against the blended map."""
    r = register_strip(template.canvas, template.canvas_cov,
                       probe.img, probe.cov,
                       angles=_angles() if angles is None else angles,
                       scales=scales, refine=refine, split_half=True)
    if r is None:
        return None
    exp, frac = _explained(r, probe.cov)
    r = dict(r)
    r.update({"route": ROUTE_CANVAS, "explained": exp, "overlap_frac": frac,
              "pose_canvas": r["pose"]})
    return r


def match_gallery(template, probe, angles=None, scales=VERIFY_SCALE_GRID,
                  refine=True):
    """One search per enrolled strip, plus the cross-check the canvas cannot give.

    When two or more enrolled strips each overlap the probe enough to place it,
    they each imply a canvas pose, and those poses have to agree.  That is cycle
    consistency, it is the standard validation for a mosaic, and it is free
    here.  Measured leave-one-out, the agreeing links land 3.98-13.33 px apart,
    which is the residual pose inconsistency of the map itself (swipe_map
    measured 4.29 px max after pose averaging), not a property of the probe.
    """
    out = []
    for s in template.strips:
        r = register_strip(s.img, s.cov, probe.img, probe.cov,
                           angles=_angles() if angles is None else angles,
                           scales=scales, refine=refine, split_half=True)
        if r is None:
            continue
        exp, frac = _explained(r, probe.cov)
        ok, why = judge(r, {"ncc": ACCEPT_NCC_GALLERY,
                            "psr": ACCEPT_PSR_VERIFY,
                            "overlap_mm2": MIN_VERIFY_OVERLAP_MM2})
        out.append(dict(r, route=ROUTE_GALLERY, via=s.name,
                        explained=exp, overlap_frac=frac,
                        link_ok=bool(ok), link_why=why,
                        pose_canvas=template.canvas_pose(s) @ r["pose"]))
    if not out:
        return None
    # Pick the reported link by `explained`, NOT by NCC.  Picking by NCC is the
    # obvious thing and it is wrong for exactly the reason section 3 gives, and
    # this route made the mistake visible on real data before it was fixed:
    # held out 20260814-165428#2, the highest-NCC link was 0.784 over only
    # 19.7 mm^2 of overlap (explained 0.299), while a link at NCC 0.741 covered
    # 26.3 mm^2 (explained 0.393).  Maximising NCC across several enrolled
    # strips systematically selects the SMALLEST overlap available, because that
    # is where the correlation is least constrained.
    best = max(out, key=lambda c: c["explained"])
    good = [c for c in out if c["link_ok"]]
    agree, disagree = [], []
    if len(good) > 1:
        anchor = max(good, key=lambda c: c["explained"])
        for c in good:
            if c is anchor:
                continue
            d = pose_distance(anchor["pose_canvas"], c["pose_canvas"],
                              probe.img.shape)
            (agree if d <= SM.AGREE_PX else disagree).append(
                (c["via"], round(d, 2)))
    best = dict(best)
    best.update({"links": [{"via": c["via"], "ncc": c["ncc"], "psr": c["psr"],
                            "overlap_mm2": c["overlap_mm2"],
                            "explained": c["explained"], "ok": c["link_ok"]}
                           for c in out],
                 "n_links_ok": len(good), "n_links": len(out),
                 "agree": agree, "disagree": disagree})
    return best


def match_hybrid(template, probe, scales=VERIFY_SCALE_GRID):
    """Canvas search for the pose, then re-score against the best single strip.

    The pose comes from the canvas, which costs one full search.  The SCORE then
    comes from the enrolled strip that overlaps the probe most under that pose,
    searched only over a small local window around the implied shift, which
    costs a handful of extra correlations rather than a whole angle sweep.

    The point is that the canvas route's 0.13 NCC deficit (section 1) is a
    property of what the probe is being compared against, not of the pose it
    found -- so it can be paid back for almost nothing.
    """
    base = match_canvas(template, probe, scales=scales)
    if base is None or not template.strips:
        return base
    # which enrolled strip does the probe land on most?
    ch, cw = template.canvas.shape
    wi, val, wm = warp(probe.img, base["pose_canvas"], (ch, cw), border=1,
                       extra=(probe.cov.astype(float),))
    pm = val & (wm > 0.99)
    best_s, best_n = None, 0
    for s in template.strips:
        sm_, sv = warp(s.cov.astype(float), template.canvas_pose(s),
                       (ch, cw), border=1)
        m = pm & sv & (sm_ > 0.99)
        if m.sum() > best_n:
            best_s, best_n = s, int(m.sum())
    if best_s is None or best_n < MIN_OVERLAP_MAP:
        base["route"] = ROUTE_HYBRID
        base["hybrid_note"] = "no enrolled strip overlapped the canvas pose enough"
        return base
    # Local re-score.  The pose RELATIVE to the chosen strip is what the local
    # search must be centred on, and it is not the same as the pose relative to
    # the canvas: the strip sits on the canvas at its own angle and scale, so
    # composing the inverse is the only way to get the right starting angle.
    rel = np.linalg.inv(template.canvas_pose(best_s)) @ base["pose_canvas"]
    th0, _sy0, sx0 = decompose(rel)
    r = register_strip(best_s.img, best_s.cov, probe.img, probe.cov,
                       angles=np.arange(th0 - 1.0, th0 + 1.001, 0.5),
                       scales=np.arange(max(0.90, sx0 - 0.02),
                                        min(1.10, sx0 + 0.02) + 1e-9, 0.01),
                       refine=False, split_half=True)
    if r is None:
        base["route"] = ROUTE_HYBRID
        return base
    exp, frac = _explained(r, probe.cov)
    out = dict(base)
    out.update({"route": ROUTE_HYBRID, "rescored_via": best_s.name,
                "canvas_ncc": base["ncc"],
                "ncc": r["ncc"], "psr": r["psr"],
                "overlap_px": r["overlap_px"],
                "overlap_mm2": r["overlap_mm2"],
                "explained": exp, "overlap_frac": frac,
                "split_support": r.get("split_support"),
                "split_excess": r.get("split_excess"),
                "split_px": r.get("split_px"),
                "pose_canvas": template.canvas_pose(best_s) @ r["pose"]})
    return out


def decide(res, template=None, route=ROUTE_CANVAS):
    """Accept or reject a verification, with every failing condition named.

    Deliberately several independent conditions rather than one fused score.
    A fused score would be tidier and would hide exactly the thing section 3 is
    about: NCC and overlap can move in opposite directions, and a weighted sum
    of them is a number that no longer tells you which happened.
    """
    if res is None:
        return False, "no shift anywhere had enough overlap with the map", {}
    ncc_t = (ACCEPT_NCC_GALLERY if route == ROUTE_GALLERY
             else ACCEPT_NCC_CANVAS)
    why = []
    if res["overlap_mm2"] < MIN_VERIFY_OVERLAP_MM2:
        why.append(f"overlap {res['overlap_mm2']:.1f} mm2 < "
                   f"{MIN_VERIFY_OVERLAP_MM2:.0f} mm2 (below this the map-crop "
                   f"sweep measures 8-105 px of pose error at NCC up to 0.79)")
    if res["explained"] < MIN_EXPLAINED:
        why.append(f"explained {res['explained']:.3f} < {MIN_EXPLAINED} "
                   f"(the map accounts for only {res['overlap_frac']:.0%} "
                   f"of the probe)")
    if res["ncc"] < ncc_t:
        why.append(f"NCC {res['ncc']:.3f} < {ncc_t}")
    if res["psr"] < ACCEPT_PSR_VERIFY:
        why.append(f"PSR {res['psr']:.1f} < {ACCEPT_PSR_VERIFY}")
    if res.get("at_rail"):
        why.append(f"best angle {res['theta']:.2f} deg is on the search rail")
    if res.get("scale_at_rail"):
        why.append(f"best along-axis scale {res['sx']:.3f} is on the rail")
    sup = res.get("split_support")
    if sup is None:
        if REQUIRE_SPLIT_HALF:
            why.append("the split-half test could not be run on this overlap")
    else:
        if sup < ACCEPT_SPLIT_SUPPORT:
            why.append(f"half-strip support {sup:.2f} < {ACCEPT_SPLIT_SUPPORT}")
        if res.get("split_excess", 0.0) > ACCEPT_SPLIT_EXCESS:
            why.append(f"a half of the probe prefers a shift "
                       f"{res['split_px']:.0f} px away by "
                       f"{res['split_excess']:.3f} NCC")
    # margin over the template's own measured null, when it exists
    extra = {}
    if template is not None and template.null.get("n"):
        nl = template.null
        extra["null_ncc_max"] = nl["ncc_max"]
        extra["margin_over_null_max"] = round(res["ncc"] - nl["ncc_max"], 4)
        if nl.get("ncc_sd"):
            extra["z_over_null"] = round(
                (res["ncc"] - nl["ncc_mean"]) / max(nl["ncc_sd"], 1e-6), 2)
        if res["ncc"] <= nl["ncc_max"]:
            why.append(f"NCC {res['ncc']:.3f} does not exceed this template's "
                       f"measured null maximum {nl['ncc_max']:.3f}")
    if why:
        return False, "; ".join(why), extra
    return True, (f"NCC {res['ncc']:.3f}, explained {res['explained']:.3f}, "
                  f"overlap {res['overlap_mm2']:.1f} mm^2, PSR "
                  f"{res['psr']:.1f}"), extra


def verify(template, probe, route=ROUTE_CANVAS, verbose=True):
    t0 = time.time()
    if route == ROUTE_GALLERY:
        res = match_gallery(template, probe)
    elif route == ROUTE_HYBRID:
        res = match_hybrid(template, probe)
    else:
        res = match_canvas(template, probe)
    ok, why, extra = decide(res, template, route)
    dt = time.time() - t0
    out = {"route": route, "accepted": bool(ok), "reason": why,
           "seconds": round(dt, 2)}
    if res is not None:
        out.update({k: res[k] for k in
                    ("ncc", "psr", "explained", "overlap_frac", "overlap_mm2",
                     "overlap_px", "theta", "sx", "split_support",
                     "split_excess", "split_px")
                    if k in res})
        for k in ("via", "rescored_via", "canvas_ncc", "n_links_ok", "n_links",
                  "agree", "disagree", "links"):
            if k in res:
                out[k] = res[k]
    out.update(extra)
    if verbose:
        print(f"  {route:<8} {'ACCEPT' if ok else 'REJECT'}  "
              f"NCC {out.get('ncc', float('nan')):.3f}  "
              f"expl {out.get('explained', float('nan')):.3f}  "
              f"PSR {out.get('psr', float('nan')):.1f}  "
              f"ov {out.get('overlap_mm2', float('nan')):.1f} mm2  "
              f"({dt:.1f}s)")
        if not ok:
            print(f"           {why}")
    return out, res


def probe_from_capture(path, verbose=False):
    """A verification swipe is ONE swipe.  If the capture holds several, the
    longest is used and the rest are reported, because a real verify gets one."""
    ss = strips_from_capture(path, verbose=verbose)
    if not ss:
        return None, []
    best = max(ss, key=lambda s: s.rank())
    return best, [s.name for s in ss]


# ---------------------------------------------------------------------------
# 3. leave-one-out: the only honest test of verification on this data
# ---------------------------------------------------------------------------
def _build_map(strips, verbose=False):
    for s in strips:
        s.pose = None
    acc, rej, tr = build_finger_map(strips, verbose=verbose)
    links = link_graph(acc, verbose=verbose)
    if links:
        refine_poses(acc, links, verbose=verbose)
    res = blend_map(acc, modes=("sharp_win",))
    img, cov = res["blends"]["sharp_win"]
    t = Template(img, cov, acc, res["origin"], meta={
        "accepted": [s.name for s in acc],
        "coverage_mm2": round(float(cov.sum()) * MM_PER_PX ** 2, 2)})
    return t, rej


def load_real_strips(verbose=False):
    out = []
    for c in CAPTURES:
        if not Path(c).exists():
            continue
        out += strips_from_capture(c, verbose=verbose)
    return out


def _fake_strip(name, img, cov):
    """A Strip carrying only an image and a mask.

    Nulls, presses and loaded templates all need something the matcher will
    accept as a probe or as an enrolled strip, without any frames behind it.
    """
    return Strip(name, np.zeros((0, FRAME_H, FRAME_W)),
                 np.zeros(0), np.zeros(0), img, cov, {})


def leave_one_out(routes=(ROUTE_CANVAS, ROUTE_GALLERY, ROUTE_HYBRID),
                  calibrate=True, verbose=True):
    """Hold each accepted strip out, rebuild the map, verify it against the rest.

    This is the ONLY configuration on this machine in which the probe is a
    swipe the template has never seen, which is the situation a real verify is
    in.  Registering an enrolled strip against the map it helped build is not a
    test of anything: measured, the two strips that dominate the blend score
    0.837 and 0.847 that way, against 0.602-0.652 when held out, and the
    difference is just those strips correlating with their own pixels.

    Four probes.  That is the entire genuine sample available, and it is far
    too small to put an error bar on.
    """
    strips = load_real_strips()
    if len(strips) < 3:
        return {"error": "not enough real captures on this machine"}
    by = {s.name: s for s in strips}
    t_all, rej_all = _build_map(list(strips))
    enrolled = [s.name for s in t_all.strips]
    if verbose:
        print(f"full map: {len(enrolled)} strips accepted "
              f"({t_all.coverage_mm2:.1f} mm^2), "
              f"{len(rej_all)} rejected at enrolment "
              f"({[s.name for s in rej_all]})")
    out = {"enrolled": enrolled,
           "rejected_at_enrolment": [s.name for s in rej_all],
           "trials": []}

    for held in enrolled:
        names = [n for n in enrolled if n != held]
        t0 = time.time()
        tmpl, _ = _build_map([by[n] for n in names])
        if calibrate:
            tmpl.null = calibrate_null(tmpl, n=9, verbose=False)
        probe = by[held]
        if verbose:
            print(f"\nheld out {held}: map from {len(tmpl.strips)} strips, "
                  f"{tmpl.coverage_mm2:.1f} mm^2, null NCC max "
                  f"{tmpl.null.get('ncc_max')}  "
                  f"[{time.time() - t0:.0f}s to rebuild]")
        rec = {"held": held, "map_mm2": round(tmpl.coverage_mm2, 1),
               "null": {k: tmpl.null.get(k) for k in
                        ("n", "ncc_mean", "ncc_sd", "ncc_max", "psr_max",
                         "explained_max")},
               "routes": {}}
        for route in routes:
            o, _r = verify(tmpl, probe, route=route, verbose=verbose)
            rec["routes"][route] = o
        rng = np.random.default_rng(abs(hash(held)) % (2 ** 31))
        neg = []
        for kind in ("phase", "flipud", "rot180"):
            if kind == "phase":
                bi, bc = phase_randomise(probe.img, probe.cov, rng), probe.cov
            elif kind == "flipud":
                bi, bc = np.flipud(probe.img), np.flipud(probe.cov)
            else:
                bi, bc = probe.img[::-1, ::-1], probe.cov[::-1, ::-1]
            o, _ = verify(tmpl, _fake_strip(f"{held}:{kind}", bi, bc),
                          route=ROUTE_CANVAS, verbose=False)
            neg.append({"kind": kind, "accepted": o["accepted"],
                        "ncc": o.get("ncc"), "explained": o.get("explained"),
                        "psr": o.get("psr")})
            if verbose:
                print(f"    null {kind:<7} NCC {o.get('ncc', 0):.3f} "
                      f"expl {o.get('explained', 0):.3f} "
                      f"{'ACCEPTED!!' if o['accepted'] else 'rejected'}")
        rec["negative_controls"] = neg
        out["trials"].append(rec)

    for route in routes:
        vals = [t["routes"][route] for t in out["trials"] if route in t["routes"]]
        acc = [v for v in vals if v["accepted"]]
        ncc = [v["ncc"] for v in vals if "ncc" in v]
        exp = [v["explained"] for v in vals if "explained" in v]
        sec = [v["seconds"] for v in vals]
        out.setdefault("summary", {})[route] = {
            "n_probes": len(vals),
            "n_accepted": len(acc),
            "ncc_min": round(min(ncc), 4) if ncc else None,
            "ncc_max": round(max(ncc), 4) if ncc else None,
            "explained_min": round(min(exp), 4) if exp else None,
            "seconds_median": round(float(np.median(sec)), 2) if sec else None,
        }
    nulls = [n for t in out["trials"] for n in t["negative_controls"]]
    out["summary"]["nulls"] = {
        "n": len(nulls),
        "n_accepted": sum(1 for n in nulls if n["accepted"]),
        "ncc_max": round(max(n["ncc"] for n in nulls), 4) if nulls else None,
        "explained_max": (round(max(n["explained"] for n in nulls), 4)
                          if nulls else None),
    }
    if verbose:
        print("\n" + "=" * 74)
        print("leave-one-out summary")
        for route in routes:
            s = out["summary"][route]
            print(f"  {route:<8} {s['n_accepted']}/{s['n_probes']} accepted, "
                  f"NCC {s['ncc_min']}-{s['ncc_max']}, "
                  f"explained >= {s['explained_min']}, "
                  f"{s['seconds_median']} s median")
        s = out["summary"]["nulls"]
        print(f"  nulls    {s['n_accepted']}/{s['n']} accepted, "
              f"NCC max {s['ncc_max']}, explained max {s['explained_max']}")
        print("  NOTE: 4 genuine probes, one finger.  This is a functional "
              "check, not a\n        false-reject rate, and the nulls are not "
              "impostors.")
    return out


# ---------------------------------------------------------------------------
# 4. ground truth built from ONE swipe
# ---------------------------------------------------------------------------
def split_truth(capture=None, verbose=True):
    """Verify a probe whose correct pose on the map is known exactly.

    One swipe is split into two DISJOINT runs of frames, A and B.  A goes into
    the enrolled map; B is the probe.  The true relation between them is known
    before either is assembled, because both are built from frames whose
    positions along the finger were measured once, in the full swipe -- this is
    swipe_map.synthetic_split_check's construction, lifted from strip-vs-strip
    to strip-vs-MAP.

    A and B share no frames at all, so every pixel in their overlap comes from a
    different exposure with different noise, different pressure and different
    blur.  What the construction does NOT reproduce is a finger that lifted and
    landed again: rotation is zero by construction and the skin never changed
    state, so this validates the translation search, the pose arithmetic and the
    scoring, not the rotation search.

    Four tiers go through the identical matcher, and the labels matter:

      B vs map-containing-A     genuine, ground-truth pose known
      B vs map-without-A        genuine as well -- the same finger from a
                                different acquisition.  NOT an impostor.  It is
                                here to size acquisition variability, which is
                                the largest term in the system.
      nulls                     non-corresponding content
      right-middle presses      a different FINGER, but see `press_check`: the
                                press captures carry no measurable identity
                                through this matcher, so a rejection here is
                                not evidence that impostors are rejected.
    """
    capture = Path(capture or CAPTURES[2])
    dxr = list(range(DX_MIN, DX_MAX + 1))
    dyr = list(range(-DY_ABS, DY_ABS + 1))
    frames, _ = load_swipe(capture)
    segs = segment_capture(frames, dxr, dyr)
    if not segs:
        return {"error": "no usable swipe segment"}
    a0, b0 = max(segs, key=lambda s: s[1] - s[0])
    full = assemble_one(frames[a0:b0], "full", verbose=False)
    kept = full.frames
    nk = len(kept)
    mid = nk // 2
    A = assemble_one(kept[:mid], "halfA", verbose=False)
    B = assemble_one(kept[mid:], "halfB", verbose=False)
    if A is None or B is None:
        return {"error": "a half did not assemble"}
    ka = A.report["contact_window"][0]
    kb = mid + B.report["contact_window"][0]
    ay, ax, asy, asx, _ = _mean_offset(A, full, ka)
    byy, bx, bsy, bsx, _ = _mean_offset(B, full, kb)
    true_ty, true_tx = ay - byy, ax - bx
    gt_err = math.hypot(asy + bsy, asx + bsx)
    if verbose:
        print(f"split of {capture.name}: {nk} in-contact frames, split at {mid}")
        print(f"  A {len(A.frames)} frames {A.report['coverage_mm2']:.1f} mm^2,"
              f"  B {len(B.frames)} frames {B.report['coverage_mm2']:.1f} mm^2")
        print(f"  ground truth B->A translation ({true_ty:.2f}, {true_tx:.2f}),"
              f" rotation 0 by construction, exact to {gt_err:.3f} px")

    others = [s for s in load_real_strips()
              if not s.name.startswith(capture.name)]
    out = {"capture": str(capture), "n_frames": int(nk), "split_at": int(mid),
           "ground_truth": {"ty": round(true_ty, 3), "tx": round(true_tx, 3),
                            "theta_deg": 0.0,
                            "uncertainty_px": round(gt_err, 3)},
           "tiers": {}}

    # tier 1 ---------------------------------------------------------------
    t_same, _ = _build_map([A] + others)
    t_same.null = calibrate_null(t_same, n=9, verbose=False)
    in_map = A.name in [s.name for s in t_same.strips]
    Ptrue = (t_same.canvas_pose(A) @ translation(true_ty, true_tx)
             if in_map else None)
    o, r = verify(t_same, B, route=ROUTE_CANVAS, verbose=False)
    err = (pose_distance(r["pose_canvas"], Ptrue, B.img.shape)
           if (r is not None and Ptrue is not None) else None)
    o["pose_error_px"] = None if err is None else round(err, 2)
    o["pose_error_mm"] = None if err is None else round(err * MM_PER_PX, 4)
    o["half_A_in_map"] = bool(in_map)
    out["tiers"]["B_vs_map_containing_A"] = o
    if verbose:
        print(f"\n  tier 1  B vs map containing A ({t_same.coverage_mm2:.1f} "
              f"mm^2, null NCC max {t_same.null.get('ncc_max')})")
        print(f"          NCC {o.get('ncc'):.3f} expl {o.get('explained'):.3f} "
              f"ov {o.get('overlap_mm2'):.1f} mm^2  pose error "
              f"{o['pose_error_px']} px ({o['pose_error_mm']} mm)  "
              f"{'ACCEPT' if o['accepted'] else 'REJECT: ' + o['reason']}")

    # tier 2 ---------------------------------------------------------------
    t_other, _ = _build_map(list(others))
    t_other.null = calibrate_null(t_other, n=9, verbose=False)
    tier2 = {}
    for nm, P in (("half_B", B), ("half_A", A), ("full_swipe", full)):
        oo, _ = verify(t_other, P, route=ROUTE_CANVAS, verbose=False)
        tier2[nm] = oo
        if verbose:
            print(f"  tier 2  {nm:<11} vs map WITHOUT this swipe: "
                  f"NCC {oo.get('ncc'):.3f} expl {oo.get('explained'):.3f} "
                  f"ov {oo.get('overlap_mm2'):.1f}  "
                  f"{'ACCEPT' if oo['accepted'] else 'reject'}")
    out["tiers"]["vs_map_without_this_swipe"] = tier2

    # tier 3 ---------------------------------------------------------------
    rng = np.random.default_rng(20260814)
    nulls = []
    for kind in ("phase", "phase", "phase", "flipud", "rot180"):
        if kind == "phase":
            bi, bc = phase_randomise(B.img, B.cov, rng), B.cov
        elif kind == "flipud":
            bi, bc = np.flipud(B.img), np.flipud(B.cov)
        else:
            bi, bc = B.img[::-1, ::-1], B.cov[::-1, ::-1]
        oo, _ = verify(t_same, _fake_strip(f"B:{kind}", bi, bc),
                       route=ROUTE_CANVAS, verbose=False)
        nulls.append({"kind": kind, "ncc": oo.get("ncc"),
                      "explained": oo.get("explained"), "psr": oo.get("psr"),
                      "accepted": oo["accepted"]})
        if verbose:
            print(f"  tier 3  null {kind:<7} NCC {oo.get('ncc'):.3f} "
                  f"expl {oo.get('explained'):.3f}  "
                  f"{'ACCEPTED!!' if oo['accepted'] else 'rejected'}")
    out["tiers"]["nulls_vs_map_containing_A"] = nulls

    # tier 4 ---------------------------------------------------------------
    press = []
    d = DATASET / "right-middle"
    if d.exists():
        for p in sorted(d.glob("*.pgm")):
            im = read_pgm(p)
            oo, _ = verify(t_same, _fake_strip(p.name, im,
                                               np.ones(im.shape, bool)),
                           route=ROUTE_CANVAS, verbose=False)
            press.append({"file": p.name, "ncc": oo.get("ncc"),
                          "explained": oo.get("explained"),
                          "accepted": oo["accepted"]})
    out["tiers"]["press_right_middle"] = press
    if verbose and press:
        v = np.array([p["ncc"] for p in press])
        e = np.array([p["explained"] for p in press])
        print(f"  tier 4  right-middle presses n={len(press)}: NCC mean "
              f"{v.mean():.3f} max {v.max():.3f}, explained max {e.max():.3f}, "
              f"{sum(1 for p in press if p['accepted'])} accepted")
        print("          (a different finger, but --press-check shows the "
              "press captures\n           carry no measurable identity through "
              "this matcher, so this is\n           weak evidence at best)")

    t1 = out["tiers"]["B_vs_map_containing_A"]
    t2 = max((v.get("ncc", 0) for v in tier2.values()), default=0)
    t3 = max((n["ncc"] for n in nulls), default=0)
    t4 = max((p["ncc"] for p in press), default=0)
    t2e = max((v.get("explained", 0) for v in tier2.values()), default=0)
    t3e = max((n["explained"] for n in nulls), default=0)
    t4e = max((p["explained"] for p in press), default=0)
    out["summary"] = {
        "tier1_same_swipe_ncc": t1.get("ncc"),
        "tier1_same_swipe_explained": t1.get("explained"),
        "tier1_pose_error_px": t1.get("pose_error_px"),
        "tier1_pose_error_mm": t1.get("pose_error_mm"),
        "tier2_other_acquisition_ncc_max": round(t2, 4),
        "tier3_null_ncc_max": round(t3, 4),
        "tier4_press_ncc_max": round(t4, 4),
        "tier4_press_explained_max": round(t4e, 4),
        # The claim worth making, and the only one the data supports: both
        # genuine tiers are above both non-genuine tiers.  Note what is NOT
        # claimed -- tier 1 above tier 2.  Measured, it is not: putting half of
        # the probe's OWN swipe into the map did not raise the probe's score
        # (0.650 against 0.681 without it), which is more evidence that blending
        # a strip into the canvas dilutes it rather than strengthens it.
        "both_genuine_tiers_above_both_non_genuine": bool(
            min(t1.get("ncc", 0), t2) > max(t3, t4)),
        "adding_own_swipe_half_to_map_raised_score": bool(t1.get("ncc", 0) > t2),
        # The margin that actually matters for security, such as it is.
        "worst_genuine_minus_best_non_genuine": round(
            min(t1.get("ncc", 0), t2) - max(t3, t4), 4),
    }
    if verbose:
        s = out["summary"]
        print(f"\n  genuine:     same-swipe {s['tier1_same_swipe_ncc']:.3f}, "
              f"other-acquisition up to "
              f"{s['tier2_other_acquisition_ncc_max']:.3f}")
        print(f"  non-genuine: null up to {s['tier3_null_ncc_max']:.3f}, "
              f"different-finger press up to {s['tier4_press_ncc_max']:.3f} "
              f"(explained {s['tier4_press_explained_max']:.3f})")
        print(f"  separated: "
              f"{'YES' if s['both_genuine_tiers_above_both_non_genuine'] else 'NO'}"
              f", by {s['worst_genuine_minus_best_non_genuine']:.3f} of NCC")
        if not s["adding_own_swipe_half_to_map_raised_score"]:
            print("  note: putting half of the probe's OWN swipe into the map "
                  "did NOT raise the\n        probe's score.  Blending a strip "
                  "into the canvas dilutes it.")
        print("  tier 2 is the SAME FINGER from a different acquisition -- a "
              "genuine\n  comparison, not an impostor one.")
    return out


# ---------------------------------------------------------------------------
# 5. score vs overlap: how much map does a verify need?
# ---------------------------------------------------------------------------
def overlap_sweep(fracs=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4),
                  sides=("keep-left", "keep-right"), n_null=4, verbose=True):
    """Crop the enrolled map and re-register the SAME probe.

    Cropping the map, rather than shortening the probe, is what isolates the
    variable: the probe is byte-identical at every point of the sweep, so any
    change in the score is caused by the overlap and by nothing else.  It is
    also the question enrolment actually poses -- an enrolment with too few
    swipes is a map that covers only part of where the probe lands.

    Ground truth is the full-map answer.  That is not an external truth, so the
    error column measures DEPARTURE FROM THE FULL-MAP ANSWER rather than
    absolute accuracy; `split_truth` supplies the absolute check separately and
    agrees with it.

    The result this exists to report: NCC RISES as the map is cropped, from
    0.637 to 0.794 on one probe, while the answer becomes 16.7 px wrong.  A
    smaller map constrains the search less, so it finds a placement that
    explains the surviving patch better than the true placement explains the
    whole thing.  A biometric score that goes up as the answer goes wrong is
    the worst shape a score can have, and raw NCC has it.
    """
    strips = load_real_strips()
    by = {s.name: s for s in strips}
    t_all, _ = _build_map(list(strips))
    enrolled = [s.name for s in t_all.strips]
    rows = []
    rng = np.random.default_rng(20260814)
    for held in enrolled:
        tmpl, _ = _build_map([by[n] for n in enrolled if n != held])
        probe = by[held]
        full = match_canvas(tmpl, probe)
        if full is None:
            continue
        Ptrue = full["pose_canvas"]
        cols = np.where(tmpl.canvas_cov.any(0))[0]
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        W = c1 - c0
        if verbose:
            print(f"\nheld out {held}: map {tmpl.coverage_mm2:.1f} mm^2, "
                  f"full-map NCC {full['ncc']:.3f}")
            print(f"  {'side':<11}{'frac':>5}{'map':>7}{'ovlap':>7}{'ovfrac':>7}"
                  f"{'NCC':>7}{'expl':>7}{'PSR':>6}{'err_px':>8}  verdict")
        for side in sides:
            for frac in fracs:
                w = int(round(W * frac))
                cc = tmpl.canvas_cov.copy()
                if side == "keep-left":
                    cc[:, c0 + w:] = False
                else:
                    cc[:, :c1 - w] = False
                if cc.sum() < 3000:
                    continue
                sub = Template(tmpl.canvas, cc, tmpl.strips, tmpl.origin)
                r = match_canvas(sub, probe)
                if r is None:
                    continue
                ok, why, _ = decide(r, None, ROUTE_CANVAS)
                err = pose_distance(r["pose_canvas"], Ptrue, probe.img.shape)
                nn = []
                for _ in range(n_null):
                    pi = phase_randomise(probe.img, probe.cov, rng)
                    rn = match_canvas(sub, _fake_strip("null", pi, probe.cov))
                    if rn:
                        nn.append((rn["ncc"], rn["explained"]))
                rec = {"held": held, "side": side, "frac": frac,
                       "map_mm2": round(float(cc.sum()) * MM_PER_PX ** 2, 1),
                       "overlap_mm2": r["overlap_mm2"],
                       "overlap_frac": r["overlap_frac"],
                       "ncc": r["ncc"], "explained": r["explained"],
                       "psr": r["psr"], "theta": r["theta"], "sx": r["sx"],
                       "err_px": round(err, 2),
                       "err_mm": round(err * MM_PER_PX, 3),
                       "accepted": bool(ok), "why": why,
                       "null_ncc_mean": (round(float(np.mean([x[0] for x in nn])), 4)
                                         if nn else None),
                       "null_ncc_max": (round(float(np.max([x[0] for x in nn])), 4)
                                        if nn else None),
                       "null_expl_max": (round(float(np.max([x[1] for x in nn])), 4)
                                         if nn else None)}
                rows.append(rec)
                if verbose:
                    print(f"  {side:<11}{frac:>5.1f}{rec['map_mm2']:>7.1f}"
                          f"{rec['overlap_mm2']:>7.1f}{rec['overlap_frac']:>7.2f}"
                          f"{rec['ncc']:>7.3f}{rec['explained']:>7.3f}"
                          f"{rec['psr']:>6.1f}{rec['err_px']:>8.2f}  "
                          f"{'ACCEPT' if ok else 'reject'}")

    out = {"rows": rows, "summary": analyse_sweep(rows)}
    if verbose:
        _print_sweep_summary(out["summary"])
    return out


def analyse_sweep(rows):
    """Which statistic actually predicts whether the answer is right?

    Split the sweep by measured pose error, not by verdict, and report the
    min of the accurate group against the max of the wrong group.  Means are
    useless here -- what matters is whether the two groups can be separated by
    a threshold at all, and only the min/max columns say that.
    """
    if not rows:
        return {}
    good = [r for r in rows if r["err_px"] <= 5.0]
    bad = [r for r in rows if r["err_px"] > 15.0]
    stats = {}
    for k in ("ncc", "psr", "explained", "overlap_mm2", "overlap_frac"):
        g = np.array([r[k] for r in good]) if good else np.array([np.nan])
        b = np.array([r[k] for r in bad]) if bad else np.array([np.nan])
        stats[k] = {
            "accurate_n": len(good), "accurate_mean": round(float(g.mean()), 4),
            "accurate_min": round(float(g.min()), 4),
            "wrong_n": len(bad), "wrong_mean": round(float(b.mean()), 4),
            "wrong_max": round(float(b.max()), 4),
            "separable": bool(len(good) and len(bad) and g.min() > b.max()),
            "ambiguous_band": [round(float(min(g.min(), b.max())), 4),
                               round(float(max(g.min(), b.max())), 4)],
        }
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]
    nulls = [r["null_ncc_max"] for r in rows if r["null_ncc_max"] is not None]
    return {
        "n": len(rows),
        "n_accepted": len(acc), "n_rejected": len(rej),
        "worst_err_px_accepted": round(max((r["err_px"] for r in acc),
                                           default=float("nan")), 2),
        "min_overlap_mm2_accepted": round(min((r["overlap_mm2"] for r in acc),
                                              default=float("nan")), 2),
        "min_explained_accepted": round(min((r["explained"] for r in acc),
                                            default=float("nan")), 4),
        "predictors": stats,
        "null_ncc_max_over_sweep": round(max(nulls), 4) if nulls else None,
        "null_flat_in_overlap": _null_slope(rows),
    }


def _null_slope(rows):
    """Log-log slope of the null score against overlap.

    The textbook expectation is -0.5: fewer samples, noisier correlation, higher
    chance maximum.  Measured here it is essentially zero, because the null is
    not one correlation over N samples but the MAXIMUM over tens of thousands of
    shifts of a field with a 54.7 px autocorrelation area, and cropping the map
    shrinks the shift set at the same time as it shrinks N.  The two effects
    very nearly cancel, which is why NCC is not rescaled by overlap anywhere in
    this file.
    """
    pts = [(r["overlap_mm2"], r["null_ncc_mean"]) for r in rows
           if r.get("null_ncc_mean")]
    if len(pts) < 4:
        return None
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    slope = float(np.polyfit(x, y, 1)[0])
    return {"log_log_slope": round(slope, 3),
            "sqrt_law_would_be": -0.5,
            "n": len(pts)}


def _print_sweep_summary(s):
    if not s:
        return
    print("\n" + "=" * 74)
    print(f"overlap sweep: {s['n']} registrations, {s['n_accepted']} accepted")
    print(f"  worst pose error among ACCEPTED: {s['worst_err_px_accepted']} px")
    print(f"  least overlap among ACCEPTED:    "
          f"{s['min_overlap_mm2_accepted']} mm^2")
    print(f"  least explained among ACCEPTED:  {s['min_explained_accepted']}")
    print("\n  which statistic predicts a correct pose?")
    print(f"  {'stat':<14}{'accurate<=5px':>22}{'wrong>15px':>22}  separable")
    for k, v in s["predictors"].items():
        print(f"  {k:<14}"
              f"{('n=%d %.3f min %.3f' % (v['accurate_n'], v['accurate_mean'], v['accurate_min'])):>22}"
              f"{('n=%d %.3f MAX %.3f' % (v['wrong_n'], v['wrong_mean'], v['wrong_max'])):>22}"
              f"  {'YES' if v['separable'] else 'no, band ' + str(v['ambiguous_band'])}")
    if s.get("null_flat_in_overlap"):
        n = s["null_flat_in_overlap"]
        print(f"\n  null score vs overlap: log-log slope "
              f"{n['log_log_slope']:+.3f} over {n['n']} points "
              f"(the 1/sqrt(N) law would give {n['sqrt_law_would_be']})")
        print("  -> NCC needs no overlap normalisation over this range; what "
              "it needs is\n     a hard overlap floor and the `explained` test.")


# ---------------------------------------------------------------------------
# 5b. score vs PROBE length: the short swipe a real user gives you
# ---------------------------------------------------------------------------
def probe_sweep(capture=None, fracs=(1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.25),
                ends=("keep-head", "keep-tail"), n_null=3, verbose=True):
    """Shorten the PROBE instead of the map, and re-register against a fixed map.

    `overlap_sweep` crops the enrolled map and holds the probe byte-identical,
    which isolates the variable cleanly and answers the ENROLMENT question: how
    much map is enough?  It does not answer the VERIFICATION question, and the
    difference is not cosmetic.

    Cropping the map moves the overlap and leaves the probe area alone, so
    `explained` = ncc * overlap/probe_area falls only because the numerator
    falls.  Shortening the probe moves BOTH: a probe half as long has half the
    area, so a probe that still lands entirely on the map keeps its overlap
    FRACTION at ~1.0 and `explained` barely moves, while the absolute overlap --
    the thing the hard gate tests -- halves.  The two sweeps therefore stress
    different tests in the acceptance rule, and a short swipe is the case the
    device's own 420-563 ms wedge makes routine.

    Ground truth is exact and does not come from the matcher.  Each truncated
    probe is re-assembled from a contiguous run of the full probe's frames, and
    `_mean_offset` gives the constant offset between the sub-strip's own track
    and the full strip's, measured on the frames they share.  So

        pose_true(sub) = pose(full) @ translation(-offset)

    where pose(full) is the full probe's registered pose.  That makes the error
    column a measurement of what truncation costs, referred to the full-length
    answer, with the ground-truth link itself accurate to the offset spread
    reported as `gt_spread_px`.

    Both ends are swept because they are not equivalent: the head of a swipe is
    the contact ramp (swipe_assemble measured band fraction 0.045-0.147 there
    against 0.75 mid-swipe) and the tail is the lift-off.
    """
    capture = Path(capture or CAPTURES[2])
    others = [s for s in load_real_strips()
              if not s.name.startswith(Path(capture).name)]
    if not others:
        return {"error": "no other captures to build a map from"}
    tmpl, _ = _build_map(others)
    tmpl.null = calibrate_null(tmpl, n=6, verbose=False)

    frames, _ = load_swipe(capture)
    segs = segment_capture(frames, list(range(DX_MIN, DX_MAX + 1)),
                           list(range(-DY_ABS, DY_ABS + 1)))
    if not segs:
        return {"error": "no usable swipe segment"}
    a0, b0 = max(segs, key=lambda s: s[1] - s[0])
    full = assemble_one(frames[a0:b0], "full", verbose=False)
    if full is None:
        return {"error": "the full probe did not assemble"}
    kept = full.frames
    nk = len(kept)

    rfull = match_canvas(tmpl, full)
    if rfull is None:
        return {"error": "the full probe did not register"}
    Pfull = rfull["pose_canvas"]

    if verbose:
        print(f"probe {capture.name}: {nk} in-contact frames, "
              f"{full.report['coverage_mm2']:.1f} mm^2, against a map of "
              f"{tmpl.coverage_mm2:.1f} mm^2 from {len(tmpl.strips)} strips")
        print(f"  full-length probe: NCC {rfull['ncc']:.3f}, overlap "
              f"{rfull['overlap_mm2']:.1f} mm^2 = "
              f"{rfull['overlap_frac']:.0%} of the probe")
        print(f"\n  {'end':<11}{'frac':>5}{'nfr':>5}{'probe':>7}{'ovlap':>7}"
              f"{'ovfrac':>7}{'NCC':>7}{'expl':>7}{'PSR':>6}{'err_px':>8}"
              f"{'null':>7}  verdict")

    rng = np.random.default_rng(20260814)
    rows = []
    for end in ends:
        for frac in fracs:
            n = int(round(nk * frac))
            if n < 4:
                continue
            i0 = 0 if end == "keep-head" else nk - n
            sub = assemble_one(kept[i0:i0 + n], f"{end}-{frac}", verbose=False)
            if sub is None:
                continue
            k0 = i0 + sub.report["contact_window"][0]
            oy, ox, sy, sx_, nsh = _mean_offset(sub, full, k0)
            Ptrue = Pfull @ translation(-oy, -ox)
            r = match_canvas(tmpl, sub)
            if r is None:
                rows.append({"end": end, "frac": frac, "n_frames": n,
                             "registered": False})
                continue
            ok, why, _ = decide(r, tmpl, ROUTE_CANVAS)
            err = pose_distance(r["pose_canvas"], Ptrue, sub.img.shape)
            nn = []
            for _ in range(n_null):
                pi = phase_randomise(sub.img, sub.cov, rng)
                rn = match_canvas(tmpl, _fake_strip("null", pi, sub.cov))
                if rn:
                    nn.append(rn["ncc"])
            rec = {"end": end, "frac": frac, "n_frames": int(n),
                   "registered": True,
                   "probe_mm2": round(float(sub.cov.sum()) * MM_PER_PX ** 2, 1),
                   "probe_len_mm": round(
                       float(np.ptp(np.where(sub.cov.any(0))[0]) + 1)
                       * MM_PER_PX, 2),
                   "overlap_mm2": r["overlap_mm2"],
                   "overlap_frac": r["overlap_frac"],
                   "ncc": r["ncc"], "explained": r["explained"],
                   "psr": r["psr"], "theta": r["theta"], "sx": r["sx"],
                   "err_px": round(err, 2),
                   "err_mm": round(err * MM_PER_PX, 3),
                   "gt_spread_px": round(math.hypot(sy, sx_), 3),
                   "gt_shared_frames": int(nsh),
                   "null_ncc_max": (round(float(np.max(nn)), 4) if nn else None),
                   "accepted": bool(ok), "why": why}
            rows.append(rec)
            if verbose:
                print(f"  {end:<11}{frac:>5.2f}{n:>5d}{rec['probe_mm2']:>7.1f}"
                      f"{rec['overlap_mm2']:>7.1f}{rec['overlap_frac']:>7.2f}"
                      f"{rec['ncc']:>7.3f}{rec['explained']:>7.3f}"
                      f"{rec['psr']:>6.1f}{rec['err_px']:>8.2f}"
                      f"{(rec['null_ncc_max'] or float('nan')):>7.3f}  "
                      f"{'ACCEPT' if ok else 'reject'}")

    out = {"capture": str(capture), "n_frames_full": int(nk),
           "map_mm2": round(tmpl.coverage_mm2, 1),
           "full_probe": {"ncc": rfull["ncc"],
                          "overlap_mm2": rfull["overlap_mm2"],
                          "overlap_frac": rfull["overlap_frac"],
                          "explained": rfull["explained"]},
           "rows": rows, "summary": _analyse_probe_sweep(rows, nk)}
    if verbose:
        _print_probe_summary(out["summary"])
    return out


def _analyse_probe_sweep(rows, nk):
    """Where does truncation start to break the answer, and which test catches it?"""
    good = [r for r in rows if r.get("registered")]
    if not good:
        return {}
    acc = [r for r in good if r["accepted"]]
    ok5 = [r for r in good if r["err_px"] <= 5.0]
    bad = [r for r in good if r["err_px"] > 15.0]
    # the shortest probe that both registered accurately AND was accepted
    safe = [r for r in acc if r["err_px"] <= 5.0]
    # which condition fires first as the probe shortens?
    first_fail = {}
    for r in sorted(good, key=lambda z: -z["frac"]):
        if not r["accepted"]:
            for part in r["why"].split(";"):
                key = part.strip().split()[0]
                first_fail.setdefault(key, r["frac"])
    return {
        "n": len(good), "n_accepted": len(acc),
        "min_frac_accepted": round(min((r["frac"] for r in acc), default=float("nan")), 3),
        "min_frames_accepted": min((r["n_frames"] for r in acc), default=None),
        "min_probe_mm2_accepted": round(min((r["probe_mm2"] for r in acc), default=float("nan")), 1),
        "min_probe_len_mm_accepted": round(min((r["probe_len_mm"] for r in acc), default=float("nan")), 2),
        "min_overlap_mm2_accepted": round(min((r["overlap_mm2"] for r in acc), default=float("nan")), 1),
        "worst_err_px_accepted": round(max((r["err_px"] for r in acc), default=float("nan")), 2),
        "safest_short": (min(safe, key=lambda r: r["frac"]) if safe else None),
        "overlap_frac_stays_high": {
            "min_over_accepted": round(min((r["overlap_frac"] for r in acc),
                                           default=float("nan")), 3),
            "note": ("if this stays near 1.0 while overlap_mm2 falls, "
                     "`explained` is NOT the test that catches a short swipe -- "
                     "the hard mm^2 floor is"),
        },
        "which_test_fires_first_as_probe_shortens": first_fail,
        "accurate_n": len(ok5), "wrong_n": len(bad),
        "null_ncc_max": round(max((r["null_ncc_max"] or 0) for r in good), 4),
        "ncc_vs_frac_slope": _probe_slope(good, "ncc"),
        "explained_vs_frac_slope": _probe_slope(good, "explained"),
    }


def _probe_slope(rows, key):
    pts = [(r["frac"], r[key]) for r in rows if r["frac"] >= 0.35]
    if len(pts) < 4:
        return None
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return round(float(np.polyfit(x, y, 1)[0]), 4)


def _print_probe_summary(s):
    if not s:
        return
    print("\n" + "=" * 74)
    print(f"probe sweep: {s['n']} registrations, {s['n_accepted']} accepted")
    if s.get("min_frames_accepted"):
        print(f"  shortest ACCEPTED probe: {s['min_frames_accepted']} frames, "
              f"{s['min_probe_len_mm_accepted']} mm long, "
              f"{s['min_probe_mm2_accepted']} mm^2, overlapping "
              f"{s['min_overlap_mm2_accepted']} mm^2 of map")
    print(f"  worst pose error among ACCEPTED: {s['worst_err_px_accepted']} px")
    o = s["overlap_frac_stays_high"]
    print(f"  overlap FRACTION of probe, worst accepted: {o['min_over_accepted']}")
    print(f"  -> {o['note']}")
    if s["which_test_fires_first_as_probe_shortens"]:
        print("  first rejection cause, by the largest frac at which it fires:")
        for k, v in sorted(s["which_test_fires_first_as_probe_shortens"].items(),
                           key=lambda kv: -kv[1]):
            print(f"    {k:<12} first fires at frac {v}")
    print(f"  NCC vs frac slope {s['ncc_vs_frac_slope']}, "
          f"explained vs frac slope {s['explained_vs_frac_slope']} "
          f"(over frac >= 0.35)")
    print(f"  null NCC max over the sweep: {s['null_ncc_max']}")


# ---------------------------------------------------------------------------
# 6. does this matcher see finger identity at all?
# ---------------------------------------------------------------------------
def press_check(n_pairs=30, verbose=True):
    """The one comparison on this machine where the two sides are DIFFERENT FINGERS.

    The press dataset has right-index (12), right-index-cover (19) and
    right-middle (14).  All of them are averaged captures from the old driver
    and are far blurrier than a swipe strip, so no number here transfers to the
    swipe system.  What it does control for is image quality: press-to-press
    keeps the same degradation on both sides, so the gap between same-finger
    and different-finger pairs isolates identity.

    Measured, it is not a weak signal, it is no signal -- right-middle against
    ITSELF scores lower than right-index against right-middle.  Two explanations
    fit and this data cannot separate them: the averaged presses may be too
    blurred to carry identity, and two presses of one finger need not image the
    same part of that finger (the window is 7.6 x 2.6 mm and the fingertip is
    not).  Either way, the 45 presses in swipe_map's null distribution were
    bounding non-matching CONTENT, not impostors, and this file's tier-4 press
    result must be read the same way.
    """
    import itertools
    groups = {}
    for g in ("right-index", "right-index-cover", "right-middle"):
        d = DATASET / g
        if d.exists():
            groups[g] = sorted(d.glob("*.pgm"))
    if len(groups) < 3:
        return {"error": "press dataset not present"}
    angles = np.arange(-10, 10.001, 1.0)
    rng = np.random.default_rng(7)

    def one(pa, pb):
        a, b = read_pgm(pa), read_pgm(pb)
        return register_strip(a, np.ones(a.shape, bool),
                              b, np.ones(b.shape, bool),
                              angles=angles, refine=False, split_half=False,
                              min_overlap=4000, scales=None)

    def sample(pairs):
        pairs = list(pairs)
        if len(pairs) <= n_pairs:
            return pairs
        return [pairs[i] for i in rng.choice(len(pairs), n_pairs, replace=False)]

    tests = {
        "same finger: right-index":
            sample(itertools.combinations(groups["right-index"], 2)),
        "same finger: right-middle":
            sample(itertools.combinations(groups["right-middle"], 2)),
        "same finger: right-index-cover":
            sample(itertools.combinations(groups["right-index-cover"], 2)),
        "same finger: index vs index-cover":
            sample(itertools.product(groups["right-index"],
                                     groups["right-index-cover"])),
        "DIFFERENT finger: index vs middle":
            sample(itertools.product(groups["right-index"],
                                     groups["right-middle"])),
        "DIFFERENT finger: index-cover vs middle":
            sample(itertools.product(groups["right-index-cover"],
                                     groups["right-middle"])),
    }
    out = {}
    for name, pairs in tests.items():
        v = []
        for pa, pb in pairs:
            r = one(pa, pb)
            if r:
                v.append((r["ncc"], r["psr"]))
        if not v:
            continue
        n = np.array([x[0] for x in v])
        p = np.array([x[1] for x in v])
        out[name] = {"n": len(n), "ncc_mean": round(float(n.mean()), 4),
                     "ncc_sd": round(float(n.std(ddof=1)), 4),
                     "ncc_min": round(float(n.min()), 4),
                     "ncc_max": round(float(n.max()), 4),
                     "psr_max": round(float(p.max()), 2)}
        if verbose:
            print(f"  {name:<42} n={len(n):<3} NCC {n.mean():.3f} "
                  f"+- {n.std(ddof=1):.3f}  [{n.min():.3f}, {n.max():.3f}]")
    same = [v["ncc_mean"] for k, v in out.items() if k.startswith("same")]
    diff = [v["ncc_mean"] for k, v in out.items() if k.startswith("DIFFERENT")]
    verdict = bool(same and diff and min(same) > max(diff))
    out["_verdict"] = {
        "every_same_finger_group_beats_every_different_finger_group": verdict,
        "same_finger_means": same, "different_finger_means": diff,
        "conclusion": ("no usable finger discrimination through this matcher on "
                       "the press captures" if not verdict else
                       "some discrimination, but on press captures only"),
    }
    if verbose:
        print(f"\n  every same-finger group above every different-finger "
              f"group? {'YES' if verdict else 'NO'}")
        print(f"  -> {out['_verdict']['conclusion']}")
        print("  This is why the press dataset cannot stand in for an "
              "impostor set, and why\n  no FAR appears anywhere in this file.")
    return out


# ---------------------------------------------------------------------------
# 7. how many enrolment swipes?
# ---------------------------------------------------------------------------
# A fingertip's contact patch is the one quantity in this file that is NOT
# measured on this machine, because nothing here images a whole fingertip.  The
# range below is geometry plus the usual capture standards: a plain-impression
# capture area is 16 x 20 mm, and the actual contact ellipse of an adult index
# finger under normal press force is smaller, roughly 13 x 17 mm, which as an
# ellipse is 0.785 * 13 * 17 = 174 mm^2.  Everything downstream is reported
# across the range so that the assumption is visible rather than buried.
FINGERTIP_MM2 = (150.0, 175.0, 250.0)     # low, central, high
FINGERTIP_WIDTH_MM = (11.0, 13.0, 16.0)   # across the finger, the binding axis


def coverage_model(verbose=True):
    """From measured per-swipe coverage, how many swipes map a finger?

    Every input except the fingertip size is measured on this machine.  The
    calculation is deliberately done twice, because the two answers are a factor
    of three apart and the difference is entirely about whether enrolment tells
    the user to move the finger sideways.
    """
    strips = load_real_strips()
    t_all, rej = _build_map(list(strips))
    per = []
    for s in t_all.strips:
        cov = s.cov
        cols = np.where(cov.any(0))[0]
        length_px = float(cols.max() - cols.min() + 1) if cols.size else 0.0
        area = float(cov.sum())
        per.append({
            "name": s.name,
            "covered_mm2": round(area * MM_PER_PX ** 2, 2),
            "length_mm": round(length_px * MM_PER_PX, 2),
            # effective height = covered area / length.  NOT the bounding box
            # height: a strip is a parallelogram and its bbox overstates it by
            # about a third, which is the mistake the 3.9x figure in the brief
            # rests on.
            "eff_height_mm": round(area / max(length_px, 1) * MM_PER_PX, 3),
        })
    A_swipe = float(np.mean([p["covered_mm2"] for p in per]))
    L = float(np.mean([p["length_mm"] for p in per]))
    H = float(np.mean([p["eff_height_mm"] for p in per]))
    map_mm2 = t_all.coverage_mm2
    n_acc = len(t_all.strips)
    seed_mm2 = max(p["covered_mm2"] for p in per)
    marginal = (map_mm2 - seed_mm2) / max(n_acc - 1, 1)

    ch, cw = t_all.canvas_cov.shape
    rows = np.where(t_all.canvas_cov.any(1))[0]
    canvas_h_mm = float(rows.max() - rows.min() + 1) * MM_PER_PX

    # -- (a) unguided: repeat the same swipe --------------------------------
    # The measured marginal gain per extra swipe, and the ceiling it is heading
    # for: the lateral scatter of an unguided finger is exactly what the canvas
    # height minus one strip height measures.
    scatter_mm = max(canvas_h_mm - H, 0.0)
    ceiling_unguided = L * (H + scatter_mm)

    # -- (b) guided: lanes with a deliberate lateral step -------------------
    # A swipe already covers the finger's whole LENGTH (L is 16-17 mm against a
    # 17-20 mm fingertip), so the map can only grow sideways.  Two lanes must
    # overlap by at least MIN_VERIFY_OVERLAP_MM2 to be linkable, and that
    # overlap is a band of height h running the shared length L.
    h_needed = MIN_VERIFY_OVERLAP_MM2 / max(L, 1e-6)
    step = H - h_needed
    lanes = {}
    for w in FINGERTIP_WIDTH_MM:
        n = 1 + max(0.0, (w - H)) / max(step, 1e-6)
        lanes[w] = {
            "lanes_needed": math.ceil(n - 1e-9),
            "area_mapped_mm2": round(L * min(w, H + (math.ceil(n) - 1) * step), 1),
        }

    rej_rate = len(rej) / max(len(strips), 1)

    out = {
        "measured": {
            "per_strip": per,
            "mean_covered_mm2_per_swipe": round(A_swipe, 2),
            "mean_strip_length_mm": round(L, 2),
            "mean_effective_height_mm": round(H, 3),
            "map_mm2_from_%d_swipes" % n_acc: round(map_mm2, 2),
            "marginal_mm2_per_extra_swipe": round(marginal, 2),
            "canvas_height_mm": round(canvas_h_mm, 2),
            "unguided_lateral_scatter_mm": round(scatter_mm, 2),
            "enrolment_rejection_rate": round(rej_rate, 3),
        },
        "assumed_fingertip": {
            "area_mm2_low_central_high": FINGERTIP_MM2,
            "width_mm_low_central_high": FINGERTIP_WIDTH_MM,
            "note": "NOT measured here; nothing in this dataset images a "
                    "whole fingertip.",
        },
        "unguided": {
            "marginal_mm2_per_swipe": round(marginal, 2),
            "ceiling_mm2": round(ceiling_unguided, 1),
            "ceiling_as_fraction_of_fingertip":
                [round(ceiling_unguided / a, 2) for a in FINGERTIP_MM2],
            "swipes_to_ceiling": (
                math.ceil((ceiling_unguided - seed_mm2) / max(marginal, 1e-6)) + 1),
            "verdict": "unguided enrolment cannot map the finger: it saturates "
                       "at a band as tall as one strip plus the user's own "
                       "lateral scatter",
        },
        "guided": {
            "overlap_band_needed_mm": round(h_needed, 2),
            "lateral_step_per_lane_mm": round(step, 2),
            "by_finger_width": {str(k): v for k, v in lanes.items()},
            "accepted_swipes_needed": [lanes[w]["lanes_needed"]
                                       for w in FINGERTIP_WIDTH_MM],
            "attempts_needed_at_measured_rejection_rate":
                [math.ceil(lanes[w]["lanes_needed"] / max(1 - rej_rate, 1e-6))
                 for w in FINGERTIP_WIDTH_MM],
        },
    }

    # -- (c) what fraction of a verification swipe lands on the map? --------
    # Measured directly, leave-one-out, rather than modelled: each held-out
    # strip's overlap with a map built from the other three.
    ov = []
    by = {s.name: s for s in strips}
    for held in [s.name for s in t_all.strips]:
        tmpl, _ = _build_map([by[n] for n in
                              [x.name for x in t_all.strips] if n != held])
        r = match_canvas(tmpl, by[held])
        if r:
            ov.append({"held": held, "map_mm2": round(tmpl.coverage_mm2, 1),
                       "overlap_mm2": r["overlap_mm2"],
                       "overlap_frac_of_probe": r["overlap_frac"]})
    fr = [o["overlap_frac_of_probe"] for o in ov]
    out["verification_overlap"] = {
        "measured_leave_one_out": ov,
        "frac_of_probe_min": round(min(fr), 3) if fr else None,
        "frac_of_probe_max": round(max(fr), 3) if fr else None,
        "overlap_mm2_min": round(min(o["overlap_mm2"] for o in ov), 1) if ov else None,
        "margin_over_requirement": (
            round(min(o["overlap_mm2"] for o in ov) / MIN_VERIFY_OVERLAP_MM2, 2)
            if ov else None),
        "note": ("measured against a 3-swipe map covering 57-70 mm^2.  A "
                 "guided enrolment covering the full finger width would raise "
                 "this, because the only thing limiting it is the map running "
                 "out sideways -- but that is an extrapolation, not a "
                 "measurement."),
    }

    if verbose:
        m = out["measured"]
        print("measured, per swipe:")
        for p in per:
            print(f"  {p['name']:<22} {p['covered_mm2']:>6.1f} mm^2  "
                  f"{p['length_mm']:>5.1f} x {p['eff_height_mm']:.2f} mm "
                  f"(effective, not bbox)")
        print(f"  mean {A_swipe:.1f} mm^2, {L:.1f} mm long, {H:.2f} mm tall")
        print(f"  {n_acc} swipes -> map {map_mm2:.1f} mm^2; marginal gain per "
              f"extra swipe {marginal:.1f} mm^2")
        print(f"  canvas is {canvas_h_mm:.2f} mm tall against a "
              f"{H:.2f} mm strip, so unguided lateral scatter is "
              f"{scatter_mm:.2f} mm")
        u = out["unguided"]
        print(f"\nunguided (repeat the same swipe):")
        print(f"  ceiling {u['ceiling_mm2']:.0f} mm^2 = "
              f"{u['ceiling_as_fraction_of_fingertip']} of a "
              f"{FINGERTIP_MM2} mm^2 fingertip")
        print(f"  {u['verdict']}")
        g = out["guided"]
        print(f"\nguided (step the finger sideways between swipes):")
        print(f"  lanes must share a {g['overlap_band_needed_mm']:.2f} mm band "
              f"to reach {MIN_VERIFY_OVERLAP_MM2:.0f} mm^2 of overlap, so the "
              f"step is {g['lateral_step_per_lane_mm']:.2f} mm")
        for w in FINGERTIP_WIDTH_MM:
            v = lanes[w]
            print(f"  finger {w:.0f} mm wide -> {v['lanes_needed']} accepted "
                  f"swipes, {v['area_mapped_mm2']:.0f} mm^2 mapped")
        print(f"  at the measured {rej_rate:.0%} enrolment rejection rate that "
              f"is {g['attempts_needed_at_measured_rejection_rate']} attempts")
        v = out["verification_overlap"]
        print(f"\nverification overlap, measured leave-one-out:")
        for o in ov:
            print(f"  {o['held']:<22} {o['overlap_mm2']:>5.1f} mm^2 = "
                  f"{o['overlap_frac_of_probe']:.0%} of the probe "
                  f"(map {o['map_mm2']:.0f} mm^2)")
        print(f"  worst case {v['overlap_mm2_min']} mm^2, "
              f"{v['margin_over_requirement']}x the "
              f"{MIN_VERIFY_OVERLAP_MM2:.0f} mm^2 requirement")
    return out


# ---------------------------------------------------------------------------
# 8. the collection that would make any of this a security claim
# ---------------------------------------------------------------------------
def protocol(verbose=True):
    """The exact data collection needed, and what each part buys.

    Written as a protocol rather than a wish list because the binding constraint
    is not analysis, it is that this device wedges after 420-563 ms and needs a
    USB selective suspend/resume between captures.  A session's length is set by
    how many suspend cycles the person will sit through, so the protocol has to
    be costed in swipes.
    """
    # Rule of three: zero failures in N INDEPENDENT trials bounds the rate at
    # 3/N with 95% confidence.  That is the whole arithmetic of a FAR claim,
    # and the word doing the work is "independent".
    #
    # Cross-comparisons are NOT independent.  F fingers with P probes each give
    # F*P*(F-1) comparisons, but they are built from only F templates and F*P
    # probes, so a single unusually generic finger -- shallow ridges, a common
    # ridge-flow class -- contributes correlated failures to (F-1)*P of them at
    # once.  This is the standard caveat in biometric evaluation (ISO/IEC
    # 19795-1 treats the subject, not the comparison, as the sampling unit) and
    # it is why published FAR claims quote the number of SUBJECTS.
    #
    # The conservative reading takes the distinct finger PAIRS as the effective
    # sample: C(F,2) = F*(F-1)/2.  For the 20 fingers of set A that is 190, not
    # 3800, and the rule of three then bounds FAR only at 3/190 = 1.6e-2 --
    # twenty times weaker than the naive count suggests.  The truth is between
    # the two and depends on how much of the variance is between-finger rather
    # than within-finger, which cannot be known before the data exists.
    #
    # So `impostor_comparisons` below is the OPTIMISTIC count and
    # `fingers_needed_*` derived from it is a lower bound on the collection.
    # `subject_limited_far_bound` is the pessimistic one.  Report both, design
    # for the pessimistic one, and never quote the optimistic one alone.
    tiers = []
    for far in (1e-2, 1e-3, 1e-4, 1e-5):
        n = math.ceil(3.0 / far)
        tiers.append({"far_upper_bound_95pct": far, "impostor_comparisons": n})

    swipes_per_finger = 10
    for t in tiers:
        # each probe swipe is compared against every OTHER finger's template
        # -> n_fingers * swipes * (n_fingers - 1) comparisons
        n = t["impostor_comparisons"]
        f = 1
        while f * swipes_per_finger * (f - 1) < n and f < 100000:
            f += 1
        t["fingers_needed_at_10_swipes_each"] = f
        t["total_swipes"] = f * (swipes_per_finger + 8)   # + enrolment
        # the same F fingers, counted the conservative way: distinct pairs
        pairs = f * (f - 1) // 2
        t["distinct_finger_pairs"] = pairs
        t["subject_limited_far_bound"] = round(3.0 / pairs, 6) if pairs else None
        # how many fingers would be needed if pairs, not comparisons, is the
        # right sampling unit: solve F(F-1)/2 >= 3/far.
        # NB use t[...], not `far`: `far` is the earlier loop's variable and
        # still holds its LAST value here, which silently gave every tier the
        # 1e-5 answer (776) before this was caught.
        need = 3.0 / t["far_upper_bound_95pct"]
        f2 = 2
        while f2 * (f2 - 1) / 2 < need and f2 < 1000000:
            f2 += 1
        t["fingers_needed_if_pair_limited"] = f2

    out = {
        "why": ("Every number in this file is one finger of one person.  There "
                "is no impostor distribution, so there is no FAR, no EER and "
                "no d-prime, and none should be quoted."),
        "far_arithmetic": tiers,
        "collection": [
            {
                "name": "A. impostor set",
                "what": "different fingers, ideally different people",
                "who": "at least 7 people, 4 fingers each (index and middle, "
                       "both hands) = 28 fingers.  Raised from 20: 28 fingers "
                       "give 378 distinct pairs, which clears the 25 fingers "
                       "the pair-limited rule of three needs for a 1e-2 bound. "
                       "20 fingers clears the naive count and not the honest "
                       "one, which is the worst place to stop.",
                "how_many_swipes": "8 enrolment + 10 verification per finger "
                                   "= 504 swipes",
                "buys": "20 * 10 * 19 = 3800 impostor comparisons.  Counted "
                        "naively that bounds FAR below 1e-3 at 95% confidence "
                        "if none is accepted, but the comparisons are not "
                        "independent -- they come from 20 templates and 190 "
                        "distinct finger pairs -- so the defensible bound is "
                        "3/190 = 1.6e-2, and the honest claim is 'no false "
                        "accept in 20 fingers'.  Neither reaches 1e-5, the "
                        "consumer-biometric expectation, which needs ~800 "
                        "fingers pair-limited.  See far_arithmetic.",
                "critical": "the impostor comparisons must use the SAME "
                            "enrolment procedure and the same thresholds as "
                            "the genuine ones, and the thresholds must be "
                            "frozen before the impostor data is looked at.",
            },
            {
                "name": "B. genuine repeatability",
                "what": "the same fingers again, on other days",
                "who": "the same 28 fingers",
                "how_many_swipes": "10 verification swipes per finger on each "
                                   "of 3 further days = 840 swipes",
                "buys": "a false-reject rate with day-to-day skin variation in "
                        "it.  A single-session FRR is meaningless: moisture, "
                        "temperature and callus change the ridge contrast far "
                        "more than anything measured here.",
                "critical": "record ambient conditions and whether the finger "
                            "was washed, because the failure mode this system "
                            "will actually have is a dry finger in winter.",
            },
            {
                "name": "C. enrolment protocol comparison",
                "what": "guided lateral stepping vs unguided repetition",
                "who": "4 fingers",
                "how_many_swipes": "two enrolments each, 8 swipes each = 64",
                "buys": "the central claim of coverage_model, which is "
                        "currently an extrapolation from 4 swipes that all "
                        "landed in the same lane: that unguided enrolment "
                        "saturates at about half the finger.  Measure the map "
                        "area after each swipe of each enrolment and plot it.",
            },
            {
                "name": "D. speed and angle range",
                "what": "deliberately fast, deliberately slow, and rotated",
                "who": "2 fingers",
                "how_many_swipes": "5 fast, 5 slow, 5 at about +20 deg, 5 at "
                                   "about -20 deg = 40",
                "buys": "whether DX_MAX = 36 px/frame is enough headroom (one "
                        "real swipe already reached 28), and whether the "
                        "+-25 deg rotation search and the 0.90-1.10 scale "
                        "search rail out.  Both currently rest on 6 strips "
                        "from one relaxed session.",
            },
            {
                "name": "E. spoof and near-miss",
                "what": "the adjacent finger of the SAME hand, and the same "
                        "finger of the other hand",
                "who": "the 7 people from A",
                "how_many_swipes": "10 per person per case = 140",
                "buys": "the hardest impostors there are.  A random stranger's "
                        "finger is an easy negative; the neighbouring finger "
                        "shares ridge flow, ridge pitch and skin condition, "
                        "and swipe_map already measured that a strip turned "
                        "END FOR END still correlates at 0.474 on ridge flow "
                        "alone.",
            },
        ],
        "total_swipes": 504 + 840 + 64 + 40 + 140,
        "device_constraint": (
            "capture wedges after 420-563 ms with a persistent 0xaf from "
            "pre_scan and needs a USB selective suspend/resume to clear.  At "
            "roughly 5 s per swipe including the recovery, 1588 swipes is "
            "about 2.2 hours of pure capture, so it is 8-10 sittings, not "
            "one, and it needs 7 people rather than 5."),
        "analysis_that_follows": [
            "freeze ACCEPT_NCC_CANVAS, MIN_EXPLAINED and "
            "MIN_VERIFY_OVERLAP_MM2 on set C+D BEFORE touching A",
            "report DET curves, not a single operating point",
            "report FRR at the FAR the impostor set can actually bound, and "
            "say so; do not extrapolate the tail",
            "report per-finger, not pooled: pooling hides the one finger whose "
            "ridges are too shallow to work at all",
        ],
    }
    if verbose:
        print(out["why"])
        print("\nwhat a FAR claim costs (rule of three, 95% confidence):")
        print("  comparisons are NOT independent, so two columns are given:")
        print(f"  {'FAR':>8}{'comparisons':>13}{'fingers':>9}"
              f"{'fingers if pair-limited':>25}")
        for t in tiers:
            print(f"  {t['far_upper_bound_95pct']:>8.0e}"
                  f"{t['impostor_comparisons']:>13}"
                  f"{t['fingers_needed_at_10_swipes_each']:>9}"
                  f"{t['fingers_needed_if_pair_limited']:>25}")
        print("  the left count assumes every comparison is an independent "
              "trial; the right\n  treats the finger PAIR as the sampling unit."
              "  Design for the right one.")
        print("\ncollection:")
        for c in out["collection"]:
            print(f"\n  {c['name']}")
            print(f"    who   {c['who']}")
            print(f"    cost  {c['how_many_swipes']}")
            print(f"    buys  {c['buys']}")
            if "critical" in c:
                print(f"    !!    {c['critical']}")
        print(f"\n  TOTAL {out['total_swipes']} swipes")
        print(f"  {out['device_constraint']}")
        print("\nanalysis:")
        for a in out["analysis_that_follows"]:
            print(f"  - {a}")
    return out


# ---------------------------------------------------------------------------
# 8b. can this run in the verify budget?
# ---------------------------------------------------------------------------
def budget(verbose=True):
    """What a coarser angle grid costs, in accuracy and in time.

    The verify budget is about a second and the search is dominated by the
    angle sweep: at 1 deg over +-25 deg it is 51 evaluations, each of which is
    one warp of the probe plus three forward and six inverse real FFTs over the
    padded canvas.  Halving the grid halves the search.

    This measures both sides of that trade on the real held-out probes rather
    than arguing about it.  The Python timings are not the C timings and should
    not be quoted as such -- what transfers is the RATIO between grids and the
    accuracy column.

    THE COARSE-TO-FINE ROUTE IS MEASURED NOT TO WORK, AND THAT IS THE PROBLEM.

    The angle grid cannot be cut far: 2.5 deg already loses two of four
    held-out probes and puts the answer 15 px out.  The standard remaining
    speedup is multiresolution -- search a decimated pair, refine at full
    resolution -- and it quarters the cost per halving.  Measured on the
    170350 probe against a 70.0 mm^2 map (scratchpad test_pyramid.py):

        k   canvas      sec   speedup    NCC    theta   err vs full-res
        1   108x349    5.32       1.0x  0.637    7.00     0.00 px
        2    54x174    1.47       3.6x  0.603   24.00   123.99 px
        3    36x116    0.65       8.1x  0.749  -13.50    91.37 px

    The speedup is real and the answer is destroyed -- 6.3 mm and 4.6 mm out,
    with theta railing.  The obvious explanation is that `ridge_dog` fixes
    sigma at 1.0/3.0 for a 9-10 px ridge pitch, so at k=3 the decimated pitch
    of ~3 px falls entirely below the pass band.  That was tested and is NOT
    the explanation: rescaling the DoG sigmas by 1/k and MASK_ERODE with them
    leaves the errors at 159.5 px (k=2) and 94.1 px (k=3).

    The real reason is that the true peak's margin is thin to begin with --
    full-resolution NCC 0.637 at PSR 14.2 -- so any loss of ridge amplitude
    lets a sidelobe win outright.  Note k=3 scores NCC 0.749, HIGHER than the
    correct full-resolution answer, while being 91 px wrong: the same
    score-rises-as-answer-goes-wrong pathology as sections 3 and 3b, from a
    third direction, and nothing in the score reports the failure.

    So the 1 s budget cannot be bought with resolution, and it cannot be bought
    with the angle grid.  What is left is the C port itself (the ratio, not the
    seconds), a cheaper pose prior (an orientation-field estimate to shrink the
    angle range rather than coarsen it), and early abort.  None is measured.
    Treat "under 1 s" as an OPEN ENGINEERING RISK, not a solved problem: the
    prototype is 2.3-4.8 s and the only two standard levers are exhausted.
    """
    strips = load_real_strips()
    by = {s.name: s for s in strips}
    t_all, _ = _build_map(list(strips))
    names = [s.name for s in t_all.strips]
    rows = []
    for held in names:
        tmpl, _ = _build_map([by[n] for n in names if n != held])
        probe = by[held]
        ref = match_canvas(tmpl, probe, angles=_angles(1.0), refine=True)
        for step, refine in ((1.0, True), (1.0, False), (2.0, True),
                             (2.5, True), (5.0, True)):
            t0 = time.time()
            r = match_canvas(tmpl, probe, angles=_angles(step), refine=refine)
            dt = time.time() - t0
            if r is None:
                continue
            ok, why, _ = decide(r, None, ROUTE_CANVAS)
            rows.append({
                "held": held, "rot_step_deg": step, "refine": bool(refine),
                "n_angles": int(len(_angles(step))),
                "seconds": round(dt, 2), "ncc": r["ncc"],
                "explained": r["explained"], "theta": r["theta"],
                "accepted": bool(ok),
                "px_from_1deg_answer": round(
                    pose_distance(r["pose_canvas"], ref["pose_canvas"],
                                  probe.img.shape), 2),
            })
    out = {"rows": rows, "by_grid": {}}
    for step in sorted({r["rot_step_deg"] for r in rows}):
        for refine in (True, False):
            sel = [r for r in rows if r["rot_step_deg"] == step
                   and r["refine"] == refine]
            if not sel:
                continue
            out["by_grid"][f"{step}deg{'+refine' if refine else ''}"] = {
                "n_angles": sel[0]["n_angles"],
                "seconds_median": round(float(np.median(
                    [r["seconds"] for r in sel])), 2),
                "n_accepted": sum(1 for r in sel if r["accepted"]),
                "n": len(sel),
                "worst_px_from_1deg": round(max(
                    r["px_from_1deg_answer"] for r in sel), 2),
                "ncc_min": round(min(r["ncc"] for r in sel), 4),
            }
    if verbose:
        print(f"  {'grid':<16}{'angles':>7}{'sec':>7}{'accepted':>10}"
              f"{'NCC min':>9}{'worst px vs 1deg':>18}")
        for k, v in out["by_grid"].items():
            print(f"  {k:<16}{v['n_angles']:>7}{v['seconds_median']:>7.2f}"
                  f"{v['n_accepted']:>6}/{v['n']:<3}{v['ncc_min']:>9.3f}"
                  f"{v['worst_px_from_1deg']:>18.2f}")
        print("  Python timings, single-threaded numpy FFT.  The C port has to "
              "hand-roll a\n  radix-2 FFT over a ~200 x 740 padded canvas: 9 "
              "transforms per angle, so a\n  1 deg grid is about 5 Gflop and a "
              "2.5 deg grid about 2 Gflop.  The ratio is\n  what transfers, "
              "not the seconds.")
    return out


# ---------------------------------------------------------------------------
# 9. selftest: checks that can fail
# ---------------------------------------------------------------------------
def _synthetic_template(rng, ch=110, cw=360, n_strips=2):
    """A fake enrolled map with known geometry.

    The canvas is a ridge-like random field at the measured 9.5 px pitch.  Two
    'enrolled strips' are cut from it at known poses, so every composition the
    matcher does can be checked against arithmetic rather than against itself.
    """
    field = SM._rand_field((ch, cw), rng, pitch=9.5)
    cov = np.zeros((ch, cw), bool)
    cov[8:ch - 8, 8:cw - 8] = True
    strips = []
    poses = [translation(6.0, 10.0), translation(20.0, 170.0)]
    for i in range(n_strips):
        P = poses[i]
        sh, sw = 70, 170
        # strip image = the canvas sampled through P
        img, val = warp(np.where(cov, field, 0.0), np.linalg.inv(P), (sh, sw),
                        border=0)
        m = val & (img != 0.0)
        m = erode(m, 3)
        s = Strip(f"synthetic#{i}", np.zeros((0, FRAME_H, FRAME_W)),
                  np.zeros(0), np.zeros(0), np.where(m, img, 0.0), m, {})
        s.pose = P
        strips.append(s)
    return Template(field, cov, strips, (0, 0)), field, cov


def _cut_probe(canvas, cov, pose, shape=(66, 200), noise=0.0, rng=None):
    """Cut a probe out of a canvas at a known pose (probe -> canvas).

    `noise` matters: a probe cut cleanly out of the canvas correlates at NCC
    1.000, which checks the geometry but is a much easier problem than any real
    swipe.  The recovery checks below run with noise on, so the search has to
    find the answer through a correlation of about 0.7 -- the same value the
    real held-out probes reach.
    """
    img, val = warp(np.where(cov, canvas, 0.0), np.linalg.inv(pose), shape,
                    border=0)
    if noise > 0:
        rng = rng or np.random.default_rng(1)
        img = img + rng.normal(0.0, noise * float(np.std(img[val])), img.shape)
    m = erode(val & (img != 0.0), 3)
    return _fake_strip("probe", np.where(m, img, 0.0), m)


def selftest(verbose=True):
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'ok ' if ok else 'FAIL'}] {name}"
                  + (f"  -- {detail}" if detail else ""))

    rng = np.random.default_rng(20260814)

    # 1. `explained` arithmetic ------------------------------------------
    # Non-vacuous: a swapped numerator and denominator, or a missing NCC
    # factor, all give a different number here.
    cov = np.zeros((10, 10), bool)
    cov[:, :5] = True                      # 50 px of probe
    res = {"overlap_px": 20, "ncc": 0.6}
    exp, frac = _explained(res, cov)
    check("explained = ncc * overlap/probe_area",
          abs(exp - 0.24) < 1e-9 and abs(frac - 0.4) < 1e-9,
          f"got explained {exp}, frac {frac}, expected 0.24 / 0.4")

    # 2. synthetic recovery, translation only ----------------------------
    tmpl, field, ccov = _synthetic_template(rng)
    Ptrue = translation(14.0, 60.0)
    probe = _cut_probe(field, ccov, Ptrue, noise=1.0, rng=rng)
    r = match_canvas(tmpl, probe)
    e = (pose_distance(r["pose_canvas"], Ptrue, probe.img.shape)
         if r else float("inf"))
    check("translation-only probe recovered to < 1 px", e < 1.0,
          f"corner error {e:.3f} px, NCC {r['ncc']:.3f}" if r else "no result")

    # 3. synthetic recovery, rotation ------------------------------------
    # Rotation cannot be checked on the real data at all: the two halves of a
    # swipe never rotate relative to each other, so every rotation claim in
    # this project rests on a check like this one.
    th = 9.0
    Prot = translation(18.0, 70.0) @ similarity(th, cy=33.0, cx=100.0)
    probe_r = _cut_probe(field, ccov, Prot, noise=1.0, rng=rng)
    rr = match_canvas(tmpl, probe_r)
    er = (pose_distance(rr["pose_canvas"], Prot, probe_r.img.shape)
          if rr else float("inf"))
    check(f"probe rotated {th:+.0f} deg recovered to < 2 px", er < 2.0,
          f"corner error {er:.3f} px, recovered theta {rr['theta']:+.2f} "
          f"(true {th:+.1f})" if rr else "no result")

    # 4. synthetic recovery, along-axis scale ----------------------------
    sxt = 0.96
    Psc = translation(16.0, 64.0) @ similarity(0.0, sx=sxt, cy=33.0, cx=100.0)
    probe_s = _cut_probe(field, ccov, Psc, noise=1.0, rng=rng)
    rs = match_canvas(tmpl, probe_s)
    # The expected answer is sxt, not 1/sxt: the probe was cut through
    # inv(Psc), so the transform that puts the probe back on the canvas is Psc
    # itself.  Asserting only one of the two is the point -- an "either sign is
    # fine" assertion would pass a matcher whose scale convention was inverted,
    # and inverted conventions are exactly the bug swipe_assemble found in its
    # own sub-pixel shift.
    check(f"probe scaled {sxt} along its axis: scale recovered within 0.025",
          rs is not None and abs(rs["sx"] - sxt) < 0.025,
          f"recovered sx {rs['sx'] if rs else None}, expected {sxt} "
          f"(the inverted convention would give {1.0 / sxt:.3f})")

    # 5. nulls rejected --------------------------------------------------
    n_acc = 0
    ncc_null = []
    for _ in range(6):
        pi = phase_randomise(probe.img, probe.cov, rng)
        o, _x = verify(tmpl, _fake_strip("null", pi, probe.cov),
                       route=ROUTE_CANVAS, verbose=False)
        ncc_null.append(o.get("ncc", 0.0))
        n_acc += bool(o["accepted"])
    check("phase-randomised probes rejected against the synthetic map",
          n_acc == 0,
          f"{n_acc}/6 accepted, null NCC max {max(ncc_null):.3f} against a "
          f"genuine {r['ncc']:.3f}")

    # 6. every acceptance condition is reachable AND fires the right way --
    # The trap this avoids is a comparison written backwards, which a test that
    # only ever feeds it good input cannot see.
    base = {"overlap_mm2": 30.0, "ncc": 0.80, "psr": 20.0, "explained": 0.45,
            "overlap_frac": 0.6, "theta": 0.0, "sx": 1.0, "split_support": 0.9,
            "split_excess": 0.0, "split_px": 1.0, "at_rail": False,
            "scale_at_rail": False}
    ok0, why0, _ = decide(dict(base))
    check("a clean result is ACCEPTED", ok0, why0)
    trips = [
        ("overlap", {"overlap_mm2": MIN_VERIFY_OVERLAP_MM2 - 0.1}, "overlap"),
        ("explained", {"explained": MIN_EXPLAINED - 0.01}, "explained"),
        ("ncc", {"ncc": ACCEPT_NCC_CANVAS - 0.01}, "NCC"),
        ("psr", {"psr": ACCEPT_PSR_VERIFY - 0.1}, "PSR"),
        ("angle rail", {"at_rail": True}, "rail"),
        ("scale rail", {"scale_at_rail": True}, "rail"),
        ("split support", {"split_support": ACCEPT_SPLIT_SUPPORT - 0.01},
         "support"),
        ("split excess", {"split_excess": ACCEPT_SPLIT_EXCESS + 0.01,
                          "split_px": 50.0}, "prefers"),
    ]
    bad = []
    for nm, patch, token in trips:
        d = dict(base)
        d.update(patch)
        okx, whyx, _ = decide(d)
        if okx or token not in whyx:
            bad.append(f"{nm}: accepted={okx} reason={whyx!r}")
    check("each acceptance condition rejects when tripped, and says so",
          not bad, "; ".join(bad) if bad else
          f"{len(trips)} conditions each fire independently")

    # 7. the null-maximum gate uses the template's own calibration --------
    t2 = Template(tmpl.canvas, tmpl.canvas_cov, tmpl.strips, tmpl.origin,
                  null={"n": 5, "ncc_mean": 0.30, "ncc_sd": 0.03,
                        "ncc_max": 0.90})
    okn, whyn, extra = decide(dict(base), t2)
    check("a result below the template's own null maximum is rejected",
          (not okn) and "null maximum" in whyn and
          extra.get("margin_over_null_max") is not None,
          f"NCC 0.80 against a calibrated null max of 0.90 -> {whyn!r}")

    # 8. template round-trip ---------------------------------------------
    tmpl.null = {"n": 3, "ncc_mean": 0.31, "ncc_sd": 0.02, "ncc_max": 0.36}
    tmpl.meta = {"hello": "world"}
    p = Path("/tmp") / f"swipe_verify_selftest_{np.random.randint(1 << 30)}.npz"
    try:
        tmpl.save(p)
        back = Template.load(p)
        same = (np.allclose(back.canvas, tmpl.canvas, atol=1e-3) and
                np.array_equal(back.canvas_cov, tmpl.canvas_cov) and
                len(back.strips) == len(tmpl.strips) and
                all(np.allclose(a.pose, b.pose)
                    for a, b in zip(back.strips, tmpl.strips)) and
                all(a.name == b.name
                    for a, b in zip(back.strips, tmpl.strips)) and
                back.null == tmpl.null and back.meta == tmpl.meta)
        r_back = match_canvas(back, probe)
        agree = (r_back is not None and abs(r_back["ncc"] - r["ncc"]) < 1e-3 and
                 pose_distance(r_back["pose_canvas"], r["pose_canvas"],
                               probe.img.shape) < 1e-6)
        check("template survives save/load and matches identically",
              same and agree,
              f"fields {'ok' if same else 'DIFFER'}, "
              f"match {'identical' if agree else 'DIFFERS'}")
    finally:
        if p.exists():
            p.unlink()

    # 9. gallery pose composition against hand arithmetic -----------------
    # canvas_pose(s) @ rel must place the probe where the canvas route does.
    g = match_gallery(tmpl, probe)
    dg = (pose_distance(g["pose_canvas"], Ptrue, probe.img.shape)
          if g else float("inf"))
    check("gallery route composes strip pose and relative pose correctly",
          dg < 1.5,
          f"corner error {dg:.3f} px against the known probe pose, via "
          f"{g['via'] if g else None}")

    # 10. hybrid picks the strip the probe actually lands on ---------------
    # The probe was cut at (14, 60); strip 0 sits at (6, 10) and strip 1 at
    # (20, 170), so strip 0 is the one it overlaps.
    h = match_hybrid(tmpl, probe)
    check("hybrid re-scores against the overlapping strip, not the far one",
          h is not None and h.get("rescored_via") == "synthetic#0",
          f"chose {h.get('rescored_via') if h else None}; canvas NCC "
          f"{h.get('canvas_ncc') if h else None} -> {h.get('ncc') if h else None}")

    # 11. explained falls as the map is cropped, on data with known truth --
    # This is the claim MIN_EXPLAINED rests on, checked where the truth is
    # exact rather than only on the four real strips.
    cols = np.where(tmpl.canvas_cov.any(0))[0]
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    seq = []
    for frac in (1.0, 0.8, 0.6, 0.45):
        cc = tmpl.canvas_cov.copy()
        cc[:, c0 + int((c1 - c0) * frac):] = False
        sub = Template(tmpl.canvas, cc, tmpl.strips, tmpl.origin)
        rr2 = match_canvas(sub, probe)
        if rr2:
            seq.append((frac, rr2["ncc"], rr2["explained"],
                        pose_distance(rr2["pose_canvas"], Ptrue,
                                      probe.img.shape)))
    mono = all(seq[i][2] >= seq[i + 1][2] - 1e-9 for i in range(len(seq) - 1))
    check("explained falls monotonically as the map is cropped",
          len(seq) >= 3 and mono,
          "  ".join(f"frac {f}: ncc {n:.3f} expl {e:.3f} err {d:.1f}px"
                    for f, n, e, d in seq))

    # 11b. ...but does NOT fall when the PROBE is shortened ----------------
    # The complement of check 11 and the claim section 3b rests on.  Cropping
    # the map shrinks the overlap and leaves the probe area fixed, so the ratio
    # falls.  Shortening a probe that still lands inside the map shrinks both
    # by nearly the same factor, so the ratio does NOT fall -- which is why
    # `explained` cannot be the test that catches a short swipe, and the hard
    # mm^2 floor has to.  Asserted here so that a future change which "fixes"
    # `explained` to respond to probe length has to break this test on purpose.
    # Fractions are kept mild on purpose: crop the synthetic probe much past
    # 0.7 and the overlap falls under MIN_OVERLAP_MAP, the matcher returns
    # None, and the check would silently run on too few points.
    pseq = []
    for frac in (1.0, 0.9, 0.8, 0.7):
        w = int(probe.img.shape[1] * frac)
        pc2 = probe.cov.copy()
        pc2[:, w:] = False
        if pc2.sum() < 400:
            continue
        rr3 = match_canvas(tmpl, _fake_strip("short", probe.img, pc2))
        if rr3:
            pseq.append((frac, rr3["overlap_mm2"], rr3["overlap_frac"],
                         rr3["ncc"], rr3["explained"]))
    # absolute overlap must really fall, or the check is vacuous
    ov_falls = (len(pseq) >= 3
                and pseq[-1][1] < pseq[0][1] * 0.85)
    frac_holds = all(p[2] > 0.80 for p in pseq)           # but the FRACTION holds up
    expl_holds = (len(pseq) >= 3
                  and min(p[4] for p in pseq) > 0.85 * pseq[0][4])
    check("explained does NOT fall when the probe is shortened "
          "(only the mm^2 floor catches that)",
          bool(ov_falls and frac_holds and expl_holds),
          "  ".join(f"frac {f}: ov {o:.1f}mm2 ovfrac {vf:.2f} "
                    f"ncc {n:.3f} expl {e:.3f}" for f, o, vf, n, e in pseq))

    # 12. erode actually erodes (the vacuous-test trap) --------------------
    m = np.zeros((30, 30), bool)
    m[10:20, 10:20] = True
    e2 = erode(m, 2)
    check("erode(r=2) removes exactly a 2 px rim from a 10x10 block",
          int(e2.sum()) == 36 and bool(e2[12:18, 12:18].all()),
          f"kept {int(e2.sum())} px, expected 36 (a 6x6 core)")

    n_fail = sum(1 for c in checks if not c["pass"])
    if verbose:
        print(f"selftest: {len(checks) - n_fail}/{len(checks)} checks passed")
    return {"checks": checks, "n": len(checks), "n_failed": n_fail}


# ---------------------------------------------------------------------------
# 10. output and CLI
# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=_jsonable))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description="verify one swipe against an enrolled finger map")
    ap.add_argument("--enrol", nargs="+", metavar="DIR",
                    help="capture directories to enrol")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--verify", metavar="DIR",
                    help="capture directory holding the probe swipe")
    ap.add_argument("--route", default=ROUTE_CANVAS,
                    choices=(ROUTE_CANVAS, ROUTE_GALLERY, ROUTE_HYBRID))
    ap.add_argument("--loo", action="store_true",
                    help="leave-one-out validation on this machine's captures")
    ap.add_argument("--split-truth", action="store_true",
                    help="ground-truth validation built from one swipe")
    ap.add_argument("--overlap-sweep", action="store_true",
                    help="score and pose accuracy vs overlap (crops the MAP)")
    ap.add_argument("--probe-sweep", action="store_true",
                    help="score and pose accuracy vs probe length (a SHORT swipe)")
    ap.add_argument("--press-check", action="store_true",
                    help="does this matcher see finger identity at all?")
    ap.add_argument("--coverage-model", action="store_true",
                    help="how many enrolment swipes map a finger")
    ap.add_argument("--protocol", action="store_true",
                    help="the collection needed to make this a security claim")
    ap.add_argument("--budget", action="store_true",
                    help="what a coarser angle grid costs in accuracy and time")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="every validation, in order")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    did = False
    report = {}

    if a.selftest or a.all:
        did = True
        print("=" * 74)
        print("SELFTEST")
        report["selftest"] = selftest()
        if report["selftest"]["n_failed"]:
            print("selftest FAILED -- nothing below should be believed")

    if a.enrol:
        did = True
        print("=" * 74)
        print("ENROL")
        t, rej, trials = enrol([Path(p) for p in a.enrol])
        t.save(a.template)
        print(f"template -> {a.template}  ({t.coverage_mm2:.1f} mm^2, "
              f"{len(t.strips)} strips, {len(rej)} rejected)")
        report["enrol"] = {"meta": t.meta, "null": {
            k: v for k, v in t.null.items() if k != "rows"}}

    if a.verify:
        did = True
        print("=" * 74)
        print("VERIFY")
        t = Template.load(a.template)
        probe, all_names = probe_from_capture(a.verify, verbose=True)
        if probe is None:
            print("no usable swipe in the probe capture")
        else:
            if len(all_names) > 1:
                print(f"  capture holds {len(all_names)} swipes {all_names}; "
                      f"using {probe.name}.  A real verify gets one.")
            o, _ = verify(t, probe, route=a.route)
            report["verify"] = o

    if a.loo or a.all:
        did = True
        print("=" * 74)
        print("LEAVE-ONE-OUT")
        report["leave_one_out"] = leave_one_out()

    if a.split_truth or a.all:
        did = True
        print("=" * 74)
        print("GROUND TRUTH FROM ONE SWIPE")
        report["split_truth"] = split_truth()

    if a.overlap_sweep or a.all:
        did = True
        print("=" * 74)
        print("SCORE VS OVERLAP")
        report["overlap_sweep"] = overlap_sweep()

    if a.probe_sweep or a.all:
        did = True
        print("=" * 74)
        print("SCORE VS PROBE LENGTH (A SHORT SWIPE)")
        report["probe_sweep"] = probe_sweep()

    if a.budget or a.all:
        did = True
        print("=" * 74)
        print("VERIFY BUDGET")
        report["budget"] = budget()

    if a.press_check or a.all:
        did = True
        print("=" * 74)
        print("DOES THIS MATCHER SEE FINGER IDENTITY?")
        report["press_check"] = press_check()

    if a.coverage_model or a.all:
        did = True
        print("=" * 74)
        print("HOW MANY ENROLMENT SWIPES")
        report["coverage_model"] = coverage_model()

    if a.protocol or a.all:
        did = True
        print("=" * 74)
        print("COLLECTION PROTOCOL")
        report["protocol"] = protocol()

    if not did:
        ap.print_help()
        return 1

    if report:
        _dump(report, out / "swipe_verify_report.json")
    if a.json:
        print(json.dumps(report, indent=1, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
