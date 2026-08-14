# ELAN 04f3:0c6e fingerprint sensor on Linux

Notes, packaging and diagnostics for the **ELAN `04f3:0c6e`** fingerprint sensor —
the power-button reader in the **ASUS ROG Flow X13 (GV301)** and several ASUS
Zenbook / Vivobook models.

Short version: **this sensor does not work reliably with upstream libfprint, and
has not for five years.** This repo collects what is actually true about it, so
the next person does not repeat the same dead ends.

> **Status:** research + packaging. The `elanpress` route below is the most
> promising known approach but is **not yet confirmed working** by this repo's
> author. Findings marked *unverified* are exactly that. PRs with results from
> other machines are very welcome — especially the model/firmware table at the
> bottom.

---

## Quick start: `elan-fp`

[`tools/elan-fp`](tools/elan-fp) drives the sensor and recovers it automatically.
It exists because this device fails in several unrelated ways, and one of them
(an unbounded poll loop in the `elanpress` driver) **hangs forever instead of
erroring**. Every operation here is timeout-bounded, and failures escalate
through a recovery ladder rather than blocking or giving up.

```sh
install -Dm755 tools/elan-fp ~/.local/bin/elan-fp

elan-fp status     # sensor, driver, power policy, enrollment, PAM state
elan-fp enroll     # unwedges first, so a bad activation isn't wasted
elan-fp test       # 10 verifications, reports a match rate
elan-fp doctor     # full diagnostic dump, safe to attach to a bug report
```

The recovery ladder, applied automatically between retries:

| Level | Action | Clears |
|---|---|---|
| 1 | stop `fprintd`, kill strays | stale device claims |
| 2 | USB **selective suspend** + resume | some firmware state — clocks actually stop |
| 3 | `authorized` 0→1 | kernel-side enumeration state only |

Level 2 is the interesting one: it's the closest thing to a power cycle you can
trigger from software. **A common udev rule pinning `power/control=on` (to stop
autosuspend killing the sensor mid-use) disables it** — `elan-fp` temporarily
overrides that and restores your setting afterwards. `elan-fp status` warns when
that rule is blocking recovery.

Note that `authorized` toggling and `g_usb_device_reset()` are **logical**
operations: VBUS is never interrupted and the sensor's MCU keeps running through
both. On units exhibiting the once-per-power-up wedge, neither is sufficient.

---

## The hardware

| | |
|---|---|
| USB ID | `04f3:0c6e` (`ELAN`, `ELAN:Fingerprint`) |
| Sensor | 150 × 52 px, 14-bit ADC |
| Firmware seen | `0x0161` (`bcdDevice 1.61`) |
| Endpoints | `0x81 IN`, `0x01 OUT`, `0x82 IN`, `0x83 IN`, `0x03 OUT` — **no `0x84`** |
| Laptop tested | ASUS ROG Flow X13 `GV301QH`, CachyOS (Arch), KDE |

The absence of endpoint `0x84` is diagnostic: it rules out the `elanmoc`
match-on-chip driver, which requires it.

---

## Key findings

### 1. It is an image sensor, not match-on-chip — do not patch `elanmoc`

`0x0c6e` is handled by libfprint's image-based **`elan`** driver, and has been in
its `id_table` since 2021. It is *not* an `elanmoc` device.

This matters because a widely-linked README
([`alejandrok5/asus-config`](https://github.com/alejandrok5/asus-config)) states
that `0c6e` is an elanmoc match-on-chip device "supported by current libfprint".
**That is wrong on every point**, and appears to be the origin of a lot of wasted
effort — including this repo author's first attempt. If you are about to add
`0x0c6e` to an `elanmoc` `id_table`, stop.

### 2. The PID was added upstream by someone who later retracted it

[MR !307](https://gitlab.freedesktop.org/libfprint/libfprint/-/merge_requests/307)
(merged 2021, commit `9ecd6236`) added `0x0c6e` to `elan.h` on the strength of
`examples/img-capture` producing *an image*. In
[issue #401](https://gitlab.freedesktop.org/libfprint/libfprint/-/issues/401)
the same submitter later wrote:

> "I'm the one who added the device ID to libfprint at all, and same, it never
> verifies. (flow X13 as well, 0c6e)"

A maintainer proposed reverting the PID. It was never reverted. **The entry in
upstream's table does not mean the device works.**

### 3. Enrollment succeeding means nothing — verification is the bar

Multiple people get clean enrollments. Nobody has demonstrated reliable
*verification* on a Flow X13. If you are testing, the only meaningful gate is
something like:

```fish
for i in (seq 10); fprintd-verify; end
```

Ten out of ten, across a suspend/resume cycle and a reboot. Anything less is
noise.

### 4. `0xaf` means "finger removed / sensor busy" — and upstream already fixed it (for a sibling PID)

If capture dies instantly with `FP_DEVICE_ERROR_PROTO` in `CAPTURE_READ_DATA`
(SSM "state 2"), the sensor is answering `pre_scan_cmd` with `0xaf` instead of
`0x55`. Upstream commit **`4610f22`** (2026-04-16) documents this for the sibling
`04f3:0c58`:

> The Elan 04f3:0c58 sensor returns 0xaf during capture which indicates finger
> removed or sensor busy. Also 0x00 can occur meaning not ready. These responses
> should trigger a retry instead of failing.

```c
else if (self->dev_type == ELAN_0C58 && self->last_read &&
         (self->last_read[0] == 0x00 || self->last_read[0] == 0xaf))
  fpi_ssm_jump_to_state_delayed (ssm, CAPTURE_WAIT_FINGER, 10);
```

Two traps when porting this to `0x0c6e`:

- It tests `dev_type == ELAN_0C58`, **exact equality, not a bitmask**. Adding a
  new quirk bit does not inherit the behaviour; you must edit the condition.
- **Bit collision:** upstream now defines `ELAN_0C58 (1 << 3)`. If you had
  previously defined your own `ELAN_0C6E (1 << 3)`, rebasing silently makes each
  device match the other's quirks. Use `(1 << 4)`.

Retry **indefinitely with a delay**, not a fixed small count. A capped retry loop
burns through in milliseconds and the user never gets a chance to touch the
sensor.

### 5. The sensor wedges: it works once per power-up, then answers `0xaf` forever

Independently observed here and diagnosed in the `fix-rog-flow-x13` branch
(see below): `pre_scan` returns a valid status **exactly once per power-up**.
After that it answers `0xaf` immediately and permanently — surviving device
close/open, process restart, and `fprintd` restart.

A USB-level reset via `/sys/bus/usb/devices/<N>/authorized` was **not** sufficient
in testing here; a full power cycle was needed. *(Unverified: whether
suspend/resume is enough.)*

**Practical consequence:** do not diagnose image quality on a wedged device. A
degraded capture here measured a dynamic-range span of 35 versus ~60 for a good
one from the same build and finger, with visibly noisier ridges. Chasing that as
an image-processing bug wastes hours.

### 6. `libfprint 1.94.10` aborts the process on a flat frame

`elan_process_frame_linear` in 1.94.10 contains `g_assert (max != min)`. A flat or
blank frame **aborts `fprintd` outright** rather than returning an error — which
surfaces to the client as `enroll-disconnected`. Upstream commit `db316a5`
replaced it with `memset` + `g_return_if_reached`. If you see
`enroll-disconnected`, this is a likely mechanism; get onto ≥ 1.94.100.

### 7. Small sensors and NBIS don't mix — which motivates a different matcher

libfprint's own `elan.c` header says it plainly:

> "The algorithm which libfprint uses to match fingerprints doesn't like small
> images like the ones these drivers produce. There's just not enough minutiae
> […] unless another matching algo is found/implemented, these readers will not
> work as good with libfprint as they do with vendor drivers."

At 150 × 52 this is the core problem, and it is why threshold tuning is a weak
lever (see dead ends).

---

## The most promising approach: the `elanpress` driver

An out-of-tree driver that treats `0c6e` as a **press** sensor and replaces
minutiae matching entirely with **cross-correlation**. It registers as an
`FpDevice` rather than an `FpImageDevice`, so NBIS/bozorth3 are bypassed.

> The upstream branch correlates **raw pixel intensities** (threshold 0.55,
> ±60 px x, ±20 px y, ±12°), which is measurably worse than chance — see the
> warning below. The local build in this repo replaces that scoring path; see
> [What shipped: the LCN matcher](#what-shipped-the-lcn-matcher-august-2026).

Use **[`Quoteme/libfprint` branch `elanpress`](https://github.com/Quoteme/libfprint/tree/elanpress)**
(commit `9e6ea5a`) — *not* the original `filip-rs` HEAD. Quoteme's branch merges
`fix-rog-flow-x13`, which removes the pre-scan finger-presence check that causes
the once-per-power-up wedge, inferring touch from image content instead. That
makes the whole `0x55`/`0xaf`/`0xcc` firmware divergence moot by construction —
which matters, because reporters have seen three different status bytes.

```fish
git clone -b elanpress https://github.com/Quoteme/libfprint.git
cd elanpress && makepkg -si     # PKGBUILD in this repo
```

**Honest caveats.** It is a single-commit, AI-assisted branch with no upstream MR
and no activity since July 2026. Two people independently had to patch it to work.
It is the best available option, not a maintained one.

**After any driver switch you must re-enroll.** `fprintd` stores prints under
`/var/lib/fprint/<user>/<driver>/…`, so prints saved under `elan/` are invisible
to `elanpress`.

---

## Dead ends — confirmed, don't re-tread

| Dead end | Why |
|---|---|
| `elanmoc` driver | Wrong family. No EP `0x84`; device streams image frames. |
| `elanmoc2` / geodic / Depau / ITx-prash forks | None list `0c6e`; geodic's `elanmoc2` branch is deleted, so AUR `libfprint-elanmoc2-newdrvs-git` is unbuildable. |
| `libfprint-2-tod1-elan` (TOD blob) | Binary's `FpIdEntry` array has exactly two entries: `0c42`, `0c4b`. Structurally closed. |
| Lowering `bz3_threshold` | Stock is **24**. Small changes don't move the needle when the image yields too few minutiae. |
| USB autosuspend | Ruled out; no `dmesg` disconnects. It's protocol desync, not a link problem. |
| `"Device has no storage"` in logs | Normal for any image driver. Red herring. |
| Waiting for upstream | Several issues have **zero comments**. Treat the tracker as a symptom archive. |

---

## ⚠️ PAM: how to not lock yourself out

This is the step that ends most attempts, including this one in May 2026.

`pam_fprintd` 1.94.5 defaults to 3 tries × 30 s. On a sensor that **enrolls but
never matches**, every `sudo` blocks for up to **90 seconds** before offering a
password prompt. Enrolled-but-never-matching is worse than not enrolling at all.

1. **Do not touch `/etc/pam.d/system-auth` or `/etc/pam.d/sudo`. Ever.** On Arch,
   `system-auth` feeds sudo, polkit, SDDM, every TTY, and cron. One bad line
   there takes out all of them at once.
2. **On KDE you probably need to edit nothing.** `/usr/lib/pam.d/kde-fingerprint`
   already ships `-auth required pam_fprintd.so`. Once verification works the lock
   screen picks it up by itself, and it's a separate PAM service from password
   auth, so failure there cannot lock you out.
3. If you do want TTY/SDDM login, the file to touch is
   `/etc/pam.d/system-local-login`, and always cap the timeout:
   ```
   -auth [success=1 default=ignore] pam_fprintd.so max-tries=1 timeout=5
   ```
4. **Keep a root shell open on another TTY** for the whole test, and don't close
   it until you've re-authenticated in a third session.
5. Know your escape hatch before you start (GRUB → `e` → append `rw init=/bin/bash`).
6. **Pin the package.** A routine `pacman -Syu` will happily replace a
   `conflicts=(libfprint)` custom package with the stock one. Add
   `IgnorePkg = libfprint` to `/etc/pacman.conf`.

---

## Diagnostics

See [`tools/`](tools/). Two things that will save you time:

- **`fprintd.service` sets `PrivateTmp=yes`.** Any debug images your patched
  driver writes to `/tmp` land in a private namespace you can't see, and are
  **deleted when the service stops**. Run `fprintd` manually instead:
  `sudo G_MESSAGES_DEBUG=all /usr/lib/fprintd -t`
- **Killing `fprintd-verify` mid-verify wedges the device**, and the next
  `EnrollStart` fails with a misleading `Timeout was reached`. Clear it with
  `systemctl stop fprintd`.

For image quality, bypass `fprintd` entirely — `libfprint`'s own
`examples/img-capture` talks to the driver directly, with no D-Bus, polkit,
enrollment or identify in the way.

> **A note on sample images:** this repo deliberately contains **no fingerprint
> captures**. They are biometric data. `.gitignore` blocks `*.pgm` — please keep
> it that way in PRs, and post statistics (dynamic range, span, stdev) rather
> than images.

---

## Upstream issues

- [#401](https://gitlab.freedesktop.org/libfprint/libfprint/-/issues/401) — the main thread; includes the submitter's retraction
- [#402](https://gitlab.freedesktop.org/libfprint/libfprint/-/issues/402) — the "swipe then lift" physical workaround (weak evidence; no PID stated)
- [#732](https://gitlab.freedesktop.org/libfprint/libfprint/-/issues/732), [#759](https://gitlab.freedesktop.org/libfprint/libfprint/-/issues/759), [#808](https://gitlab.freedesktop.org/libfprint/libfprint/-/issues/808), [#814](https://gitlab.freedesktop.org/libfprint/libfprint/-/issues/814) — further reports, several with no replies

Note: `gitlab.freedesktop.org` sits behind an Anubis proof-of-work wall that
blocks browser-like user agents. Plain `curl` and `git clone` over HTTPS work
fine.

---

## Reports from other machines

If you have `04f3:0c6e`, please open a PR adding a row.

| Laptop | Firmware | Driver tried | Enroll | Verify (n/10) | Notes |
|---|---|---|---|---|---|
| ASUS ROG Flow X13 GV301QH | `0x0161` | stock `elan` 1.94.100 | ✗ | — | proto error in capture |
| ASUS ROG Flow X13 GV301QH | `0x0161` | patched `elan` (press averaging) | ✓ | 0/5 | captures show real ridges; never matches |
| ASUS ROG Flow X13 GV301QH | `0x0161` | `elanpress` (template from wedged device) | ✓ 8 stages | 4/9 | mean NCC 0.581, sd 0.159 |
| ASUS ROG Flow X13 GV301QH | `0x0161` | offline: LCN + rotation + 30 templates | — | d′ 1.54, EER 20% | best measured; see findings |
| **ASUS ROG Flow X13 GV301QH** | **`0x0161`** | **`elanpress` (template from healthy device)** | **✓** | **8/10** | **stock 0.55 threshold, no hangs** |

### ⚠️ `elanpress`'s matcher does not discriminate — do not use it for auth

> **This section describes the *upstream* raw-intensity matcher.** It has since
> been replaced in this repo's build — see
> [What shipped: the LCN matcher](#what-shipped-the-lcn-matcher-august-2026).
> The replacement is a real improvement and fixes the inversion described here,
> but it is still not authentication-grade and still must not be wired to PAM.

Measured on a GV301QH (firmware `0x0161`), presenting a **different finger**
against an enrolled template:

| | NCC | Outcome |
|---|---|---|
| genuine | 0.364 | no-match |
| genuine | 0.364 | no-match |
| **impostor** | **0.578** | **ACCEPTED at the stock 0.55 threshold** |
| impostor | 0.536 | no-match |
| impostor | 0.459 | no-match |
| impostor | 0.338 | no-match |

Genuine mean ≈ 0.54 (sd 0.17), impostor mean ≈ 0.48 (sd 0.09) → **d′ ≈ 0.5**.
A wrong finger repeatedly outscored the correct one. **No threshold separates
these distributions**, so the 8/10 genuine match rate above is not evidence of a
working biometric — it is a coin weighted slightly in your favour.

The likely cause is structural, not a tuning error: `elanpress_ncc_best` takes
the **maximum** zero-mean NCC over ±60px × ±20px translation and ±12° rotation —
on the order of 35,000 candidate alignments — computed on **raw, unenhanced**
pixels. Maximising over that many trials inflates the score for any pair of
textured images, which is exactly the observed symptom (and also why a single
verification takes seconds).

**Tightening the search window does not help — tested and refuted.** The
hypothesis was that maximising over ~35,000 weakly-overlapping alignments
inflated impostor scores. Narrowing to ±15px / ±6px / ±4° (1,209 alignments) and
raising the minimum overlap from 19% to 64% of the frame gave:

| | n | mean | sd | max |
|---|---|---|---|---|
| genuine | 8 | 0.469 | 0.090 | 0.610 |
| **impostor** | 7 | 0.452 | 0.190 | **0.771** |

**d′ = 0.12.** The impostor maximum now *exceeds* the genuine maximum. In that
run a wrong finger was accepted **3 times out of 10** while the enrolled finger
was accepted **2 out of 8** — the matcher admits the wrong finger more often
than the right one. Search-space inflation is therefore **not** the cause, and
no threshold or window tuning rescues this.

### What actually limits accuracy: fingertip coverage, not the algorithm

Measured offline on a labelled dataset of 12 right-index and 14 right-middle
captures (`tools/matcher-lab.py`, `tools/sweep.py`), evaluated the way the driver
works — a template *set* versus a probe, taking the best score.

**Finding 1 — raw-intensity correlation is worse than chance.** Reimplementing
the shipped matcher gives **d′ = −0.72**: impostors systematically outscore
genuine presses.

**Finding 2 — local contrast normalisation is the single biggest algorithmic
win.** Dividing out local energy removes the pressure/contact-area signal that
swamps identity. With a rotation search it reaches **d′ ≈ 1.5**.

**Finding 3 — parameter tuning is exhausted.** A 19-variant sweep over LCN
sigma, rotation range/step, translation window, minimum overlap, both Gabor
orientation conventions, bandpass and binarisation landed everything between
**d′ 0.96 and 1.50**. Measured ridge frequency is 0.110 cyc/px (≈9 px period),
so a hard-coded 0.11 is correct and is *not* the problem.

**Finding 4 — the presses barely overlap.** Only **6 of 66 genuine pairs (9%)**
score above the best impostor pair; genuine median 0.18 versus impostor median
0.15. Two presses of the same finger typically correlate no better than presses
of two different fingers, because a 150×52 window sees a small patch of
fingertip and placement varies by more than the window is wide.

**Finding 5 — template count dominates everything.**

| enrolled templates | d′ | FAR @ 10% FRR |
|---|---|---|
| 1 | 0.68 | 92.9% |
| 3 | 0.66 | 78.6% |
| 6 | 1.61 | 21.4% |
| 10–11 | **2.04** | 28.6% |

Still rising at the edge of the data. `elanpress` enrols 8.

**Finding 6 — more templates help, then saturate.** Averaged over random
template subsets (a fixed ordering gives an optimistic selection effect):

| enrolled templates | d′ | EER |
|---|---|---|
| 1 | 0.56 | 35.8% |
| 10 | 1.21 | 28.4% |
| 20 | 1.40 | 22.9% |
| 30 | 1.54 | 20.4% |

Nearly a tripling from 1 to 30, but flattening after ~20.

**Finding 7 — band-limited phase-only correlation performs *worse*, not better.**
BLPOC is the standard recommendation for small-area fingerprint sensors (Ito et
al. 2004), and it is implemented and sanity-checked here (`tools/poc.py`,
`blpoc(a,a) = 1.0000`). On this data it scores **d′ 0.09–0.52** against NCC's
1.54, across full-band and band-limited variants, with and without LCN.

The reason is the same overlap problem: POC assumes two views of the *same
scene* under translation and correlates phase over the whole image, so partial
overlap degrades it more than NCC, which restricts to an explicit overlap region
with a minimum-area constraint. **Do not assume the small-area literature
transfers here** — those results come from sensors where successive captures see
substantially the same region.

**Conclusion.** The fix is not primarily a better matcher. It is (a) local
contrast normalisation before correlating, (b) a rotation search, and (c) **many
more enrolment templates, deliberately covering the fingertip** — which is what
commercial small-area readers do. Until that is in place, treat `elanpress` as a
**capture driver that proves the hardware works**, not as an authentication
mechanism, and do not wire it to PAM.

---

## What shipped: the LCN matcher (August 2026)

Findings 1 and 2 above are now implemented in the driver, in
`libfprint-elanpress` commit **`85e7cd4`**, packaged as
**`1.94.9.elanpress.lcn`**. The scoring path was replaced; capture, enrolment
and the stored print format were not.

**What changed.** Three things, in `elanpress-match.c`:

1. **Local contrast normalisation** (separable Gaussian, σ = 6) of both images
   before correlating. This is the fix for finding 1 — raw intensity on a press
   sensor mostly encodes press force and contact area, and the old matcher was
   ranking on that instead of on ridge structure, which is why it inverted.
2. **Rotation search** ±12° in 4° steps, bilinear resampling.
3. **A per-pixel validity mask** through the rotation, with the correlation
   assembled from six weighted sums so out-of-frame pixels are *excluded*
   rather than counted. The old code zero-filled the corners and correlated
   them anyway; zero is an ordinary value in a normalised image, so those
   corners dragged both means and — being in roughly the same place in both
   images — contributed spurious agreement. Worth d′ 1.83 → 1.95 on its own.

The decision threshold moves **0.55 → 0.30**. The score scale changed
completely, so the old number carries no meaning here.

**Existing enrolments stay valid.** The print blob format and version are
unchanged, and scores are recomputed from the stored images at match time
rather than persisted, so there is no mixed-scale hazard. No re-enrolment is
needed for this upgrade. (Switching *drivers* still requires one — see above.)

### Measured accuracy

45-image labelled dataset, 8 templates enrolled (what `elanpress` enrols),
scored the way the driver actually scores:

| protocol | d′ | EER | FRR @ 0.30 | FAR @ 0.30 |
|---|---|---|---|---|
| shipped **before** this change (raw-intensity NCC) | **−0.72** | — | — | — |
| pooled (random enrolment subsets) | 1.17 | 28.6% | 47.6% | **12.5%** |
| realistic (enrol one session, probe a later one) | 2.15 | 15.5% | 0.0% | **14.3%** |

The C implementation is cross-checked against the Python reference in
`tools/matcher-lab.py` and `tools/exp-py-minutiae/wncc.py`: LCN planes agree to
8.9e-16, masked NCC to 5.0e-10, end-to-end to 2.0e-08, and every image scores
exactly 1.000000 against itself. The harness is `tools/ctest/` in the libfprint
tree and compiles the driver's own source file, so what it measures is what the
driver runs. Cost is ~25 ms per probe-vs-template pair, ~200 ms for an
8-template verify.

### ⚠️ This is still NOT authentication-grade. Do not wire it to PAM.

The improvement is real, and the change removes a genuine security defect — the
previous matcher was *worse than chance*, and a d′ of −0.72 means a wrong
finger was systematically **favoured** over the enrolled one. But look at the
FAR column: **at the shipped threshold roughly one impostor press in eight is
accepted**, and because the driver takes the maximum over an 8-template set,
more enrolled fingers means more chances to get in.

An equal-error rate of 15–29% is orders of magnitude away from what a
fingerprint reader is normally expected to deliver (commercial sensors target
an FAR around 1 in 50,000). The ceiling here is not the algorithm — it is
finding 4 above: a 150×52 window sees too small a patch of fingertip for two
presses of the same finger to reliably overlap.

Treat this as **a convenience unlock at best, and a demonstration that the
hardware produces usable images at worst.** Specifically:

- **Do not add it to `/etc/pam.d/system-auth`, `/etc/pam.d/sudo`, or any other
  PAM stack.** See the PAM section above for how that goes wrong even when the
  matcher works.
- **Do not use it as a sole authentication factor** for anything you care
  about — screen unlock on a machine holding secrets included.
- A synthetic ridge-frequency grating scores *higher* than any genuine pair in
  the dataset, so this is not presentation-attack resistant in any sense.

The driver marks itself accordingly: `fprintd-list` and the GNOME Settings
enrolment dialog show it as *"ElanTech press-type fingerprint sensor
(experimental, not authentication-grade)"*, and it logs a warning to the
journal once per `fprintd` activation.

---

**The single biggest factor is enrolling on a non-wedged device.** The same
driver and threshold went from 4/9 to 8/10 purely by re-enrolling after a
successful USB selective-suspend recovery. If your match rate is poor, re-enroll
before touching anything else — `elan-fp enroll` unwedges first for this reason.

---

## Licence

Documentation and packaging here: MIT. The `elan-patch/` diff is against
libfprint and is therefore **LGPL-2.1-or-later**, as is anything derived from
libfprint sources.
