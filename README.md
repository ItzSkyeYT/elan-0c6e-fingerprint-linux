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
minutiae matching entirely with **normalized cross-correlation** (threshold 0.55,
±60 px x, ±20 px y, ±12° rotation tolerance). It registers as an `FpDevice`
rather than an `FpImageDevice`, so NBIS/bozorth3 are bypassed.

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
| **ASUS ROG Flow X13 GV301QH** | **`0x0161`** | **`elanpress` (template from healthy device)** | **✓** | **8/10** | **stock 0.55 threshold, no hangs** |

**The single biggest factor is enrolling on a non-wedged device.** The same
driver and threshold went from 4/9 to 8/10 purely by re-enrolling after a
successful USB selective-suspend recovery. If your match rate is poor, re-enroll
before touching anything else — `elan-fp enroll` unwedges first for this reason.

---

## Licence

Documentation and packaging here: MIT. The `elan-patch/` diff is against
libfprint and is therefore **LGPL-2.1-or-later**, as is anything derived from
libfprint sources.
