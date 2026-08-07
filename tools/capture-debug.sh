#!/bin/sh
# Capture an image from an ELAN 04f3:0c6e sensor with full driver debug,
# bypassing the traps that make this painful.
#
#   1. fprintd.service sets PrivateTmp=yes, so debug images a patched driver
#      writes to /tmp are invisible AND are destroyed when the service stops.
#      Running fprintd manually avoids the namespace entirely.
#   2. A killed fprintd-verify leaves the device claimed; the next EnrollStart
#      then fails with a misleading "Timeout was reached".
#
# Usage:  sudo ./capture-debug.sh [output.pgm]
#
# Prefer ./img-capture.sh if you only want an image — it skips fprintd, D-Bus,
# polkit, enrollment and identify, all of which have their own failure modes.

set -e

OUT="${1:-/tmp/elan-capture.pgm}"
LOG="${OUT%.pgm}.log"

if [ "$(id -u)" -ne 0 ]; then
  echo "needs root (USB access)" >&2
  exit 1
fi

echo "==> stopping fprintd.service (clears any wedged claim)"
systemctl stop fprintd 2>/dev/null || true
killall fprintd 2>/dev/null || true
sleep 1

echo "==> starting fprintd manually with G_MESSAGES_DEBUG=all"
G_MESSAGES_DEBUG=all /usr/lib/fprintd -t > "$LOG" 2>&1 &
FPID=$!
sleep 2

echo "==> running verify — touch the sensor when prompted"
fprintd-verify "${SUDO_USER:-$USER}" || true

kill $FPID 2>/dev/null || true
wait $FPID 2>/dev/null || true

echo
echo "==> driver lines (G_DEBUG_HERE noise stripped):"
grep -a "libfprint-elan" "$LOG" \
  | sed 's/.*DEBUG: //' \
  | grep -avE '^[0-9:.]+: [0-9]+: \.\./' || true

echo
echo "==> capture state machine:"
grep -a "libfprint-SSM" "$LOG" | sed 's/.*DEBUG: //' || true

echo
echo "full log: $LOG"
echo
echo "Reading the result:"
echo "  'failed in state 1 ... timed out'  -> normal end of a swipe (good)"
echo "  'failed in state 2 ... proto'      -> pre_scan returned a non-0x55 byte."
echo "                                        0xaf = finger removed / sensor busy."
echo "                                        Persistent 0xaf = device wedged;"
echo "                                        needs a power cycle, not a reset."
