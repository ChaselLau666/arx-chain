#!/usr/bin/env bash
# Non-interactive wrapper around 04_safe_shutdown.sh.
#
# Auto-answers the LOWER AND SHUTDOWN / CONFIRM LOW prompts; every safety gate
# that is backed by telemetry still applies (manifest verification, HOLD
# handshake, wait_for_safe_height blocking on stable feedback <= 1.0).
#
# What you give up versus the interactive script: the visual inspection of the
# platform before processes stop. Use only when you can see the robot or the
# platform is already known-low. The legacy broad-match path still refuses to
# run without a human.
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUMAN_DAGGER_AUTO_CONFIRM=1 exec bash "${script_dir}/04_safe_shutdown.sh" "$@"
