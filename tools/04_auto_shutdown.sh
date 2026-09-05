#!/usr/bin/env bash
# Non-interactive, whole robot-stack shutdown via 04_safe_shutdown.sh.
#
# Auto-answers the LOWER AND SHUTDOWN / CONFIRM LOW prompts; every safety gate
# that is backed by telemetry still applies (HOLD
# handshake, wait_for_safe_height blocking on stable feedback <= 1.0).
#
# What you give up versus the interactive script: the visual inspection of the
# platform before processes stop. Use only when you can see the robot or the
# platform is already known-low. Session ownership and PID confirmation are
# skipped: this also stops other developers' matching robot processes locally.
# Unrelated applications, the remote model server and CAN transport stay up.
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUMAN_DAGGER_AUTO_CONFIRM=1 HUMAN_DAGGER_SHUTDOWN_ALL=1 \
  exec bash "${script_dir}/04_safe_shutdown.sh" "$@"
