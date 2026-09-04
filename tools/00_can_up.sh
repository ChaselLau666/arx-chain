#!/usr/bin/env bash
# Bring up the slcand CAN transports this robot needs (can1/can3/can5).
#
# slcand does not survive a reboot, so run this once after every boot, before
# 05_dagger_pickplace.sh / 10_robot_up_pickplace.sh / 01_collect.sh.
#
# This intentionally does NOT start the vendor arx_can*.sh watchdog loops:
# each of those busy-spins and can globally kill every slcand process when one
# interface drops (see the note in 10_robot_up.sh). One slcand per interface,
# no watchdog, idempotent.
set -Eeuo pipefail

interfaces=(1 3 5)

for n in "${interfaces[@]}"; do
  dev="/dev/arxcan${n}"
  iface="can${n}"

  if [[ ! -e "$dev" ]]; then
    echo "Refused: ${dev} does not exist; is the USB2CAN adapter plugged in?" >&2
    exit 1
  fi

  # Idempotent: if the interface is already UP with a live slcand, leave it.
  if ip -br link show "$iface" 2>/dev/null | grep -q "UP"; then
    echo "${iface}: already UP, leaving as is"
    continue
  fi

  # A dead slcand can leave a half-created interface behind; clear it first.
  if ip link show "$iface" >/dev/null 2>&1; then
    echo "${iface}: exists but not UP; recreating"
    sudo ip link set "$iface" down 2>/dev/null || true
    pids=$(pgrep -f "slcand .*${dev} ${iface}" || true)
    [[ -n "$pids" ]] && sudo kill $pids && sleep 0.5
  fi

  sudo slcand -o -f -s8 "$dev" "$iface"
  sudo ip link set "$iface" up
  echo "${iface}: created and UP"
done

echo
ip -br link show | grep -E "can[135]"
