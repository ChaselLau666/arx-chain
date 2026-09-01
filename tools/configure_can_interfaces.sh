#!/bin/bash
set -euo pipefail

# Configure each delivered slcan adapter independently. Unlike the delivered
# watchdog scripts, this tool never kills unrelated slcand processes.

configure_one() {
  local interface=$1
  local device="/dev/arx${interface}"

  if ip link show "${interface}" >/dev/null 2>&1; then
    if ! ip link show "${interface}" | grep -q "UP"; then
      sudo ip link set "${interface}" up
    fi
    echo "${interface}: already configured and UP"
    return
  fi

  if [[ ! -e "${device}" ]]; then
    echo "Refused: ${device} is missing" >&2
    exit 1
  fi
  if [[ "${interface}" == "can5" ]] && pgrep -f '/arx_lift_controller/lift_controller' >/dev/null; then
    echo "Refused: body is running but can5 is missing; lower safely and stop body before recovery" >&2
    exit 1
  fi
  if [[ "${interface}" =~ ^can[13]$ ]] && pgrep -f '/arx_x5_controller/' >/dev/null; then
    echo "Refused: arm controller is running but ${interface} is missing" >&2
    exit 1
  fi

  echo "${interface}: starting slcand from ${device}"
  sudo slcand -o -f -s8 "${device}" "${interface}"
  for _ in $(seq 1 20); do
    ip link show "${interface}" >/dev/null 2>&1 && break
    sleep 0.1
  done
  if ! ip link show "${interface}" >/dev/null 2>&1; then
    echo "Refused: slcand did not create ${interface}" >&2
    exit 1
  fi
  sudo ip link set "${interface}" up
  echo "${interface}: UP"
}

for interface in can1 can3 can5; do
  configure_one "${interface}"
done

ip -br link show can1
ip -br link show can3
ip -br link show can5
