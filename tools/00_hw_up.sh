#!/bin/bash
# Idempotent hardware bringup for ark-1, shared by the ACT and Tau0VLA
# inference paths.
#
# Brings up only the layers that are safe to automate:
#   1. can1 / can3 / can5
#   2. /lift (body controller)
#   3. /lift fixed_height  -- THIS RAISES THE PLATFORM, so it is confirmed
#
# The vendor arx_can*.sh scripts are deliberately not used: each is a
# `while true` watchdog that runs `pkill -9 slcand`, so three of them running
# together kill each other's daemons. This script starts one slcand per device
# and only ever signals the daemon bound to that device.
#
# It does NOT start the two v2_joint_control arm controllers. The SDK calls
# arx_x(...) during initialization and may move the arms before any model
# publisher exists, and this machine has no hold_guard to hold them in the
# meantime. Start those by hand with the workspace clear and the emergency stop
# reachable. Cameras are left to 03_*_inference.sh, which starts them itself.
#
# Re-running is safe: every step is skipped when already satisfied.

set -euo pipefail

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-62}
: "${LIFT_HEIGHT:=15.5}"
: "${LOG_DIR:=/home/arx/logs/hw_up}"
: "${EXPECTED_HOST:=ark-1}"

check_only=false
assume_yes=false
skip_height=false
for argument in "$@"; do
  case "${argument}" in
    --check) check_only=true ;;
    --yes) assume_yes=true ;;
    --no-height) skip_height=true ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${argument}" >&2
      exit 1
      ;;
  esac
done

if [[ "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Refused: this script is for ${EXPECTED_HOST}, but hostname is $(hostname)." >&2
  echo "ark-2 has its own split 10_/20_/30_ lifecycle; the two are not interchangeable." >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

can_interfaces=(can1 can3 can5)

can_is_up() {
  ip link show "$1" >/dev/null 2>&1 && ip link show "$1" | grep -q 'UP'
}

lift_is_running() {
  ros2 node list 2>/dev/null | grep -qx '/lift'
}

read_fixed_height() {
  ros2 param get /lift fixed_height 2>/dev/null | awk '/Double value is:/{print $NF}'
}

report_state() {
  echo "Current state:"
  local interface
  for interface in "${can_interfaces[@]}"; do
    if can_is_up "${interface}"; then
      echo "  ${interface}: UP"
    elif ip link show "${interface}" >/dev/null 2>&1; then
      echo "  ${interface}: exists but DOWN"
    else
      echo "  ${interface}: absent"
    fi
  done
  if lift_is_running; then
    echo "  /lift: running, fixed_height=$(read_fixed_height)"
  else
    echo "  /lift: not running"
  fi
  local arms cameras
  arms=$(pgrep -fc '/arx_x5_controller/[X]5Controller.*v2_joint_control.yaml' || true)
  cameras=$(pgrep -fc '/realsense2_camera/[r]ealsense2_camera_node' || true)
  echo "  v2 arm controllers: ${arms:-0} (started by hand, expected 2 before inference)"
  echo "  realsense: ${cameras:-0} (started by 03_*_inference.sh, expected 0 or 3)"
}

if [[ "${check_only}" == true ]]; then
  report_state
  exit 0
fi

confirm() {
  local prompt=$1 phrase=$2 reply
  if [[ "${assume_yes}" == true ]]; then
    echo "${prompt}"
    echo "  (--yes given, proceeding without confirmation)"
    return 0
  fi
  echo "${prompt}"
  read -r -p "Type '${phrase}' to continue: " reply
  if [[ "${reply}" != "${phrase}" ]]; then
    echo "Aborted; nothing further was started." >&2
    exit 1
  fi
}

echo '=== 1/3  CAN ==='
for interface in "${can_interfaces[@]}"; do
  device="/dev/arx${interface}"
  if can_is_up "${interface}"; then
    echo "  ${interface} already UP"
    continue
  fi
  if [[ ! -e "${device}" ]]; then
    echo "Refused: ${device} is missing; the USB-CAN adapter did not enumerate." >&2
    echo "Check the cable, then look for its ttyACM node with: ls -l /dev/arxcan*" >&2
    exit 1
  fi
  echo "  starting ${interface} on ${device}"
  # Signal only the daemon bound to this device, never a global pkill.
  sudo pkill -f "slcand.*${device}" 2>/dev/null || true
  sleep 0.3
  sudo ip link set "${interface}" down 2>/dev/null || true
  sudo slcand -o -f -s8 "${device}" "${interface}"
  for _ in $(seq 1 30); do
    ip link show "${interface}" >/dev/null 2>&1 && break
    sleep 0.1
  done
  if ! ip link show "${interface}" >/dev/null 2>&1; then
    echo "Refused: slcand did not create ${interface}." >&2
    exit 1
  fi
  sudo ip link set "${interface}" up
  if ! can_is_up "${interface}"; then
    echo "Refused: ${interface} did not come UP." >&2
    exit 1
  fi
  echo "  ${interface} UP"
done

echo
echo '=== 2/3  body / lift ==='
if lift_is_running; then
  echo '  /lift already running; reusing it'
else
  confirm "Starting the body controller. Confirm the platform is at a safe low position." 'START BODY'
  mkdir -p "${LOG_DIR}"
  body_log="${LOG_DIR}/body_$(date +%Y%m%d_%H%M%S).log"
  setsid bash -c 'cd /home/arx/LIFT/body/ROS2 && source install/setup.bash && exec ros2 launch arx_lift_controller lift.launch.py' \
    >"${body_log}" 2>&1 &
  body_pid=$!
  echo "  body launched, pid ${body_pid}, log ${body_log}"
  for _ in $(seq 1 40); do
    lift_is_running && break
    sleep 1
  done
  if ! lift_is_running; then
    echo "Refused: /lift did not appear within 40s. See ${body_log}." >&2
    exit 1
  fi
  echo '  /lift is up'
fi

echo
echo '=== 3/3  fixed_height ==='
if [[ "${skip_height}" == true ]]; then
  echo '  --no-height given; leaving fixed_height untouched'
  echo "  note: inference verifies fixed_height == its --expected-height and refuses otherwise"
else
  current_height=$(read_fixed_height)
  if [[ -n "${current_height}" ]] && awk -v a="${current_height}" -v b="${LIFT_HEIGHT}" 'BEGIN{exit !(a==b)}'; then
    echo "  fixed_height is already ${LIFT_HEIGHT}"
  else
    echo "  fixed_height is currently ${current_height:-unset} (-1 means it follows raw VR height)"
    confirm "Setting fixed_height to ${LIFT_HEIGHT} WILL RAISE THE PLATFORM. Clear the workspace." 'RAISE LIFT'
    height_set=false
    for _ in $(seq 1 20); do
      if ros2 param set /lift fixed_height "${LIFT_HEIGHT}"; then
        height_set=true
        break
      fi
      sleep 0.5
    done
    if [[ "${height_set}" != true ]]; then
      echo 'Refused: could not set /lift fixed_height.' >&2
      echo 'The body process is intentionally left running so the lift is never stopped at height;' >&2
      echo 'use tools/04_safe_shutdown.sh to lower and stop it.' >&2
      exit 1
    fi
    echo "  fixed_height set to ${LIFT_HEIGHT}"
  fi
fi

echo
report_state
cat <<'NEXT'

Hardware layer is up. Remaining steps are manual by design:

  1. Clear the workspace and put the emergency stop within reach.
  2. Start the two arm controllers -- this can move the arms immediately:
       cd /home/arx/LIFT/ARX_X5/ROS2/X5_ws && source install/setup.bash
       ros2 launch arx_x5_controller v2_joint_control.launch.py
  3. Dry-run the policy (it starts the three cameras itself):
       cd /home/arx/ROS2_LIFT_Play/tools && ./03_tau0vla_inference.sh
     then, once the dry-run looks right:
       ./03_tau0vla_inference.sh --execute

Shut down with tools/04_safe_shutdown.sh. Never SIGINT the lift controller
directly: it crashes in its destructor, which is unsafe while raised.
NEXT
