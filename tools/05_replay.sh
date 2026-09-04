#!/bin/bash
set -euo pipefail

# Replay a recorded episode onto the arms.
#
# Unlike 01_collect.sh this script starts neither body nor VR. Replay is an
# open-loop playback with nobody in the control loop, so every precondition is
# checked and refused rather than repaired, and the replay itself runs in this
# terminal so Ctrl+C reaches it directly.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Domain must come from the machine (/etc/environment); see 03_inference.sh.
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. The robot identity lives in /etc/environment (ark-1=62, ark-2=63); refusing to guess which robot to talk to}"
: "${EPISODE:?Set EPISODE to the recorded HDF5, for example datasets/episode_19.hdf5}"

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

episode_path="${EPISODE}"
[[ "${episode_path}" = /* ]] || episode_path="${repo_root}/act/${EPISODE}"
if [[ ! -f "${episode_path}" ]]; then
  echo "Refused: episode not found: ${episode_path}" >&2
  exit 1
fi

# CAN first: body talks to the lift over can5, so it cannot come up before this.
# Brought up with a one-shot slcand rather than arx_can1.sh and friends, whose
# repair path runs `pkill -9 slcand` and takes down every other interface too.
declare -A CAN_DEVICE=( [can1]=/dev/arxcan1 [can3]=/dev/arxcan3 [can5]=/dev/arxcan5 )
for interface in can1 can3 can5; do
  ip link show "${interface}" 2>/dev/null | grep -q 'UP' && continue
  if [[ "${SKIP_AUTOSTART:-0}" == 1 ]]; then
    echo "Refused: ${interface} is not UP." >&2
    exit 1
  fi
  device=${CAN_DEVICE[${interface}]}
  if ! ip link show "${interface}" >/dev/null 2>&1; then
    if [[ ! -e "${device}" ]]; then
      echo "Refused: ${device} is missing; the CAN adapter for ${interface} is unplugged." >&2
      exit 1
    fi
    echo "  ${interface}: starting slcand on ${device}"
    sudo slcand -o -f -s8 "${device}" "${interface}" || {
      echo "Refused: slcand failed for ${interface}." >&2; exit 1; }
    for _ in $(seq 1 20); do
      ip link show "${interface}" >/dev/null 2>&1 && break
      sleep 0.25
    done
  fi
  sudo ip link set "${interface}" up || { echo "Refused: could not bring ${interface} up." >&2; exit 1; }
  ip link show "${interface}" 2>/dev/null | grep -q 'UP' || {
    echo "Refused: ${interface} is still not UP after bring-up." >&2; exit 1; }
done

# Body: launch when absent, reuse when already up - the same shape as the arm
# stack below, so running this script twice in a row starts one body, not two.
# Replay pins /lift through its parameter service, so it has to exist.
if ros2 node list 2>/dev/null | grep -qx '/lift'; then
  echo "Reusing the running body."
elif [[ "${SKIP_AUTOSTART:-0}" == 1 ]]; then
  echo "Refused: /lift is not running." >&2
  exit 1
else
  body_log=/tmp/replay_body.log
  echo "Starting body; log: ${body_log}"
  echo "WARNING: body starts now and the lift may home itself. Stand clear."
  setsid bash -c '
    set +u
    source /opt/ros/jazzy/setup.bash
    source /home/arx/LIFT/body/ROS2/install/setup.bash
    set -u
    exec ros2 launch arx_lift_controller lift.launch.py
  ' > "${body_log}" 2>&1 < /dev/null &

  body_ready=false
  for _ in $(seq 1 60); do
    sleep 0.5
    if ros2 node list 2>/dev/null | grep -qx '/lift'; then
      body_ready=true
      break
    fi
  done
  if [[ "${body_ready}" != true ]]; then
    echo "Refused: /lift did not appear within 30s. Check ${body_log}" >&2
    exit 1
  fi
  echo "Body is up."
fi

# VR must be absent. Its arm stack (v2_pos_control) subscribes to ARX_VR_* instead
# of arm_master_*_status, so replay commands would be ignored while VR keeps
# driving the same arms.
if pgrep -f '/serial_port_node$' >/dev/null; then
  echo "Refused: the VR serial node is running; replay and VR would both drive the arms." >&2
  exit 1
fi
if ros2 topic info /ARX_VR_L 2>/dev/null | grep -qE 'Publisher count: [1-9]'; then
  echo "Refused: /ARX_VR_L has a publisher. Stop VR before replaying." >&2
  exit 1
fi
for vr_node in /vr_arm_l /vr_arm_r; do
  if ros2 node list 2>/dev/null | grep -qx "${vr_node}"; then
    echo "Refused: ${vr_node} is running, which is the VR teleop arm stack." >&2
    echo "Replay needs the joint-command stack; stop v2_pos_control first." >&2
    exit 1
  fi
done

# Joint-command arm stack: launch when absent, reuse when complete, refuse when partial.
arm_topics=0
for topic in /arm_slave_l_status /arm_slave_r_status; do
  ros2 topic list 2>/dev/null | grep -qx "${topic}" && arm_topics=$((arm_topics + 1)) || true
done
if [[ ${arm_topics} -eq 0 ]]; then
  # Started headless on purpose: this script must work over SSH, where a
  # gnome-terminal launch fails with "cannot open display".
  arm_log=/tmp/replay_arms.log
  echo "Starting the joint-command arm stack; log: ${arm_log}"
  echo "WARNING: the arms power up now and may home themselves. Stand clear."
  setsid bash -c '
    set +u
    source /opt/ros/jazzy/setup.bash
    source /home/arx/LIFT/ARX_X5/ROS2/X5_ws/install/setup.bash
    set -u
    exec ros2 launch arx_x5_controller v2_joint_control.launch.py
  ' > "${arm_log}" 2>&1 < /dev/null &

  ready=0
  for _ in $(seq 1 30); do
    sleep 1
    ready=0
    for topic in /arm_slave_l_status /arm_slave_r_status; do
      ros2 topic list 2>/dev/null | grep -qx "${topic}" && ready=$((ready + 1)) || true
    done
    [[ ${ready} -eq 2 ]] && break
  done
  if [[ ${ready} -ne 2 ]]; then
    echo "Refused: arm stack did not come up (${ready}/2 feedback topics)." >&2
    echo "Check ${arm_log}" >&2
    exit 1
  fi
  echo "Arm stack is up."
elif [[ ${arm_topics} -ne 2 ]]; then
  echo "Refused: arm stack is partial (${arm_topics}/2 feedback topics)." >&2
  exit 1
else
  echo "Reusing the running joint-command arm stack."
fi

# Cameras are intentionally not started: replay never reads images.

set +u
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate act
set -u

echo "Episode: ${episode_path}"
echo "Default mode is DRY-RUN; pass --execute to publish arm targets."
cd "${repo_root}/act"
exec python replay.py --episode_path "${episode_path}" "$@"
