#!/bin/bash
set -euo pipefail

# Replay a recorded episode onto the arms.
#
# Unlike 01_collect.sh this script starts neither body nor VR. Replay is an
# open-loop playback with nobody in the control loop, so every precondition is
# checked and refused rather than repaired, and the replay itself runs in this
# terminal so Ctrl+C reaches it directly.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-62}
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

# Body must already be up: replay pins /lift through its parameter service and
# refuses to start if that service is missing. Starting body is deliberately
# left to the operator, who must first confirm the platform is safely low.
if ! ros2 node list 2>/dev/null | grep -qx '/lift'; then
  echo "Refused: /lift is not running. Start body only while the platform is at a safe low position." >&2
  exit 1
fi

for interface in can1 can3 can5; do
  if ! ip link show "${interface}" 2>/dev/null | grep -q 'UP'; then
    echo "Refused: ${interface} is not UP." >&2
    exit 1
  fi
done

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
