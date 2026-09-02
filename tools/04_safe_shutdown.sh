#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-62}

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

if ros2 node list 2>/dev/null | grep -qx '/lift'; then
  current_height=$(timeout 4 ros2 topic echo --once /body_information 2>/dev/null \
    | awk '/^height:/{print $2; exit}')
  if [[ -z "${current_height}" ]]; then
    echo "Refused: /lift is running but /body_information feedback is unavailable." >&2
    echo "Keep body running and diagnose its feedback; it is unsafe to stop an unobservable body." >&2
    exit 1
  fi

  echo "Current feedback height: ${current_height}"
  echo "Target command: 0.0"
  echo "Direction: DOWN (or HOLD if already low)"
  echo "Expected behavior: platform reaches a stable feedback <= 1.0 before body is stopped."
  echo "WARNING: any unsaved in-memory episode will be discarded."
  read -r -p 'Type LOWER AND SHUTDOWN to continue: ' confirmation
  if [[ "${confirmation}" != "LOWER AND SHUTDOWN" ]]; then
    echo "Cancelled; no command or signal sent."
    exit 1
  fi

  ros2 param set /lift fixed_height 0.0
  # Fallback for an older already-running binary whose fixed_height was applied
  # only from VR/joy callbacks. This command also requests zero wheel/body motion.
  ros2 topic pub --once /body_control arm_control/msg/PosCmd '{height: 0.0, mode1: 2}'

  /home/arx/miniconda3/envs/act/bin/python \
    "${repo_root}/act/wait_for_safe_height.py" \
    --safe-max 1.0 --tolerance 0.02 --window 2.0 --timeout 90.0

  echo "Feedback is low and stable. Visually inspect the platform now."
  read -r -p 'Type CONFIRM LOW to stop all control programs: ' low_confirmation
  if [[ "${low_confirmation}" != "CONFIRM LOW" ]]; then
    echo "Refused: body remains running at low target; no processes were stopped."
    exit 1
  fi
else
  if pgrep -f '/arx_lift_controller/lift_controller' >/dev/null; then
    echo "Refused: a body process exists but /lift is not visible in ROS_DOMAIN_ID=${ROS_DOMAIN_ID}." >&2
    echo "Do not stop it while its height is unobservable. Check ROS_DOMAIN_ID and body logs." >&2
    exit 1
  fi

  echo "INCOMPLETE STACK: /lift and its body process are not running."
  echo "The script cannot read height or lower the platform in this state."
  echo "Required physical state: platform is already at a safe low position."
  echo "Expected behavior: only remaining collector, VR, cameras, arms, and CAN watchdogs will stop."
  echo "WARNING: any unsaved in-memory episode will be discarded."
  read -r -p 'After visually confirming the platform is low, type CONFIRM ALREADY LOW: ' low_confirmation
  if [[ "${low_confirmation}" != "CONFIRM ALREADY LOW" ]]; then
    echo "Cancelled; no command or signal sent."
    exit 1
  fi
fi

stop_pattern() {
  local label=$1
  local pattern=$2
  local pids
  pids=$(pgrep -f "${pattern}" || true)
  if [[ -z "${pids}" ]]; then
    echo "${label}: not running"
    return
  fi
  echo "${label}: SIGINT -> ${pids//$'\n'/ }"
  kill -INT ${pids} 2>/dev/null || true
  for _ in $(seq 1 30); do
    sleep 0.2
    if ! pgrep -f "${pattern}" >/dev/null; then
      return 0
    fi
  done
  pids=$(pgrep -f "${pattern}" || true)
  if [[ -n "${pids}" ]]; then
    echo "${label}: SIGTERM -> ${pids//$'\n'/ }"
    kill -TERM ${pids} 2>/dev/null || true
  fi
}

stop_pattern "collector" '[p]ython .*collect.py'
stop_pattern "inference" '[p]ython .*inference.py'
stop_pattern "VR diagnostics" 'ros2 topic (echo|hz) /ARX_VR_[LR]'
stop_pattern "VR serial launcher" '/opt/ros/jazzy/bin/ros2 run serial_port serial_port_node'
stop_pattern "VR serial node" '/serial_port_node$'
stop_pattern "RealSense launchers" '/opt/ros/jazzy/bin/ros2 launch realsense2_camera rs_launch.py'
stop_pattern "RealSense nodes" '/realsense2_camera_node'
stop_pattern "arm launcher" '/opt/ros/jazzy/bin/ros2 launch arx_x5_controller (v2_pos_control|v2_joint_control|open_double_arm).launch.py'
stop_pattern "arm nodes" '/arx_x5_controller/X5Controller'
stop_pattern "body launcher" '/opt/ros/jazzy/bin/ros2 launch arx_lift_controller lift.launch.py'
stop_pattern "body node" '/arx_lift_controller/lift_controller'
stop_pattern "CAN watchdogs" '/bin/bash ./arx_can[135].sh'

echo "Verifying control processes..."
remaining=$(ps -eo pid,args | grep -E \
  '(lift_controller|X5Controller|serial_port_node|realsense2_camera_node|collect.py|inference.py)' \
  | grep -v grep || true)
if [[ -n "${remaining}" ]]; then
  echo "WARNING: some control processes remain:" >&2
  echo "${remaining}" >&2
  exit 1
fi

echo "Control stack stopped safely. CAN transport is intentionally left UP for the next run."
for interface in can1 can3 can5; do
  ip -br link show "${interface}" 2>/dev/null || echo "${interface}: absent"
done
