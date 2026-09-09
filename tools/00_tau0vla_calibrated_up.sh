#!/bin/bash
# One-command, idempotent profiled ARX hardware bring-up for calibrated Tau0VLA.
# It never starts policy or calibration publishers. The calibrated rollout can
# opt into the reviewed non-interactive sequence with --auto-confirm.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/tau0vla_robot_profile.sh"
load_tau0vla_robot_profile
: "${MODEL_SERVER_URL:=http://192.168.50.2:8000}"
: "${DIRECT_INTERFACE:=enp130s0}"
: "${DIRECT_CLIENT_IP:=192.168.50.1}"
: "${LIFT_HEIGHT:=12.5}"

load_tau0vla_model_profile
check_tau0vla_server
source_tau0vla_ros

publisher_count() {
  ros2 topic info "$1" 2>/dev/null | awk '/^Publisher count:/{print $3; found=1} END{if (!found) print 0}'
}

wait_node() {
  local name=$1
  for _ in $(seq 1 60); do
    ros2 node list 2>/dev/null | grep -qx "${name}" && return 0
    sleep 1
  done
  echo "Refused: ${name} did not start within 60s." >&2
  return 1
}

wait_topic() {
  local topic=$1
  for _ in $(seq 1 60); do
    [[ $(publisher_count "${topic}") -ge 1 ]] && return 0
    sleep 1
  done
  echo "Refused: no publisher for ${topic} after 60s." >&2
  return 1
}

report() {
  echo "Network: $(ip -br addr show "${DIRECT_INTERFACE}" 2>/dev/null || true)"
  for interface in can1 can3 can5; do ip -br link show "${interface}" 2>/dev/null || true; done
  echo "Nodes:"
  ros2 node list 2>/dev/null | grep -E '^/(lift|arm_slave_l|arm_slave_r|camera/camera_[hlr])$' || true
  echo "Candidate:"
  curl --noproxy '*' --max-time 5 "${MODEL_SERVER_URL}/health" 2>/dev/null || true
  echo
}

auto_confirm=false
if [[ "${1:-}" == --check ]]; then
  report
  exit 0
fi
if [[ "${1:-}" == --auto-confirm ]]; then
  auto_confirm=true
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 1
fi

echo "This starts CAN, raises the lift, powers both v2 arms, and starts three cameras."
echo "It does not calibrate grippers or run a policy. Clear the full workspace first."
if [[ "${auto_confirm}" == true ]]; then
  echo "AUTO-CONFIRM: starting the calibrated hardware stack."
else
  read -r -p "Type START CALIBRATED STACK to continue: " confirmation
  if [[ "${confirmation}" != "START CALIBRATED STACK" ]]; then
    echo "Cancelled; nothing was changed."
    exit 1
  fi
fi

route_info=$(ip route get 192.168.50.2 2>/dev/null || true)
if [[ "${route_info}" != *"dev ${DIRECT_INTERFACE}"* || "${route_info}" != *"src ${DIRECT_CLIENT_IP}"* ]]; then
  nmcli connection up "有线连接 1" >/dev/null
  route_info=$(ip route get 192.168.50.2 2>/dev/null || true)
fi
if [[ "${route_info}" != *"dev ${DIRECT_INTERFACE}"* || "${route_info}" != *"src ${DIRECT_CLIENT_IP}"* ]]; then
  echo "Refused: direct route is not active: ${route_info:-unavailable}" >&2
  exit 1
fi
curl --fail --silent --show-error --noproxy '*' --max-time 5 "${MODEL_SERVER_URL}/health" >/dev/null

"${repo_root}/tools/00_can_up.sh"

if ! ros2 node list 2>/dev/null | grep -qx /lift; then
  gnome-terminal --title="tau0vla-body" -- bash -ic \
    "cd /home/arx/LIFT/body/ROS2; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; exec bash"
  wait_node /lift
fi
height_set=false
for _ in $(seq 1 20); do
  if ros2 param set /lift fixed_height "${LIFT_HEIGHT}"; then
    height_set=true
    break
  fi
  sleep .5
done
if [[ "${height_set}" != true ]]; then
  echo "Refused: could not set /lift fixed_height; use safe shutdown." >&2
  exit 1
fi
"${TAU0VLA_PYTHON}" "${repo_root}/act/tau0vla_wait_height.py" \
  --target "${LIFT_HEIGHT}"

arm_count=$(pgrep -fc '/arx_x5_controller/[X]5Controller' || true)
v2_count=$(pgrep -fc '/arx_x5_controller/[X]5Controller.*v2_joint_control.yaml' || true)
if [[ ${arm_count} -eq 0 ]]; then
  gnome-terminal --title="tau0vla-v2-arms" -- bash -ic \
    "cd /home/arx/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; \
     ros2 launch arx_x5_controller v2_joint_control.launch.py; exec bash"
  wait_node /arm_slave_l
  wait_node /arm_slave_r
elif [[ ${arm_count} -ne 2 || ${v2_count} -ne 2 ]]; then
  echo "Refused: found ${arm_count} X5 controllers (${v2_count} v2); stop the conflicting stack." >&2
  exit 1
fi
wait_topic /arm_slave_l_status
wait_topic /arm_slave_r_status

camera_count=$(pgrep -fc '/realsense2_camera/[r]ealsense2_camera_node' || true)
if [[ ${camera_count} -eq 0 ]]; then
  cameras=(
    "camera_h:${CAMERA_H_SERIAL}:/camera/camera_h/color/image_rect_raw/compressed"
    "camera_l:${CAMERA_L_SERIAL}:/camera/camera_l/color/image_rect_raw/compressed"
    "camera_r:${CAMERA_R_SERIAL}:/camera/camera_r/color/image_rect_raw/compressed"
  )
  for entry in "${cameras[@]}"; do
    IFS=: read -r name serial topic <<<"${entry}"
    gnome-terminal --title="${name}" -- bash -ic \
      "cd /home/arx/ROS2_LIFT_Play/realsense; source install/setup.bash; \
       ros2 launch realsense2_camera rs_launch.py camera_name:=${name} \
       depth_module.color_profile:=640x480x90 depth_module.depth_profile:=640x480x90 \
       serial_no:=_${serial}; exec bash"
    wait_topic "${topic}"
  done
elif [[ ${camera_count} -ne 3 ]]; then
  echo "Refused: expected zero or three RealSense processes, found ${camera_count}." >&2
  exit 1
fi
for topic in \
  /camera/camera_h/color/image_rect_raw/compressed \
  /camera/camera_l/color/image_rect_raw/compressed \
  /camera/camera_r/color/image_rect_raw/compressed
do
  wait_topic "${topic}"
done

# Reused cameras must have the same identity as freshly started cameras.
for entry in "camera_h:${CAMERA_H_SERIAL}" "camera_l:${CAMERA_L_SERIAL}" "camera_r:${CAMERA_R_SERIAL}"; do
  IFS=: read -r name expected_serial <<<"${entry}"
  actual_serial=$(ros2 param get "/camera/${name}" serial_no 2>/dev/null || true)
  actual_serial=${actual_serial##*: }
  actual_serial=${actual_serial#_}
  if [[ "${actual_serial}" != "${expected_serial}" ]]; then
    echo "Refused: ${name} serial ${actual_serial} does not match ${expected_serial}." >&2
    exit 1
  fi
done

echo "CALIBRATED_STACK_READY"
report
