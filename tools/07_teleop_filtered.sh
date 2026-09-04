#!/bin/bash
set -Eeuo pipefail

# Teleop only, with the VR pose stream low-passed before it reaches the arms.
#
# This is 06_collect_filtered.sh with the recording removed: no cameras, no
# collector, no lift. It exists to answer one question - does teleop actually
# go through the filter - on a machine that has arms and a VR rig and nothing
# else. tools/vr_filter_monitor.py reports the answer while this runs.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. It lives in /etc/environment; refusing to guess which robot to talk to}"

SMOOTH_TAU=${SMOOTH_TAU:-0.05}
LOG_DIR=${LOG_DIR:-$HOME/teleop_logs}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
X5_WS=/home/arx/LIFT/ARX_X5/ROS2/X5_ws
VR_WS=/home/arx/LIFT/ARX_VR_SDK/ROS2
FILTERED_L=/ARX_VR_L_filtered
FILTERED_R=/ARX_VR_R_filtered

mkdir -p "$LOG_DIR"
pids=()

die() { echo "Refused: $*" >&2; exit 1; }

cleanup_on_error() {
    local status=$?
    if (( status != 0 )) && (( ${#pids[@]} )); then
        echo "Startup failed; stopping what this script started." >&2
        kill -INT "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup_on_error EXIT

start_component() {
    local name=$1; shift
    echo "  starting ${name} (log: ${LOG_DIR}/${name}.log)"
    setsid "$@" > "${LOG_DIR}/${name}.log" 2>&1 < /dev/null &
    pids+=("$!")
}

wait_for_topic() {
    local topic=$1 timeout=${2:-20}
    for _ in $(seq 1 $((timeout * 2))); do
        ros2 topic list 2>/dev/null | grep -qx "$topic" && return 0
        sleep 0.5
    done
    die "${topic} did not appear within ${timeout}s; see ${LOG_DIR}"
}

set +u
source /opt/ros/jazzy/setup.bash
# The VR workspace ships a stale arm_control carrying only PosCmd, so it must be
# sourced before X5: whichever is sourced last wins, and X5Controller aborts at
# startup with an undefined JointControl typesupport symbol otherwise.
source "${VR_WS}/install/setup.bash"
source "${X5_WS}/install/setup.bash"
set -u

for pattern in '/arx_x5_controller/X5Controller( |$)' '/serial_port_node( |$)' \
               '/act/vr_pose_filter\.py( |$)'; do
    if conflicting=$(pgrep -f -- "$pattern" 2>/dev/null); then
        echo "Conflicting processes for ${pattern}:" >&2
        ps -o pid,args -p "$(tr '\n' ',' <<< "$conflicting" | sed 's/,$//')" >&2 || true
        die "another stack is already running"
    fi
done

for interface in can1 can3; do
    ip link show "$interface" 2>/dev/null | grep -q 'UP' || die "${interface} is not UP"
done

# --- arms, pointed at whichever stream this run is testing -------------------

echo "WARNING: the arms power up now and may home themselves. Stand clear."

start_arm() {
    start_component "$1" \
        ros2 run arx_x5_controller X5Controller --ros-args \
        -r __node:="$1" -p arm_can_id:="$2" -p arm_control_type:=vr_slave \
        -p arm_end_type:=2 -p arm_pub_topic_name:="$3" -p arm_sub_topic_name:="$4"
}

if [[ "${SMOOTH_TAU}" == "0" || "${SMOOTH_TAU}" == "0.0" ]]; then
    echo "  SMOOTH_TAU=0: arms take the raw VR stream"
    start_arm vr_arm_l can1 arm_l_status /ARX_VR_L
    start_arm vr_arm_r can3 arm_r_status /ARX_VR_R
else
    start_arm vr_arm_l can1 arm_l_status "${FILTERED_L}"
    start_arm vr_arm_r can3 arm_r_status "${FILTERED_R}"
fi
wait_for_topic /arm_l_status_full 25
wait_for_topic /arm_r_status_full 25

# --- VR, then the filter ----------------------------------------------------

start_component vr_serial ros2 run serial_port serial_port_node
wait_for_topic /ARX_VR_L 25
wait_for_topic /ARX_VR_R 25

if [[ "${SMOOTH_TAU}" != "0" && "${SMOOTH_TAU}" != "0.0" ]]; then
    start_component vr_filter_l "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
        --in-topic /ARX_VR_L --out-topic "${FILTERED_L}" \
        --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_l
    start_component vr_filter_r "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
        --in-topic /ARX_VR_R --out-topic "${FILTERED_R}" \
        --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_r
    wait_for_topic "${FILTERED_L}" 20
    wait_for_topic "${FILTERED_R}" 20
fi

trap - EXIT
echo
echo "Teleop is up. Logs in ${LOG_DIR}."
if [[ "${SMOOTH_TAU}" != "0" && "${SMOOTH_TAU}" != "0.0" ]]; then
    echo "The arms are following ${FILTERED_L} / ${FILTERED_R} at tau=${SMOOTH_TAU}s."
else
    echo "The arms are following the raw VR stream."
fi
echo
echo "Move the VR controllers, then in another terminal:"
echo "  ${ACT_PYTHON} ${repo_root}/tools/vr_filter_monitor.py"
echo
echo "Stop everything with:"
echo "  kill -INT ${pids[*]}"
