#!/bin/bash
set -Eeuo pipefail

# Collection with the VR pose stream low-passed before it reaches the arms.
#
# The only difference from 01_collect.sh is where the arms get their commands:
# a filter node sits between the VR serial node and the arm controllers, so the
# poses that drive inverse kinematics are smooth. Nothing in the ARX SDK is
# modified - the arms are started with ros2 run and told which topic to listen
# to through the arm_sub_topic_name parameter they already expose.
#
# Filtering here rather than on the recorded joint angles is what teleop-app
# does, and for the same reason: the pose is what feeds IK.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# The domain is this robot's identity and must come from the machine
# (/etc/environment). Defaulting it once drove another robot's arms on a shared
# LAN, so this refuses rather than guesses.
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. It lives in /etc/environment (ark-1=62, ark-2=63); refusing to guess which robot to talk to}"
: "${LIFT_HEIGHT:?Set LIFT_HEIGHT to the fixed lift command in [0, 20]}"
: "${TASK_NAME:?Set TASK_NAME, for example pickplace_right_to_bowl}"

SMOOTH_TAU=${SMOOTH_TAU:-0.05}
LOG_DIR=${LOG_DIR:-$HOME/collect_logs}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
LIFT_WS=/home/arx/LIFT/body/ROS2
X5_WS=/home/arx/LIFT/ARX_X5/ROS2/X5_ws
VR_WS=/home/arx/LIFT/ARX_VR_SDK/ROS2
# The realsense build lives in the main checkout; this worktree only carries
# the tracked sources, so point at the built one rather than a missing install.
REALSENSE_WS=${REALSENSE_WS:-/home/arx/ROS2_LIFT_Play/realsense}
CAMERA_PROFILE=${CAMERA_PROFILE:-640x480x90}
FILTERED_L=/ARX_VR_L_filtered
FILTERED_R=/ARX_VR_R_filtered

mkdir -p "$LOG_DIR"
pids=()

die() { echo "Refused: $*" >&2; exit 1; }

cleanup_on_error() {
    local status=$?
    if (( status != 0 )) && (( ${#pids[@]} )); then
        echo "Startup failed; stopping the components this script started." >&2
        kill -INT "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup_on_error EXIT

start_component() {
    local name=$1; shift
    local log="${LOG_DIR}/${name}.log"
    echo "  starting ${name} (log: ${log})"
    setsid "$@" > "$log" 2>&1 < /dev/null &
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
# The VR workspace ships a stale arm_control carrying only PosCmd, so it must
# be sourced before X5: whichever is sourced last wins, and X5Controller aborts
# at startup with an undefined JointControl typesupport symbol if it resolves
# arm_control against the VR copy.
source "${VR_WS}/install/setup.bash"
source "${LIFT_WS}/install/setup.bash"
source "${X5_WS}/install/setup.bash"
source "${REALSENSE_WS}/install/setup.bash"
set -u

# --- preconditions, all refused rather than repaired -------------------------

# Anchored on the executable path, not on any mention of the name: a plain
# substring also matches the shell that happens to have the word in its command
# line, which refuses startup for no reason.
for pattern in '/arx_x5_controller/X5Controller$' \
               '/serial_port_node$' \
               '/act/collect\.py( |$)' \
               '/act/vr_pose_filter\.py( |$)'; do
    if conflicting=$(pgrep -f -- "$pattern" 2>/dev/null); then
        echo "Conflicting processes for ${pattern}:" >&2
        ps -o pid,args -p "$(tr '\n' ',' <<< "$conflicting" | sed 's/,$//')" >&2 || true
        die "another control stack is already running"
    fi
done

for interface in can1 can3 can5; do
    ip link show "$interface" 2>/dev/null | grep -q 'UP' || die "${interface} is not UP"
done

ros2 node list 2>/dev/null | grep -qx '/lift' \
    || die "/lift is not running. Start body only while the platform is at a safe low position."

# --- pin the lift before VR starts ------------------------------------------
# The body must never briefly follow a raw VR height during collection.

height_set=false
for _ in $(seq 1 20); do
    if ros2 param set /lift fixed_height "${LIFT_HEIGHT}" >/dev/null 2>&1; then
        height_set=true
        break
    fi
    sleep 0.5
done
[[ "${height_set}" == true ]] || die "could not set /lift fixed_height"
echo "  /lift fixed_height set to ${LIFT_HEIGHT}"

# --- arms, pointed at the filtered pose stream ------------------------------

echo "WARNING: the arms power up now and may home themselves. Stand clear."

start_arm() {
    local node=$1 can=$2 pub=$3 sub=$4
    start_component "$node" \
        ros2 run arx_x5_controller X5Controller --ros-args \
        -r __node:="$node" \
        -p arm_can_id:="$can" \
        -p arm_control_type:=vr_slave \
        -p arm_end_type:=2 \
        -p arm_pub_topic_name:="$pub" \
        -p arm_sub_topic_name:="$sub"
}

if [[ "${SMOOTH_TAU}" == "0" || "${SMOOTH_TAU}" == "0.0" ]]; then
    echo "  SMOOTH_TAU=0: arms take the raw VR stream, matching 01_collect.sh"
    start_arm vr_arm_l can1 arm_l_status /ARX_VR_L
    start_arm vr_arm_r can3 arm_r_status /ARX_VR_R
else
    start_arm vr_arm_l can1 arm_l_status "${FILTERED_L}"
    start_arm vr_arm_r can3 arm_r_status "${FILTERED_R}"
fi
wait_for_topic /arm_l_status_full 25
wait_for_topic /arm_r_status_full 25

# --- cameras ----------------------------------------------------------------

# realsense.sh opens a gnome-terminal per camera, which cannot work once this
# script is detached from a display. The serials still come from that file, so
# it stays the one place they are configured.
declare -A CAMERA_SERIAL
while read -r name serial; do
    [[ -n "$name" ]] && CAMERA_SERIAL["$name"]="$serial"
done < <(sed -n 's/^ *\[\([a-z_]*\)\]="\([0-9]*\)".*/\1 \2/p' \
         "${repo_root}/realsense/realsense.sh")
(( ${#CAMERA_SERIAL[@]} == 3 )) \
    || die "expected 3 camera serials in realsense/realsense.sh, found ${#CAMERA_SERIAL[@]}"

for camera in camera_h camera_l camera_r; do
    serial=${CAMERA_SERIAL[$camera]:-}
    [[ -n "$serial" ]] || die "no serial configured for ${camera}"
    start_component "$camera" \
        ros2 launch realsense2_camera rs_launch.py \
        camera_name:="$camera" \
        depth_module.color_profile:="${CAMERA_PROFILE}" \
        depth_module.depth_profile:="${CAMERA_PROFILE}" \
        serial_no:="_${serial}"
done
for camera in camera_h camera_l camera_r; do
    wait_for_topic "/camera/${camera}/color/image_rect_raw/compressed" 40
done

# --- VR, then the filter that feeds the arms --------------------------------

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
    echo "  VR poses are low-passed at tau=${SMOOTH_TAU}s before reaching the arms"
fi

trap - EXIT
echo
echo "Stack is up. Components log to ${LOG_DIR}."
echo "Starting the collector in this terminal; Ctrl+C reaches it directly."
echo

cd "${repo_root}/act"
set +u
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate act
set -u
exec python collect.py --episode_idx -1 --height "${LIFT_HEIGHT}" --task "${TASK_NAME}" "$@"
