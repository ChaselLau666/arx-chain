#!/bin/bash
set -Eeuo pipefail

# Teleop only, with the VR pose stream low-passed before it reaches the arms.
#
# This is 06_collect_filtered.sh with the recording removed: no cameras, no
# collector. It exists to answer one question - does teleop actually go through
# the filter - on a machine that has arms and a VR rig and nothing else.
# tools/vr_filter_monitor.py reports the answer while this runs.
#
# WITH_BODY=1 also starts the lift and leaves its height unpinned, so the VR
# stick raises and lowers the platform. Collection pins it instead; see below.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. It lives in /etc/environment; refusing to guess which robot to talk to}"

SMOOTH_TAU=${SMOOTH_TAU:-0.05}
LOG_DIR=${LOG_DIR:-$HOME/teleop_logs}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
WITH_BODY=${WITH_BODY:-0}
LIFT_WS=/home/arx/LIFT/body/ROS2
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
source "${LIFT_WS}/install/setup.bash"
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

# Same one-shot bring-up as 06_collect_filtered.sh, and not arx_can1.sh: the
# repair path in those scripts runs `pkill -9 slcand`, killing the daemon behind
# every other interface, and their success path loops without ever sleeping.
# can1 and can3 carry the arms; can5 carries the lift and is only needed when
# WITH_BODY asks for it.
SKIP_AUTOSTART=${SKIP_AUTOSTART:-0}
declare -A CAN_DEVICE=( [can1]=/dev/arxcan1 [can3]=/dev/arxcan3 [can5]=/dev/arxcan5 )

can_is_up() { ip link show "$1" 2>/dev/null | grep -q 'UP'; }

bring_up_can() {
    # Two statements, not one: local expands all its arguments before it assigns
    # any of them, so ${CAN_DEVICE[$iface]} on the same line reads an unset iface.
    local iface=$1
    local dev=${CAN_DEVICE[$iface]}
    if ! ip link show "$iface" >/dev/null 2>&1; then
        [[ -e "$dev" ]] || die "${dev} is missing; the CAN adapter for ${iface} is unplugged"
        echo "  ${iface}: starting slcand on ${dev}"
        sudo slcand -o -f -s8 "$dev" "$iface" || die "slcand failed for ${iface}"
        for _ in $(seq 1 20); do
            ip link show "$iface" >/dev/null 2>&1 && break
            sleep 0.25
        done
    fi
    sudo ip link set "$iface" up || die "could not bring ${iface} up"
    can_is_up "$iface" || die "${iface} is still not UP after bring-up"
}

can_interfaces=(can1 can3)
(( WITH_BODY )) && can_interfaces+=(can5)
for interface in "${can_interfaces[@]}"; do
    can_is_up "$interface" && continue
    (( SKIP_AUTOSTART )) && die "${interface} is not UP"
    bring_up_can "$interface"
done

# --- body, only when the lift is meant to follow the stick ------------------

if (( WITH_BODY )); then
    lift_is_up() { ros2 node list 2>/dev/null | grep -qx '/lift'; }
    if ! lift_is_up; then
        echo "WARNING: body starts now and the lift may home itself. Stand clear."
        start_component body ros2 launch arx_lift_controller lift.launch.py
        for _ in $(seq 1 60); do
            lift_is_up && break
            sleep 0.5
        done
        lift_is_up || die "/lift did not appear within 30s; see ${LOG_DIR}/body.log"
        wait_for_topic /body_information 20
    fi
    # No fixed_height on purpose. The patched body reads -1.0 as "not pinned" and
    # falls through to the height carried in the VR message, which is what makes
    # the stick work; the filter passes that field through untouched. Collection
    # pins it instead, because a recorded episode needs one known height.
    echo "  lift follows the VR stick - height is NOT pinned"
    echo "  WARNING: the platform moves to the stick's height on the first VR message"
fi

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
