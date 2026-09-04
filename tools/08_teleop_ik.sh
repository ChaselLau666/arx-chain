#!/bin/bash
set -Eeuo pipefail

# Teleop with IK solved on the host instead of inside the arm controller.
#
# 07_teleop_filtered.sh runs the arms in vr_slave mode: they take the VR pose
# and solve IK in a closed library, so the joint targets they follow never
# appear on a topic. This script runs the arms in remote_slave mode - the
# joint-command mode replay already uses - and puts act/vr_ik_node.py between
# the filtered VR pose and the arm. The joint target then exists as a message
# on /arm_master_{l,r}_status, which is what makes it recordable as a true
# action later.
#
# Everything else matches 07: same CAN bring-up, same VR serial node, same
# pose filter. IK_SIDE picks which arm(s) get an IK node; the other arm sits
# in remote_slave with nothing commanding it and simply holds.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. It lives in /etc/environment; refusing to guess which robot to talk to}"

SMOOTH_TAU=${SMOOTH_TAU:-0.05}
IK_SIDE=${IK_SIDE:-right}          # right | left | both
IK_DRY_RUN=${IK_DRY_RUN:-0}         # 1: solve and report, publish nothing
IK_MAX_STEP=${IK_MAX_STEP:-0.06}    # rad per VR message, per joint
IK_DT=${IK_DT:-0.01}                # solver step, about the VR message period
LOG_DIR=${LOG_DIR:-$HOME/teleop_logs}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
X5_WS=/home/arx/LIFT/ARX_X5/ROS2/X5_ws
VR_WS=/home/arx/LIFT/ARX_VR_SDK/ROS2
FILTERED_L=/ARX_VR_L_filtered
FILTERED_R=/ARX_VR_R_filtered

case "$IK_SIDE" in right|left|both) ;; *) echo "IK_SIDE must be right, left or both" >&2; exit 1;; esac

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
# VR before X5: the VR workspace ships a stale arm_control with only PosCmd,
# and whichever is sourced last wins. See 07_teleop_filtered.sh.
source "${VR_WS}/install/setup.bash"
source "${X5_WS}/install/setup.bash"
set -u

"$ACT_PYTHON" -c 'import placo' 2>/dev/null || die "placo is not installed in ${ACT_PYTHON}: pip install placo"

for pattern in '/arx_x5_controller/X5Controller( |$)' '/serial_port_node( |$)' \
               '/act/vr_pose_filter\.py( |$)' '/act/vr_ik_node\.py( |$)'; do
    if conflicting=$(pgrep -f -- "$pattern" 2>/dev/null); then
        echo "Conflicting processes for ${pattern}:" >&2
        ps -o pid,args -p "$(tr '\n' ',' <<< "$conflicting" | sed 's/,$//')" >&2 || true
        die "another stack is already running"
    fi
done

# One-shot CAN bring-up, same as 06/07 and for the same reason: arx_can1.sh's
# repair path runs `pkill -9 slcand`, which takes down every other interface.
SKIP_AUTOSTART=${SKIP_AUTOSTART:-0}
declare -A CAN_DEVICE=( [can1]=/dev/arxcan1 [can3]=/dev/arxcan3 )

can_is_up() { ip link show "$1" 2>/dev/null | grep -q 'UP'; }

bring_up_can() {
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

for interface in can1 can3; do
    can_is_up "$interface" && continue
    (( SKIP_AUTOSTART )) && die "${interface} is not UP"
    bring_up_can "$interface"
done

# --- arms, in the joint-command mode replay uses -----------------------------

echo "WARNING: the arms power up now and may home themselves. Stand clear."
start_component arms ros2 launch arx_x5_controller v2_joint_control.launch.py
wait_for_topic /arm_slave_l_status 25
wait_for_topic /arm_slave_r_status 25

# --- VR, the filter, then IK ------------------------------------------------

start_component vr_serial ros2 run serial_port serial_port_node
wait_for_topic /ARX_VR_L 25
wait_for_topic /ARX_VR_R 25

if [[ "${SMOOTH_TAU}" == "0" || "${SMOOTH_TAU}" == "0.0" ]]; then
    die "SMOOTH_TAU=0 is not supported here: vr_ik_node reads the filtered topics"
fi
start_component vr_filter_l "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
    --in-topic /ARX_VR_L --out-topic "${FILTERED_L}" --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_l
start_component vr_filter_r "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
    --in-topic /ARX_VR_R --out-topic "${FILTERED_R}" --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_r
wait_for_topic "${FILTERED_L}" 20
wait_for_topic "${FILTERED_R}" 20

ik_args=(--dt "${IK_DT}" --max-step "${IK_MAX_STEP}")
(( IK_DRY_RUN )) || ik_args+=(--execute)

sides=()
case "$IK_SIDE" in right) sides=(right);; left) sides=(left);; both) sides=(left right);; esac
for side in "${sides[@]}"; do
    s=${side:0:1}
    echo "WARNING: the ${side} arm will start moving toward the VR pose once vr_ik_${s} engages."
    start_component "vr_ik_${s}" "$ACT_PYTHON" "${repo_root}/act/vr_ik_node.py" --side "$side" "${ik_args[@]}"
    (( IK_DRY_RUN )) || wait_for_topic "/arm_master_${s}_status" 20
done

trap - EXIT
echo
if (( IK_DRY_RUN )); then
    echo "Teleop is up in IK DRY-RUN: the solver runs and reports, the arms hold."
else
    echo "Teleop is up. ${IK_SIDE} arm(s) follow the VR pose through host-side IK."
fi
echo "Logs in ${LOG_DIR}; watch vr_ik_${sides[0]:0:1}.log for rate, residual and clamps."
echo
echo "Joint targets are now on /arm_master_{l,r}_status - a recordable action:"
echo "  ros2 topic hz /arm_master_${sides[0]:0:1}_status"
echo
echo "Stop everything with:"
echo "  kill -INT ${pids[*]}"
