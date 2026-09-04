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
# Rerunnable on purpose. Tuning the IK node means restarting it many times
# with different flags, so a healthy stack that is already up - arms in
# remote_slave, VR serial, pose filters - is reused, and only the IK node(s)
# are restarted with the current settings. What is refused is the wrong kind
# of stack: arms in vr_slave from 07, a collector, or a partial arm pair.
#
# IK_SIDE picks which arm(s) get an IK node; the other arm sits in
# remote_slave with nothing commanding it and simply holds.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. It lives in /etc/environment; refusing to guess which robot to talk to}"

SMOOTH_TAU=${SMOOTH_TAU:-0.05}
IK_SIDE=${IK_SIDE:-right}                    # right | left | both
IK_DRY_RUN=${IK_DRY_RUN:-0}                   # 1: solve and report, publish nothing
IK_ENGAGE_MM=${IK_ENGAGE_MM:-50}              # target must come this close to the arm to engage
IK_ENGAGE_DEG=${IK_ENGAGE_DEG:-20}
IK_MAX_VEL=${IK_MAX_VEL:-1.5}                 # rad/s per joint
IK_MAX_RESIDUAL_MM=${IK_MAX_RESIDUAL_MM:-30}  # above this the target is treated as unreachable
IK_DT=${IK_DT:-0.01}                          # solver step, about the VR message period
LOG_DIR=${LOG_DIR:-$HOME/teleop_logs}
ACT_PYTHON=${ACT_PYTHON:-/home/arx/miniconda3/envs/act/bin/python}
X5_WS=/home/arx/LIFT/ARX_X5/ROS2/X5_ws
VR_WS=/home/arx/LIFT/ARX_VR_SDK/ROS2
FILTERED_L=/ARX_VR_L_filtered
FILTERED_R=/ARX_VR_R_filtered

case "$IK_SIDE" in right|left|both) ;; *) echo "IK_SIDE must be right, left or both" >&2; exit 1;; esac
if [[ "${SMOOTH_TAU}" == "0" || "${SMOOTH_TAU}" == "0.0" ]]; then
    echo "SMOOTH_TAU=0 is not supported here: vr_ik_node reads the filtered topics" >&2; exit 1
fi

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

topic_up() { ros2 topic list 2>/dev/null | grep -qx "$1"; }

wait_for_topic() {
    local topic=$1 timeout=${2:-20}
    for _ in $(seq 1 $((timeout * 2))); do
        topic_up "$topic" && return 0
        sleep 0.5
    done
    die "${topic} did not appear within ${timeout}s; see ${LOG_DIR}"
}

# Stop matching processes gently, then firmly. The vr_slave binaries have been
# seen to ignore SIGINT and SIGTERM after losing their input, so this always
# ends in SIGKILL if needed rather than assuming the signal worked.
stop_matching() {
    local label=$1 pattern=$2 found
    found=$(pgrep -f -- "$pattern" 2>/dev/null || true)
    [[ -n "$found" ]] || return 0
    echo "  stopping ${label}: ${found//$'\n'/ }"
    kill -INT $found 2>/dev/null || true
    for _ in $(seq 1 10); do sleep 0.3; pgrep -f -- "$pattern" >/dev/null 2>&1 || return 0; done
    kill -TERM $found 2>/dev/null || true
    for _ in $(seq 1 6); do sleep 0.3; pgrep -f -- "$pattern" >/dev/null 2>&1 || return 0; done
    kill -KILL $found 2>/dev/null || true
    sleep 0.5
    pgrep -f -- "$pattern" >/dev/null 2>&1 && die "${label} would not stop"
    return 0
}

set +u
source /opt/ros/jazzy/setup.bash
# VR before X5: the VR workspace ships a stale arm_control with only PosCmd,
# and whichever is sourced last wins. See 07_teleop_filtered.sh.
source "${VR_WS}/install/setup.bash"
source "${X5_WS}/install/setup.bash"
set -u

"$ACT_PYTHON" -c 'import placo' 2>/dev/null || die "placo is not installed in ${ACT_PYTHON}: pip install placo"

# --- what can never be reused -------------------------------------------------

if conflicting=$(pgrep -f -- 'X5Controller.*arm_control_type:=vr_slave' 2>/dev/null); then
    echo "Arms are running in vr_slave mode (from 07_teleop_filtered.sh):" >&2
    ps -o pid,args -p "$(tr '\n' ',' <<< "$conflicting" | sed 's/,$//')" >&2 || true
    die "stop that stack first; this script needs the arms in remote_slave"
fi
if pgrep -f -- '/act/collect\.py( |$)' >/dev/null 2>&1; then
    die "a collector is running; stop it before restarting teleop underneath it"
fi

# --- CAN, one-shot -----------------------------------------------------------

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

# --- arms: launch when absent, reuse when complete, refuse when partial ------

arm_topics=0
for t in /arm_slave_l_status /arm_slave_r_status; do topic_up "$t" && arm_topics=$((arm_topics + 1)); done
if (( arm_topics == 2 )); then
    echo "  reusing the running remote_slave arm stack"
elif (( arm_topics == 0 )); then
    echo "WARNING: the arms power up now and may home themselves. Stand clear."
    start_component arms ros2 launch arx_x5_controller v2_joint_control.launch.py
    wait_for_topic /arm_slave_l_status 25
    wait_for_topic /arm_slave_r_status 25
else
    die "arm stack is partial (${arm_topics}/2 feedback topics); stop it and rerun"
fi

# --- VR serial and the pose filter: reuse or start --------------------------

# Alive means a message arrived just now, not that the topic is listed: a
# serial node whose headset went to sleep, or whose USB port reset, keeps its
# topics advertised while publishing nothing. Starting a second node beside
# it makes things worse - two processes contend for the same tty and neither
# delivers - which is exactly how the VR stream died once. So a stale node is
# stopped and replaced rather than joined.
vr_alive() { timeout 3 ros2 topic echo --once "$1" >/dev/null 2>&1; }

if vr_alive /ARX_VR_R; then
    echo "  reusing the running VR serial node"
else
    stop_matching "stale VR serial node(s)" \
        '^[^ ]*python[0-9.]* [^ ]*/ros2 run serial_port serial_port_node( |$)|^[^ ]*/serial_port_node( |$)'
    start_component vr_serial ros2 run serial_port serial_port_node
    wait_for_topic /ARX_VR_L 25
    wait_for_topic /ARX_VR_R 25
    vr_alive /ARX_VR_R || die "serial_port_node is up but publishes nothing; is the headset on and its USB cable in? see ${LOG_DIR}/vr_serial.log"
fi

if topic_up "$FILTERED_L" && topic_up "$FILTERED_R"; then
    echo "  reusing the running pose filters (their tau is whatever they were started with, not necessarily ${SMOOTH_TAU})"
else
    start_component vr_filter_l "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
        --in-topic /ARX_VR_L --out-topic "${FILTERED_L}" --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_l
    start_component vr_filter_r "$ACT_PYTHON" "${repo_root}/act/vr_pose_filter.py" \
        --in-topic /ARX_VR_R --out-topic "${FILTERED_R}" --tau "${SMOOTH_TAU}" --node-name vr_pose_filter_r
    wait_for_topic "${FILTERED_L}" 20
    wait_for_topic "${FILTERED_R}" 20
fi

# --- IK nodes: always restarted, so the flags on this command line apply ----

# Anchored on a python interpreter as argv[0] and the script as argv[1]. A
# looser pattern also matches any shell whose command line merely mentions
# the script - such as the ssh session that ran this launcher during
# testing, which it promptly killed. argv[1] only has to end in the script
# name, so a node started by hand from inside act/ with a bare relative
# path is stopped too; leaving one behind would mean two publishers on the
# arm's command topic.
stop_matching "previous IK node(s)" '^[^ ]*python[0-9.]* [^ ]*vr_ik_node\.py( |$)'

ik_args=(--dt "${IK_DT}" --max-velocity "${IK_MAX_VEL}"
         --engage-distance "$(awk "BEGIN{print ${IK_ENGAGE_MM}/1000}")" --engage-angle "${IK_ENGAGE_DEG}"
         --max-residual "$(awk "BEGIN{print ${IK_MAX_RESIDUAL_MM}/1000}")")
(( IK_DRY_RUN )) || ik_args+=(--execute)

sides=()
case "$IK_SIDE" in right) sides=(right);; left) sides=(left);; both) sides=(left right);; esac
for side in "${sides[@]}"; do
    s=${side:0:1}
    start_component "vr_ik_${s}" "$ACT_PYTHON" "${repo_root}/act/vr_ik_node.py" --side "$side" "${ik_args[@]}"
done
sleep 1
for side in "${sides[@]}"; do
    s=${side:0:1}
    pgrep -f -- "vr_ik_node\.py --side ${side}" >/dev/null 2>&1 || die "vr_ik_${s} exited at once; see ${LOG_DIR}/vr_ik_${s}.log"
done

trap - EXIT
echo
if (( IK_DRY_RUN )); then
    echo "IK is in DRY-RUN: solving and reporting, publishing nothing. The arms hold."
else
    echo "IK is live for the ${IK_SIDE} arm(s). Nothing moves until the VR target comes within"
    echo "${IK_ENGAGE_MM} mm / ${IK_ENGAGE_DEG} deg of where the arm is; the log says how far it is."
fi
echo "  tail -f ${LOG_DIR}/vr_ik_${sides[0]:0:1}.log"
echo
echo "Rerun this script to restart the IK node(s) with different settings; the rest is reused."
echo "Joint targets are on /arm_master_{l,r}_status. Stop just the IK node(s) with:"
echo "  pkill -INT -f '/act/vr_ik_node.py'"
