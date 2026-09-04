#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Domain must come from the machine (/etc/environment); see 05_human_dagger.sh.
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is not set. The robot identity lives in /etc/environment (ark-1=62, ark-2=63); refusing to guess which robot to talk to}"

set +u
source /opt/ros/jazzy/setup.bash
source /home/arx/LIFT/body/ROS2/install/setup.bash
set -u

runtime_root=${HUMAN_DAGGER_RUNTIME_DIR:-"${XDG_RUNTIME_DIR:-/tmp}/human_dagger-${UID}"}
active_manifest="${runtime_root}/active.manifest"

proc_start_ticks() {
  local pid=$1 stat rest
  [[ -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r stat < "/proc/${pid}/stat" || return 1
  rest=${stat#*) }
  set -- ${rest}
  [[ $# -ge 20 ]] || return 1
  printf '%s\n' "${20}"
}

manifest_entry_is_live() {
  local pid=$1 expected_ticks=$2 current_ticks
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ "$pid" != "$$" && "$pid" != "$PPID" ]] || return 1
  current_ticks=$(proc_start_ticks "$pid" 2>/dev/null) || return 1
  [[ "$current_ticks" == "$expected_ticks" ]]
}

manifest_has_live_processes() {
  local candidate=$1 label pid ticks
  [[ -f "$candidate" ]] || return 1
  while IFS=$'\t' read -r label pid ticks; do
    [[ -n "$label" && "$label" != \#* ]] || continue
    if manifest_entry_is_live "$pid" "$ticks"; then
      return 0
    fi
  done < "$candidate"
  return 1
}

manifest_matches_this_machine() {
  # A manifest created on another robot, or under another ROS domain, must
  # never drive shutdown here: /lift in that graph is a different machine.
  local candidate=$1 recorded_host recorded_domain
  [[ -f "$candidate" ]] || return 1
  recorded_host=$(sed -n 's/^# hostname=//p' "$candidate" | head -n1)
  recorded_domain=$(sed -n 's/^# ros_domain_id=//p' "$candidate" | head -n1)
  if [[ -z "$recorded_host" || -z "$recorded_domain" ]]; then
    echo "Refused: session manifest lacks hostname/ros_domain_id provenance: ${candidate}" >&2
    echo "It predates cross-robot guarding; verify the stack by hand and use" >&2
    echo "HUMAN_DAGGER_ALLOW_LEGACY_SHUTDOWN=1 for the legacy path instead." >&2
    return 1
  fi
  if [[ "$recorded_host" != "$(hostname)" ]]; then
    echo "Refused: manifest was created on ${recorded_host}; this machine is $(hostname)." >&2
    return 1
  fi
  if [[ "$recorded_domain" != "$ROS_DOMAIN_ID" ]]; then
    echo "Refused: manifest ROS_DOMAIN_ID=${recorded_domain} != current ${ROS_DOMAIN_ID}." >&2
    return 1
  fi
  return 0
}

manifest_has_live_label() {
  local candidate=$1 wanted=$2 label pid ticks
  [[ -f "$candidate" ]] || return 1
  while IFS=$'\t' read -r label pid ticks; do
    [[ "$label" == "$wanted" ]] || continue
    manifest_entry_is_live "$pid" "$ticks" && return 0
  done < "$candidate"
  return 1
}

tracked_session=false
tracked_arm_running=false
resolved_manifest=''
if [[ -e "$active_manifest" || -L "$active_manifest" ]]; then
  resolved_manifest=$(readlink "$active_manifest" 2>/dev/null || printf '%s' "$active_manifest")
  if manifest_has_live_processes "$resolved_manifest" \
    && manifest_matches_this_machine "$resolved_manifest"; then
    tracked_session=true
    if manifest_has_live_label "$resolved_manifest" arm_left \
      || manifest_has_live_label "$resolved_manifest" arm_right; then
      tracked_arm_running=true
    fi
  fi
fi

if [[ "$tracked_session" != true ]]; then
  if [[ "${HUMAN_DAGGER_ALLOW_LEGACY_SHUTDOWN:-0}" != 1 ]]; then
    echo "Refused: no live, verified Human DAgger session manifest is available." >&2
    echo "Broad process matching could stop another developer's stack." >&2
    echo "For an intentional legacy-stack shutdown, inspect the processes and rerun with" >&2
    echo "HUMAN_DAGGER_ALLOW_LEGACY_SHUTDOWN=1." >&2
    exit 1
  fi
  legacy_candidates=$(ps -eo pid,ppid,pgid,args | grep -E \
    '(lift_controller|X5Controller|serial_port_node|realsense2_camera_node|collect.py|human_dagger.py|vr_pose_filter.py)' \
    | grep -v grep || true)
  echo "LEGACY SHUTDOWN OPT-IN: these broad-matched processes may be stopped:"
  printf '%s\n' "${legacy_candidates:-<none found>}"
  read -r -p 'Type CONFIRM LEGACY PIDS to continue: ' legacy_confirmation
  if [[ "$legacy_confirmation" != "CONFIRM LEGACY PIDS" ]]; then
    echo "Cancelled; no HOLD, height command, or signal sent."
    exit 1
  fi
fi

dagger_process_running=false
if pgrep -f '[p]ython(3)? .*[/]human_dagger.py' >/dev/null 2>&1; then
  dagger_process_running=true
fi

echo "Requesting Human DAgger HOLD before any lift command..."
if timeout 3 ros2 service list 2>/dev/null | grep -qx '/human_dagger/request_hold'; then
  if ! hold_output=$(timeout 6 ros2 service call \
    /human_dagger/request_hold std_srvs/srv/Trigger '{}' 2>&1); then
    echo "$hold_output" >&2
    echo "Refused: Human DAgger HOLD service did not respond." >&2
    exit 1
  fi
  echo "$hold_output"
  if ! grep -Eiq 'success[=:][[:space:]]*(true|True)' <<< "$hold_output"; then
    echo "Refused: Human DAgger did not acknowledge HOLD." >&2
    exit 1
  fi
  echo "Human DAgger HOLD acknowledged."
elif [[ "$dagger_process_running" == true || "$tracked_arm_running" == true ]]; then
  if [[ "$tracked_arm_running" == true ]]; then
    echo "Refused: a verified tracked arm is still running, but /human_dagger/request_hold is unavailable." >&2
  else
    echo "Refused: human_dagger.py is running but /human_dagger/request_hold is unavailable." >&2
  fi
  echo "Use the physical emergency stop if arm motion is not already stopped." >&2
  exit 1
else
  echo "Human DAgger service is not active; continuing with legacy-stack shutdown checks."
fi

if ros2 node list 2>/dev/null | grep -qx '/lift'; then
  if [[ "$tracked_session" == true ]] \
    && ! manifest_has_live_label "$resolved_manifest" body; then
    echo "Refused: /lift is visible, but it is not the live body PID recorded by this session." >&2
    echo "This may be another developer's stack; no height command was sent." >&2
    exit 1
  fi
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
  echo "WARNING: an active episode will be closed as partial/quarantined data."
  read -r -p 'Type LOWER AND SHUTDOWN to continue: ' confirmation
  if [[ "${confirmation}" != "LOWER AND SHUTDOWN" ]]; then
    echo "Cancelled; no command or signal sent."
    exit 1
  fi

  ros2 param set /lift fixed_height 0.0
  # Fallback for an older already-running binary whose fixed_height was applied
  # only from VR/joy callbacks. This command also requests zero wheel/body motion.
  body_control_topic=/body_control
  if ros2 topic info /human_dagger/body/control 2>/dev/null \
    | grep -Eq 'Subscription count:[[:space:]]*[1-9]'; then
    body_control_topic=/human_dagger/body/control
  fi
  ros2 topic pub --once "$body_control_topic" arm_control/msg/PosCmd \
    '{height: 0.0, mode1: 2}'

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
  echo "WARNING: an active episode will be closed as partial/quarantined data."
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

stop_manifest_pid() {
  local label=$1 pid=$2 ticks=$3 pgid signal_target
  if ! manifest_entry_is_live "$pid" "$ticks"; then
    echo "${label}: tracked PID is no longer the same process; skipped"
    return 0
  fi

  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  signal_target=$pid
  if [[ "$pgid" == "$pid" ]]; then
    signal_target="-${pid}"
  fi

  echo "${label}: verified session PID ${pid}, SIGINT"
  kill -INT -- "$signal_target" 2>/dev/null || true
  for _ in $(seq 1 30); do
    sleep 0.2
    if ! manifest_entry_is_live "$pid" "$ticks"; then
      return 0
    fi
  done

  echo "${label}: verified session PID ${pid}, SIGTERM"
  kill -TERM -- "$signal_target" 2>/dev/null || true
}

stop_active_manifest() {
  local resolved_manifest label pid ticks index
  local -a entries=()

  if [[ ! -e "$active_manifest" && ! -L "$active_manifest" ]]; then
    echo "Human DAgger session manifest: not present"
    return 0
  fi
  resolved_manifest=$(readlink "$active_manifest" 2>/dev/null || printf '%s' "$active_manifest")
  if [[ ! -f "$resolved_manifest" ]]; then
    echo "Human DAgger session manifest is stale: ${resolved_manifest}"
    rm -f -- "$active_manifest"
    return 0
  fi

  while IFS= read -r line; do
    [[ -n "$line" && "$line" != \#* ]] && entries+=("$line")
  done < "$resolved_manifest"

  # Reverse startup order. Historical manifests may contain vendor CAN
  # watchdog shells; stop those shells as well while leaving the configured
  # CAN interfaces/slcand transport itself UP.
  for ((index=${#entries[@]} - 1; index >= 0; index--)); do
    IFS=$'\t' read -r label pid ticks <<< "${entries[index]}"
    stop_manifest_pid "$label" "$pid" "$ticks"
  done

  for line in "${entries[@]}"; do
    IFS=$'\t' read -r label pid ticks <<< "$line"
    if manifest_entry_is_live "$pid" "$ticks"; then
      echo "WARNING: verified session process remains: ${label} (${pid})" >&2
      return 1
    fi
  done

  printf '# stopped_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$resolved_manifest"
  printf '# retained_can_interfaces=can1,can3,can5\n' >> "$resolved_manifest"
  if [[ "$(readlink "$active_manifest" 2>/dev/null || true)" == "$resolved_manifest" ]]; then
    rm -f -- "$active_manifest"
  fi
  echo "Human DAgger session processes stopped from verified manifest."
}

if [[ "$tracked_session" == true ]]; then
  # The manifest records PID plus /proc start time. Stop only this developer's
  # verified session; never broad-match another user's robot processes.
  stop_active_manifest
else
  echo "No verified Human DAgger manifest; using the legacy stack shutdown path."
  stop_pattern "Human DAgger frontend" '[p]ython(3)? .*[/]human_dagger.py'
  stop_pattern "collector" '[p]ython .*collect.py'
  stop_pattern "inference" '[p]ython .*inference.py'
  stop_pattern "replay" '[p]ython .*replay.py'
  stop_pattern "VR diagnostics" 'ros2 topic (echo|hz) /ARX_VR_[LR]'
  stop_pattern "VR serial launcher" '/opt/ros/jazzy/bin/ros2 run serial_port serial_port_node'
  stop_pattern "VR serial node" '/serial_port_node$'
  # Between the VR serial node and the arms, so it is stopped with the stream
  # it filters. Left out, it survives shutdown and the next launch refuses.
  stop_pattern "VR pose filter" '[/]act/vr_pose_filter\.py'
  stop_pattern "RealSense launchers" '/opt/ros/jazzy/bin/ros2 launch realsense2_camera rs_launch.py'
  stop_pattern "RealSense nodes" '/realsense2_camera_node'
  stop_pattern "arm launcher" '/opt/ros/jazzy/bin/ros2 launch arx_x5_controller (v2_pos_control|v2_joint_control|open_double_arm).launch.py'
  stop_pattern "arm nodes" '/arx_x5_controller/X5Controller'
  stop_pattern "body launcher" '/opt/ros/jazzy/bin/ros2 launch arx_lift_controller lift.launch.py'
  stop_pattern "body node" '/arx_lift_controller/lift_controller'
  stop_pattern "CAN watchdog shells" '/bin/bash ./arx_can[135].sh'
fi
echo "Verifying control processes..."
remaining=$(ps -eo pid,args | grep -E \
  '(lift_controller|X5Controller|serial_port_node|realsense2_camera_node|collect.py|inference.py|human_dagger.py|human_dagger_arm_(left|right)|arx_can[135].sh)' \
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
