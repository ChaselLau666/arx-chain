#!/bin/bash
# Configuration/validation only. Sourcing this file does not start hardware.
load_tau0vla_robot_profile() {
  local robot_host expected_domain
  robot_host=$(hostname -s)
  case "${robot_host}" in
    ark-1) expected_domain=62 ;;
    ark-2) expected_domain=63 ;;
    *) echo "Refused: no reviewed calibrated robot profile for ${robot_host}." >&2; return 1 ;;
  esac
  if [[ "${ROS_DOMAIN_ID:-}" != "${expected_domain}" ]]; then
    echo "Refused: ${robot_host} requires ROS_DOMAIN_ID=${expected_domain}." >&2
    return 1
  fi
  source "${repo_root}/realsense/camera_serials.sh"
  load_camera_serials dagger
  : "${DIRECT_INTERFACE:=enp130s0}"
  : "${DIRECT_CLIENT_IP:=192.168.50.1}"
  : "${DIRECT_SERVER_IP:=192.168.50.2}"
  : "${LIFT_HEIGHT:=12.5}"
  : "${TAU0VLA_PYTHON:=/home/arx/miniconda3/envs/act/bin/python}"
  export ROS_DOMAIN_ID DIRECT_INTERFACE DIRECT_CLIENT_IP DIRECT_SERVER_IP LIFT_HEIGHT
  export CAMERA_H_SERIAL CAMERA_L_SERIAL CAMERA_R_SERIAL TAU0VLA_PYTHON
}

source_tau0vla_ros() {
  set +u
  source /opt/ros/jazzy/setup.bash
  source /home/arx/LIFT/body/ROS2/install/setup.bash
  set -u
}

load_tau0vla_model_profile() {
  : "${MODEL_PROFILE:=all-blue-feedback}"
  : "${MODEL_VARIANT:=0908}"
  : "${MODEL_SERVER_URL:=http://192.168.50.2:8000}"
  case "${MODEL_VARIANT}" in
    0908|0909) ;;
    *)
      echo "Unknown MODEL_VARIANT=${MODEL_VARIANT}; use 0908 or 0909." >&2
      return 1
      ;;
  esac
  protocol_version=arx-calibrated-v3

  case "${MODEL_PROFILE}" in
    blue-feedback)
      experiment=joint-feedback
      expected_route=arx-lift2s-0907-blue-joint-feedback-ft
      task='Pick up the blue box and place it in its designated position on the board.'
      ;;
    t-feedback)
      experiment=joint-feedback
      expected_route=arx-lift2s-0907-t-joint-feedback-ft
      task='Pick up the T-shaped part and place it in its designated position on the board.'
      ;;
    blue-vr)
      experiment=joint-vr
      expected_route=arx-lift2s-0907-blue-joint-vr-ft
      task='Pick up the blue box and place it in its designated position on the board.'
      ;;
    t-vr)
      experiment=joint-vr
      expected_route=arx-lift2s-0907-t-joint-vr-ft
      task='Pick up the T-shaped part and place it in its designated position on the board.'
      ;;
    all-l-feedback)
      experiment=joint-feedback
      protocol_version=arx-feedback-v4
      expected_route=arx-lift2s-0908-all-joint-feedback-ft
      task='Pick up the L-shaped part and place it in its designated position on the board.'
      ;;
    all-t-feedback)
      experiment=joint-feedback
      protocol_version=arx-feedback-v4
      expected_route=arx-lift2s-0908-all-joint-feedback-ft
      task='Pick up the T-shaped part and place it in its designated position on the board.'
      ;;
    all-banana-feedback)
      experiment=joint-feedback
      protocol_version=arx-feedback-v4
      expected_route=arx-lift2s-0908-all-joint-feedback-ft
      task='Pick up the banana and place it in its designated position on the board.'
      ;;
    all-red-feedback)
      experiment=joint-feedback
      protocol_version=arx-feedback-v4
      expected_route=arx-lift2s-0908-all-joint-feedback-ft
      task='Pick up the red object and place it in its designated position on the board.'
      ;;
    all-blue-feedback)
      experiment=joint-feedback
      protocol_version=arx-feedback-v4
      expected_route=arx-lift2s-0908-all-joint-feedback-ft
      task='Pick up the blue box and place it in its designated position on the board.'
      ;;
    all-circle-feedback)
      if [[ "${MODEL_VARIANT}" == 0909 ]]; then
        echo "MODEL_VARIANT=0909 has no circular-part task; choose all-cylinder-upper-feedback or all-cylinder-lower-feedback." >&2
        return 1
      fi
      experiment=joint-feedback
      protocol_version=arx-feedback-v4
      expected_route=arx-lift2s-0908-all-joint-feedback-ft
      task='Pick up the circular part and place it in its designated position on the board.'
      ;;
    all-cylinder-upper-feedback|all-cylinder-lower-feedback)
      if [[ "${MODEL_VARIANT}" != 0909 ]]; then
        echo "${MODEL_PROFILE} requires MODEL_VARIANT=0909." >&2
        return 1
      fi
      experiment=joint-feedback
      protocol_version=arx-feedback-v4
      expected_route=arx-lift2s-0909-all-joint-feedback-64g50k-ft
      if [[ "${MODEL_PROFILE}" == all-cylinder-upper-feedback ]]; then
        task='Pick up the cylindrical part and place it in the upper hole on the board.'
      else
        task='Pick up the cylindrical part and place it in the lower hole on the board.'
      fi
      ;;
    *)
      echo "Unknown MODEL_PROFILE=${MODEL_PROFILE}." >&2
      echo "Use blue-feedback, t-feedback, blue-vr, t-vr, or all-{l,t,banana,red,blue,circle,cylinder-upper,cylinder-lower}-feedback." >&2
      return 1
      ;;
  esac
  if [[ "${MODEL_VARIANT}" == 0909 ]]; then
    if [[ "${protocol_version}" != arx-feedback-v4 ]]; then
      echo "MODEL_VARIANT=0909 requires an all-{l,t,banana,red,blue,cylinder-upper,cylinder-lower}-feedback profile." >&2
      return 1
    fi
    expected_route=arx-lift2s-0909-all-joint-feedback-64g50k-ft
  fi
  task=${TASK_INSTRUCTION:-${task}}
  # Stack bring-up runs in a child shell and repeats the same route preflight.
  export MODEL_VARIANT

}

check_tau0vla_server() {
  "${TAU0VLA_PYTHON}" "${repo_root}/act/tau0vla_server_preflight.py" \
    --server-url "${MODEL_SERVER_URL}" --route "${expected_route}" \
    --experiment "${experiment}" --protocol-version "${protocol_version}"
}
