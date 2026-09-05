#!/usr/bin/env bash
# Configuration only: sourcing this file never starts cameras or ROS nodes.
load_camera_serials() {
  local profile=${1:-standard} camera_host
  local head_serial=260422273990 left_serial=260422273222 right_serial=260422272473
  case "$profile" in
    standard) ;;
    # Preserve the existing DAgger head-camera replacement on ark-2.
    dagger) head_serial=260522275257 ;;
    *) echo "Unknown camera profile: $profile" >&2; return 1 ;;
  esac
  camera_host=$(hostname -s) || return
  if [[ "$camera_host" == "ark-1" ]]; then
    head_serial=260422272688
    left_serial=260422274927
    right_serial=260522274175
  fi
  CAMERA_H_SERIAL=${CAMERA_H_SERIAL:-$head_serial}
  CAMERA_L_SERIAL=${CAMERA_L_SERIAL:-$left_serial}
  CAMERA_R_SERIAL=${CAMERA_R_SERIAL:-$right_serial}
}
