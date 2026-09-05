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
  case "$camera_host" in
    ark-1)
      head_serial=260422272688
      left_serial=260422274927
      right_serial=260522274175
      ;;
    # This robot reports its hostname as "arx", not ark-N, and carries a third
    # set of D405s. Identified by the last four digits of the asic serial that
    # rs-enumerate-devices reports beside each device serial: head 6400,
    # left 8906, right 4386. Both profiles share these three: there is no spare
    # head camera on this robot to switch to.
    arx)
      head_serial=262622273173
      left_serial=262622270575
      right_serial=262422271983
      ;;
  esac
  CAMERA_H_SERIAL=${CAMERA_H_SERIAL:-$head_serial}
  CAMERA_L_SERIAL=${CAMERA_L_SERIAL:-$left_serial}
  CAMERA_R_SERIAL=${CAMERA_R_SERIAL:-$right_serial}
}
