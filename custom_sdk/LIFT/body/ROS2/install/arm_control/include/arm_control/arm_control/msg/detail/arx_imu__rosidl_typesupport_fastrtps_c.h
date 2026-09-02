// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from arm_control:msg/ArxImu.idl
// generated code does not contain a copyright notice
#ifndef ARM_CONTROL__MSG__DETAIL__ARX_IMU__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define ARM_CONTROL__MSG__DETAIL__ARX_IMU__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "arm_control/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "arm_control/msg/detail/arx_imu__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
bool cdr_serialize_arm_control__msg__ArxImu(
  const arm_control__msg__ArxImu * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
bool cdr_deserialize_arm_control__msg__ArxImu(
  eprosima::fastcdr::Cdr &,
  arm_control__msg__ArxImu * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
size_t get_serialized_size_arm_control__msg__ArxImu(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
size_t max_serialized_size_arm_control__msg__ArxImu(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
bool cdr_serialize_key_arm_control__msg__ArxImu(
  const arm_control__msg__ArxImu * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
size_t get_serialized_size_key_arm_control__msg__ArxImu(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
size_t max_serialized_size_key_arm_control__msg__ArxImu(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_arm_control
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, arm_control, msg, ArxImu)();

#ifdef __cplusplus
}
#endif

#endif  // ARM_CONTROL__MSG__DETAIL__ARX_IMU__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
