// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice

#include "arm_control/msg/detail/joint_control__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_arm_control
const rosidl_type_hash_t *
arm_control__msg__JointControl__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x07, 0xec, 0x28, 0x71, 0x15, 0xcb, 0xd0, 0x29,
      0x6d, 0x66, 0xe3, 0x6c, 0xe1, 0x88, 0x48, 0x5a,
      0xc0, 0x55, 0xaf, 0xd7, 0xff, 0x00, 0x6b, 0x78,
      0x03, 0x57, 0x1a, 0x3d, 0x15, 0x63, 0x67, 0xf6,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char arm_control__msg__JointControl__TYPE_NAME[] = "arm_control/msg/JointControl";

// Define type names, field names, and default values
static char arm_control__msg__JointControl__FIELD_NAME__joint_pos[] = "joint_pos";
static char arm_control__msg__JointControl__FIELD_NAME__joint_vel[] = "joint_vel";
static char arm_control__msg__JointControl__FIELD_NAME__joint_cur[] = "joint_cur";
static char arm_control__msg__JointControl__FIELD_NAME__mode[] = "mode";

static rosidl_runtime_c__type_description__Field arm_control__msg__JointControl__FIELDS[] = {
  {
    {arm_control__msg__JointControl__FIELD_NAME__joint_pos, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT_ARRAY,
      8,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__JointControl__FIELD_NAME__joint_vel, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT_ARRAY,
      8,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__JointControl__FIELD_NAME__joint_cur, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT_ARRAY,
      8,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__JointControl__FIELD_NAME__mode, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
arm_control__msg__JointControl__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {arm_control__msg__JointControl__TYPE_NAME, 28, 28},
      {arm_control__msg__JointControl__FIELDS, 4, 4},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "float32[8] joint_pos\n"
  "float32[8] joint_vel\n"
  "float32[8] joint_cur\n"
  "int32 mode";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
arm_control__msg__JointControl__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {arm_control__msg__JointControl__TYPE_NAME, 28, 28},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 73, 73},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
arm_control__msg__JointControl__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *arm_control__msg__JointControl__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
