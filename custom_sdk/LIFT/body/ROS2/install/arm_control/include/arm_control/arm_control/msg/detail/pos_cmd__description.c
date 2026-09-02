// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from arm_control:msg/PosCmd.idl
// generated code does not contain a copyright notice

#include "arm_control/msg/detail/pos_cmd__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_arm_control
const rosidl_type_hash_t *
arm_control__msg__PosCmd__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x88, 0x41, 0xbb, 0x1a, 0xe0, 0xbe, 0xf7, 0x9c,
      0x92, 0xa3, 0x4a, 0x17, 0xe6, 0x29, 0x4c, 0x5e,
      0xc5, 0xb1, 0x71, 0xdb, 0x6c, 0xb4, 0x9a, 0x50,
      0xaa, 0x8c, 0xc5, 0x7e, 0xaa, 0xdc, 0xf1, 0xde,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char arm_control__msg__PosCmd__TYPE_NAME[] = "arm_control/msg/PosCmd";

// Define type names, field names, and default values
static char arm_control__msg__PosCmd__FIELD_NAME__x[] = "x";
static char arm_control__msg__PosCmd__FIELD_NAME__y[] = "y";
static char arm_control__msg__PosCmd__FIELD_NAME__z[] = "z";
static char arm_control__msg__PosCmd__FIELD_NAME__roll[] = "roll";
static char arm_control__msg__PosCmd__FIELD_NAME__pitch[] = "pitch";
static char arm_control__msg__PosCmd__FIELD_NAME__yaw[] = "yaw";
static char arm_control__msg__PosCmd__FIELD_NAME__gripper[] = "gripper";
static char arm_control__msg__PosCmd__FIELD_NAME__quater_x[] = "quater_x";
static char arm_control__msg__PosCmd__FIELD_NAME__quater_y[] = "quater_y";
static char arm_control__msg__PosCmd__FIELD_NAME__quater_z[] = "quater_z";
static char arm_control__msg__PosCmd__FIELD_NAME__quater_w[] = "quater_w";
static char arm_control__msg__PosCmd__FIELD_NAME__chx[] = "chx";
static char arm_control__msg__PosCmd__FIELD_NAME__chy[] = "chy";
static char arm_control__msg__PosCmd__FIELD_NAME__chz[] = "chz";
static char arm_control__msg__PosCmd__FIELD_NAME__vel_l[] = "vel_l";
static char arm_control__msg__PosCmd__FIELD_NAME__vel_r[] = "vel_r";
static char arm_control__msg__PosCmd__FIELD_NAME__height[] = "height";
static char arm_control__msg__PosCmd__FIELD_NAME__head_pit[] = "head_pit";
static char arm_control__msg__PosCmd__FIELD_NAME__head_yaw[] = "head_yaw";
static char arm_control__msg__PosCmd__FIELD_NAME__temp_float_data[] = "temp_float_data";
static char arm_control__msg__PosCmd__FIELD_NAME__temp_int_data[] = "temp_int_data";
static char arm_control__msg__PosCmd__FIELD_NAME__mode1[] = "mode1";
static char arm_control__msg__PosCmd__FIELD_NAME__mode2[] = "mode2";
static char arm_control__msg__PosCmd__FIELD_NAME__time_count[] = "time_count";

static rosidl_runtime_c__type_description__Field arm_control__msg__PosCmd__FIELDS[] = {
  {
    {arm_control__msg__PosCmd__FIELD_NAME__x, 1, 1},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__y, 1, 1},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__z, 1, 1},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__roll, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__pitch, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__yaw, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__gripper, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__quater_x, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__quater_y, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__quater_z, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__quater_w, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__chx, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__chy, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__chz, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__vel_l, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__vel_r, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__height, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__head_pit, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__head_yaw, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__temp_float_data, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      6,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__temp_int_data, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32_ARRAY,
      6,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__mode1, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__mode2, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {arm_control__msg__PosCmd__FIELD_NAME__time_count, 10, 10},
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
arm_control__msg__PosCmd__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {arm_control__msg__PosCmd__TYPE_NAME, 22, 22},
      {arm_control__msg__PosCmd__FIELDS, 24, 24},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "float64 x\n"
  "float64 y\n"
  "float64 z\n"
  "float64 roll\n"
  "float64 pitch\n"
  "float64 yaw\n"
  "float64 gripper\n"
  "float64 quater_x\n"
  "float64 quater_y\n"
  "float64 quater_z\n"
  "float64 quater_w\n"
  "float64 chx\n"
  "float64 chy\n"
  "float64 chz\n"
  "float64 vel_l\n"
  "float64 vel_r\n"
  "float64 height\n"
  "float64 head_pit\n"
  "float64 head_yaw\n"
  "float64[6] temp_float_data\n"
  "int32[6] temp_int_data\n"
  "int32 mode1\n"
  "int32 mode2\n"
  "int32 time_count\n"
  "";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
arm_control__msg__PosCmd__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {arm_control__msg__PosCmd__TYPE_NAME, 22, 22},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 358, 358},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
arm_control__msg__PosCmd__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *arm_control__msg__PosCmd__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
