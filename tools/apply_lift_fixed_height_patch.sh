#!/bin/bash
set -euo pipefail

sdk_root=${LIFT_SDK_ROOT:-/home/arx/LIFT}
source_file="${sdk_root}/body/ROS2/src/ARX_LIFT_ros2/arx_lift_controller/src/lift_controller.cpp"

python3 - "${source_file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if 'declare_parameter("fixed_height"' not in text:
    replacements = [
        (
            "#include <sensor_msgs/msg/imu.hpp>\n#include <csignal>\n",
            "#include <sensor_msgs/msg/imu.hpp>\n#include <csignal>\n#include <cmath>\n",
        ),
        (
            '  int robot_type = node->declare_parameter("robot_type", 0);\n',
            '  int robot_type = node->declare_parameter("robot_type", 0);\n'
            '  double fixed_height = node->declare_parameter("fixed_height", -1.0);\n'
            '  auto resolve_height = [&](double vr_height) {\n'
            '    node->get_parameter("fixed_height", fixed_height);\n'
            '    return std::isfinite(fixed_height) && fixed_height >= 0.0 && fixed_height <= 20.0\n'
            '               ? fixed_height : vr_height;\n'
            '  };\n',
        ),
        (
            "                                                                   control_loop->setHeight(msg.height);\n",
            "                                                                   control_loop->setHeight(resolve_height(msg.height));\n",
        ),
        (
            "    control_loop->setHeight(lift_height);\n",
            "    control_loop->setHeight(resolve_height(lift_height));\n",
        ),
    ]
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"refused: expected one SDK match, found {count}: {old!r}")
        text = text.replace(old, new, 1)

marker = "// Enforce fixed height independently of VR/joy callback availability."
if marker not in text:
    old = "  while (rclcpp::ok()) {\n    control_loop->loop();\n"
    new = (
        "  while (rclcpp::ok()) {\n"
        f"    {marker}\n"
        "    node->get_parameter(\"fixed_height\", fixed_height);\n"
        "    if (std::isfinite(fixed_height) && fixed_height >= 0.0 && fixed_height <= 20.0)\n"
        "      control_loop->setHeight(fixed_height);\n"
        "    control_loop->loop();\n"
    )
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"refused: expected one control loop match, found {count}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
PY

set +u
source /opt/ros/jazzy/setup.bash
set -u
cd "${sdk_root}/body/ROS2"
colcon build --packages-select arx_lift_controller
echo "fixed_height SDK patch built successfully"
