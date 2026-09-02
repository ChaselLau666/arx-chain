import os

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

lift_node = Node(
    package='arx_lift_controller',
    executable='lift_controller',
    name='x7s',
    output='screen',
    parameters=[{'robot_type': 1}],
)


def generate_launch_description():
    return LaunchDescription([
        lift_node,
    ])
