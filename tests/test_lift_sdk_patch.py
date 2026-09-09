from __future__ import annotations

import os
import unittest
from pathlib import Path


SDK_ROOT = Path(os.environ.get('LIFT_SDK_ROOT', '/home/arx/LIFT'))
SOURCE = (
    SDK_ROOT
    / 'body/ROS2/src/ARX_LIFT_ros2/arx_lift_controller/src/lift_controller.cpp'
)


class LiftSdkFixedHeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_file():
            raise unittest.SkipTest(f'LIFT SDK source is not installed at {SOURCE}')
        cls.text = SOURCE.read_text(encoding='utf-8')

    def test_fixed_height_is_a_double_ros_parameter(self):
        self.assertIn(
            'double fixed_height = node->declare_parameter("fixed_height", -1.0);',
            self.text,
        )

    def test_fixed_height_overrides_both_external_height_paths(self):
        self.assertIn('control_loop->setHeight(resolve_height(msg.height));', self.text)
        self.assertIn('control_loop->setHeight(resolve_height(lift_height));', self.text)

    def test_control_loop_enforces_height_without_fresh_vr_or_joy_messages(self):
        marker = '// Enforce fixed height independently of VR/joy callback availability.'
        self.assertIn(marker, self.text)
        loop = self.text[self.text.index(marker):]
        self.assertLess(
            loop.index('control_loop->setHeight(fixed_height);'),
            loop.index('control_loop->loop();'),
        )


if __name__ == '__main__':
    unittest.main()
