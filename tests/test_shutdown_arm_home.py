"""No ROS imports, CAN access, or process signals in these tests."""
import importlib.util
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shutdown_arm_home', ROOT / 'act/shutdown_arm_home.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HomeTests(unittest.TestCase):
    def test_home_routes_legacy_and_dagger(self):
        old, left, right = module.HOME_TOPICS
        self.assertEqual(module.home_routes({old: ['/vr_arm_r', '/vr_arm_l']}),
                         {old: ('/vr_arm_l', '/vr_arm_r')})
        self.assertEqual(module.home_routes({left: ['/human_dagger_arm_left'],
                                            right: ['/human_dagger_arm_right']}),
                         {left: ('/human_dagger_arm_left',),
                          right: ('/human_dagger_arm_right',)})
        invalid = [
            {}, {old: ['/vr_arm_l']}, {old: ['/vr_arm_l', '/vr_arm_l']},
            {old: ['_NODE_NAME_UNKNOWN_', '/vr_arm_r']},
            {old: ['/a', '/b', '/c']},
            {left: ['/human_dagger_arm_left']},
            {left: ['/human_dagger_arm_right'], right: ['/human_dagger_arm_left']},
            {left: ['/human_dagger_arm_left', '/other'], right: ['/human_dagger_arm_right']},
            {old: ['/a', '/b'], left: ['/human_dagger_arm_left'], right: ['/human_dagger_arm_right']},
        ]
        for graph in invalid:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                module.home_routes(graph)

    def test_feedback_gate(self):
        target = [0.0] * 6
        self.assertTrue(module.at_home([0.0] * 6 + [2.0], [0.0] * 7, target))
        self.assertFalse(module.at_home([0.1] + [0.0] * 6, [0.0] * 7, target))
        self.assertTrue(module.at_home([0.0] * 7, [0.1] + [0.0] * 6, target))
        self.assertTrue(module.at_home([0.0] * 7, [-0.066] + [0.0] * 6, target))
        self.assertFalse(module.at_home([0.0] * 7, [0.101] + [0.0] * 6, target))
        self.assertFalse(module.at_home([0.0] * 7, [-0.101] + [0.0] * 6, target))
        self.assertFalse(module.at_home([float('nan')] * 7, [0.0] * 7, target))
        self.assertFalse(module.at_home([0.0] * 6, [0.0] * 7, target))
        self.assertTrue(module.at_home([0.2] * 6 + [0], [0.0] * 7, [0.2] * 6))

    def test_home_before_lower_and_failure_blocks_lower(self):
        script = (ROOT / 'tools/04_safe_shutdown.sh').read_text()
        home = script.index('"${repo_root}/act/shutdown_arm_home.py"')
        lower = script.index('ros2 param set /lift fixed_height 0.0')
        self.assertLess(home, lower)
        self.assertLess(script.index('echo "Human DAgger HOLD acknowledged."'), home)
        self.assertLess(script.index('stop_pattern "Tau0VLA inference"'), home)
        self.assertIn('set -euo pipefail', script)
        # Exercise the exact HOME block with stubbed process/ROS environment.
        block = script[script.index('# HOME must complete'):script.index("if ros2 node list")]
        for helper_code in (0, 1):
            fake = ('set -euo pipefail\nshutdown_all=1\ntracked_session=false\nauto_confirm=1\n'
                    'repo_root=/unused\npgrep() { return 0; }\n'
                    'stop_pattern() { :; }\nsource() { :; }\n')
            block_stub = block.replace('/home/arx/miniconda3/envs/act/bin/python', 'home_helper')
            result = subprocess.run(['bash', '-c', fake +
                f'home_helper() {{ return {helper_code}; }}\n' + block_stub + '\necho LOWER\n'],
                text=True, capture_output=True, timeout=3)
            self.assertEqual(result.returncode, helper_code, result.stderr)
            self.assertEqual('LOWER' in result.stdout, helper_code == 0)
        subprocess.run(['bash', '-n', str(ROOT / 'tools/04_safe_shutdown.sh')], check=True)


if __name__ == '__main__':
    unittest.main()
