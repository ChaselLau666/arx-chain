from __future__ import annotations

import os
import re
import subprocess
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / 'tools' / '05_human_dagger.sh'
SHUTDOWN = ROOT / 'tools' / '04_safe_shutdown.sh'
CONFIG = ROOT / 'act' / 'data' / 'human_dagger.yaml'
DOC = ROOT / 'docs' / 'HUMAN_DAGGER_WORKFLOW.md'
ENTRYPOINT = ROOT / 'act' / 'human_dagger.py'


class HumanDaggerScriptTests(unittest.TestCase):
    def test_timestamp_dataset_directory_is_new_and_override_is_preserved(self):
        text = START.read_text()
        assignment = next(line for line in text.splitlines() if line.startswith('DATASET_DIR='))
        begin = text.index('if [[ -n "${HUMAN_DAGGER_DATASET_DIR:-}" ]]; then')
        block = text[begin:text.index('[[ -w "$DATASET_DIR" ]]', begin)]
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.pop('HUMAN_DAGGER_DATASET_DIR', None)
            env['repo_root'] = directory
            def run(stamp, override=None):
                current = env.copy()
                if override is not None:
                    current['HUMAN_DAGGER_DATASET_DIR'] = override
                return subprocess.run(['bash', '-c',
                    'set -euo pipefail\ndie() { exit 1; }\n'
                    + f'date() {{ echo {stamp}; }}\n' + assignment + '\n' + block],
                    env=current, capture_output=True, text=True)
            first = '20260905_180000_000000001'
            second = '20260905_180000_000000002'
            self.assertEqual(run(first).returncode, 0)
            self.assertNotEqual(run(first).returncode, 0)  # collision cannot reuse data
            self.assertEqual(run(second).returncode, 0)
            self.assertTrue((Path(directory) / ('dagger_datasets_' + first)).is_dir())
            self.assertTrue((Path(directory) / ('dagger_datasets_' + second)).is_dir())
            custom = str(Path(directory) / 'custom')
            self.assertEqual(run(first, custom).returncode, 0)
            self.assertEqual(run(second, custom).returncode, 0)

    @classmethod
    def setUpClass(cls):
        cls.start = START.read_text(encoding='utf-8')
        cls.shutdown = SHUTDOWN.read_text(encoding='utf-8')
        cls.config = CONFIG.read_text(encoding='utf-8')
        cls.entrypoint = ENTRYPOINT.read_text(encoding='utf-8')

    def test_shell_scripts_pass_bash_syntax_check(self):
        for script in (START, SHUTDOWN):
            result = subprocess.run(
                ['bash', '-n', str(script)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_start_fails_before_hardware_when_required_env_is_missing(self):
        env = os.environ.copy()
        for name in ('TASK_NAME', 'LIFT_HEIGHT', 'CKPT_DIR'):
            env.pop(name, None)
        result = subprocess.run(
            ['bash', str(START)],
            cwd='/',
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Set TASK_NAME', result.stderr)

    def test_start_uses_script_location_and_dedicated_domain(self):
        self.assertIn('dirname "${BASH_SOURCE[0]}"', self.start)
        self.assertIn('repo_root="$(cd "${script_dir}/.." && pwd)"', self.start)
        # The domain is the machine identity and must never be hardcoded in the
        # repo: both robots share one LAN, and a wrong domain drives the other
        # robot. The script must refuse to run without an externally set value.
        self.assertIn('${ROS_DOMAIN_ID:?', self.start)
        self.assertNotIn('export ROS_DOMAIN_ID=62', self.start)
        self.assertNotIn('workspace=$(pwd)', self.start)

    def test_known_arm_owners_are_rejected(self):
        for pattern in (
            '[v]2_pos_control',
            '[v]2_joint_control',
            '[o]pen_double_arm',
            '[/]arx_x5_controller/[X]5Controller',
        ):
            self.assertIn(pattern, self.start)

    def test_normal_x5_nodes_have_isolated_topics_and_can_ids(self):
        expected = (
            '-p arm_control_type:=normal',
            '-p arm_end_type:=2',
            '-p arm_can_id:=can1',
            '-p arm_can_id:=can3',
            '-r __node:=human_dagger_arm_left',
            '-r __node:=human_dagger_arm_right',
            'arm_pub_topic_name:=/human_dagger/arm/left/status',
            'arm_sub_topic_name:=/human_dagger/arm/left/command',
            'arm_pub_topic_name:=/human_dagger/arm/right/status',
            'arm_sub_topic_name:=/human_dagger/arm/right/command',
        )
        for value in expected:
            self.assertIn(value, self.start)

    def test_vr_is_remapped_without_diagnostic_launcher(self):
        self.assertIn('ros2 run serial_port serial_port_node', self.start)
        self.assertIn(
            '-r /ARX_VR_L:=/human_dagger/vr/left_raw', self.start
        )
        self.assertIn(
            '-r /ARX_VR_R:=/human_dagger/vr/right_raw', self.start
        )
        self.assertNotIn('./ARX_VR.sh', self.start)

    def test_three_cameras_and_fixed_height_are_preflighted(self):
        for name in ('camera_h', 'camera_l', 'camera_r'):
            self.assertIn(f'start_camera {name}', self.start)
            self.assertIn(
                f'/camera/{name}/color/image_rect_raw/compressed', self.start
            )
        self.assertIn('ros2 param set /lift fixed_height', self.start)
        self.assertIn('wait_for_topic /body_information', self.start)

    def test_integer_lift_height_is_normalized_as_double_everywhere(self):
        result = subprocess.run(
            ['awk', '-v', 'value=15', 'BEGIN { printf "%.12f", value + 0.0 }'],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout, '15.000000000000')
        self.assertIn(
            'LIFT_HEIGHT_ROS=$(awk -v value="$LIFT_HEIGHT"', self.start
        )
        self.assertIn(
            'ros2 param set /lift fixed_height "$LIFT_HEIGHT_ROS"', self.start
        )
        self.assertIn('-p "fixed_height:=${LIFT_HEIGHT_ROS}"', self.start)
        self.assertIn('--height "$LIFT_HEIGHT_ROS"', self.start)

    def test_session_manifest_guards_against_pid_reuse(self):
        for source in (self.start, self.shutdown):
            self.assertIn('/proc/${pid}/stat', source)
            self.assertIn('proc_start_ticks', source)
            self.assertIn('manifest_entry_is_live', source)
        self.assertIn('setsid "$@"', self.start)
        self.assertIn('--session-manifest "$manifest"', self.start)
        self.assertIn('stop_active_manifest', self.shutdown)

    def test_vendor_can_watchdogs_are_never_started_or_retained(self):
        for script_name in ('arx_can1.sh', 'arx_can3.sh', 'arx_can5.sh'):
            self.assertNotIn(script_name, self.start)
        self.assertIn('configure CAN safely before starting Human DAgger', self.start)
        self.assertIn("ip -o link show dev", self.start)
        self.assertIn("-F'[<>]'", self.start)
        self.assertNotIn("$2 == \"UP\"", self.start)
        self.assertNotIn('transport watchdog retained for reuse', self.shutdown)

    def test_legacy_broad_shutdown_requires_explicit_opt_in(self):
        opt_in = self.shutdown.index('HUMAN_DAGGER_ALLOW_LEGACY_SHUTDOWN')
        broad_stop = self.shutdown.index('stop_pattern "collector"')
        self.assertLess(opt_in, broad_stop)
        self.assertIn('CONFIRM LEGACY PIDS', self.shutdown)

    def test_shutdown_requests_hold_before_lowering(self):
        hold = self.shutdown.index('/human_dagger/request_hold')
        lower = self.shutdown.index('ros2 param set /lift fixed_height 0.0')
        self.assertLess(hold, lower)
        self.assertIn('${ROS_DOMAIN_ID:?', self.shutdown)
        self.assertNotIn('export ROS_DOMAIN_ID=62', self.shutdown)
        # Shutdown must refuse a manifest recorded on another machine or domain.
        self.assertIn('manifest_matches_this_machine', self.shutdown)
        self.assertIn('# hostname=', self.shutdown)
        self.assertIn('# ros_domain_id=', self.shutdown)
        self.assertIn('Human DAgger HOLD acknowledged', self.shutdown)
        self.assertIn('human_dagger.py', self.shutdown)

    def test_live_tracked_arm_requires_hold_service_while_commanded(self):
        arm_check = self.shutdown.index(
            'manifest_has_live_label "$resolved_manifest" arm_left'
        )
        missing_service_guard = self.shutdown.index(
            '"$tracked_arm_running" == true && "$tracked_commander_running" == true'
        )
        lower = self.shutdown.index('ros2 param set /lift fixed_height 0.0')
        self.assertLess(arm_check, missing_service_guard)
        self.assertLess(missing_service_guard, lower)
        self.assertIn(
            'manifest_has_live_label "$resolved_manifest" arm_right',
            self.shutdown,
        )
        self.assertIn(
            'a verified tracked arm is still running', self.shutdown
        )
        # A commander (frontend/coordinator/policy) that is still alive must
        # keep the hard refusal; only a fully commander-less session may take
        # the crash-recovery path.
        for commander in ("frontend", "coordinator", "policy"):
            self.assertIn(commander, self.shutdown)
        self.assertIn('crash-recovery shutdown', self.shutdown)

    def test_hold_service_can_receive_post_publish_feedback(self):
        self.assertIn('ReentrantCallbackGroup', self.entrypoint)
        self.assertIn('callback_group=self.io_callback_group', self.entrypoint)
        self.assertIn('callback_group=node.service_callback_group', self.entrypoint)
        self.assertIn('hold_node.create_service(', self.entrypoint)
        self.assertIn('hold_executor.add_node(hold_node)', self.entrypoint)
        self.assertIn('target=hold_executor.spin', self.entrypoint)
        self.assertIn('external_hold_published_ns = monotonic_ns()', self.entrypoint)
        self.assertIn('_feedback_pair_acknowledges_hold(', self.entrypoint)
        self.assertIn('>= hold_published_ns', self.entrypoint)

    def test_config_locks_topics_modes_timeouts_and_storage(self):
        expected = (
            'schema_version: 2',
            'status_topic: /human_dagger/arm/left/status',
            'command_topic: /human_dagger/arm/right/command',
            'joint_mode: 5',
            'eef_mode: 4',
            'feedback_timeout_ms: 100',
            'vr_timeout_ms: 100',
            'policy_timeout_ms: 250',
            'handoff_timeout_s: 2.0',
            'vr_engage_enabled: true',
            'vr_engage_field: mode1',
            'gripper_trigger_open_below: 2.0',
            'gripper_trigger_close_above: 3.0',
            'gripper_open_value: 0.0',
            'gripper_closed_value: -3.384',
            'dataset_dir: dagger_datasets',
            'quarantine_dir: dagger_datasets/quarantine',
        )
        for value in expected:
            self.assertIn(value, self.config)

    def test_operator_workflow_is_documented(self):
        text = DOC.read_text(encoding='utf-8')
        for value in (
            '本地 GNOME Terminal',
            'Space',
            '/human_dagger/request_hold',
            'validate_dagger_episode.py',
            '物理急停',
        ):
            self.assertIn(value, text)

    def test_tau0vla_gripper_defaults_match_standalone_and_allow_overrides(self):
        # Evaluate only the argument array, never the hardware startup script.
        array = re.search(r'backend_args=\(\s*--policy-backend tau0vla.*?\n    \)', self.start, re.S)
        self.assertIsNotNone(array)
        defaults = {
            'CHUNK_BLEND_STEPS': '6', 'GRIPPER_BLEND_STEPS': '0',
            'GRIPPER_DEBOUNCE_FRAMES': '12', 'ARM_EMA_ALPHA': '0.6',
            'GRIPPER_EMA_ALPHA': '0.6', 'GRIPPER_LOW_THRESHOLD': '-2.1',
            'GRIPPER_HIGH_THRESHOLD': '-1.05', 'GRIPPER_LOW_VALUE': '-3.384',
            'GRIPPER_HIGH_VALUE': '0.0',
        }
        standalone = (ROOT / 'tools' / '03_tau0vla_inference.sh').read_text()
        for name, value in defaults.items():
            self.assertIn('${' + name + ':=' + value + '}', standalone)
        env = os.environ.copy()
        for name in defaults:
            env.pop(name, None)
        env.update(MODEL_SERVER_URL='http://fake', TASK_INSTRUCTION='test task')
        for overrides in ({}, {'GRIPPER_DEBOUNCE_FRAMES': '0', 'ARM_EMA_ALPHA': '1.0'}):
            output = subprocess.check_output(
                ['bash', '-c', array.group(0) + '\nprintf "%s\\n" "${backend_args[@]}"'],
                env={**env, **overrides}, text=True,
            ).splitlines()
            args = dict(zip(output[::2], output[1::2]))
            for name, value in {**defaults, **overrides}.items():
                flag = '--' + name.lower().replace('_', '-')
                self.assertEqual(args[flag], value)
                field = name.lower()
                self.assertIn(f'{field}=args.{field}', self.entrypoint)


if __name__ == '__main__':
    unittest.main()
