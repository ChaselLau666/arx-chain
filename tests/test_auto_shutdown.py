"""Exercise shutdown routing with stubs only: never call ROS or signal processes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SAFE = ROOT / "tools/04_safe_shutdown.sh"
AUTO = ROOT / "tools/04_auto_shutdown.sh"


class AutoShutdownTests(unittest.TestCase):
    def run_shell(self, script, **values):
        env = os.environ.copy()
        for key in ("HUMAN_DAGGER_SHUTDOWN_ALL", "HUMAN_DAGGER_ALLOW_LEGACY_SHUTDOWN"):
            env.pop(key, None)
        env.update(values)
        return subprocess.run(["bash", "-c", script], env=env, text=True,
                              input="", capture_output=True, timeout=5)

    def test_wrapper_explicitly_selects_whole_stack_and_auto_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / AUTO.name).write_text(AUTO.read_text())
            (target / SAFE.name).write_text(
                'printf "%s %s\\n" "$HUMAN_DAGGER_AUTO_CONFIRM" "$HUMAN_DAGGER_SHUTDOWN_ALL"\n'
            )
            result = self.run_shell(f'bash "{target / AUTO.name}"')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "1 1")

    def test_missing_manifest_needs_no_pid_confirmation_only_in_whole_stack_mode(self):
        text = SAFE.read_text()
        gate = text[text.index('if [[ "$shutdown_all" == 1 ]]; then'):
                    text.index('dagger_process_running=false')]
        for all_stacks in ("0", "1"):
            with self.subTest(all_stacks=all_stacks):
                result = self.run_shell(
                    'set -euo pipefail\ntracked_session=false\nauto_confirm=1\n'
                    'ps() { printf "123 1 123 X5Controller\\n"; }\n' + gate,
                    shutdown_all=all_stacks, HUMAN_DAGGER_ALLOW_LEGACY_SHUTDOWN="1",
                )
                self.assertEqual(result.returncode, 0 if all_stacks == "1" else 1, result.stderr)
                if all_stacks == "1":
                    self.assertIn("skipping session ownership", result.stdout)
                else:
                    self.assertIn("auto-confirm never covers", result.stderr)

    def test_whole_stack_stops_both_manifest_workers_and_untracked_nodes(self):
        text = SAFE.read_text()
        start = text.rindex('if [[ "$tracked_session" == true ]]; then')
        routing = text[start:text.index('echo "Verifying control processes..."')]
        for tracked, all_stacks in (("true", "1"), ("false", "1"), ("true", "0")):
            with self.subTest(tracked=tracked, all_stacks=all_stacks):
                result = self.run_shell(
                    'set -euo pipefail\n'
                    'stop_active_manifest() { echo MANIFEST; }\n'
                    'stop_pattern() { printf "PATTERN:%s\\n" "$1"; }\n' + routing,
                    tracked_session=tracked, shutdown_all=all_stacks,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("MANIFEST" in result.stdout, tracked == "true")
                self.assertEqual("PATTERN:arm nodes" in result.stdout, all_stacks == "1")
                self.assertEqual("PATTERN:body node" in result.stdout, all_stacks == "1")

    def test_whole_stack_keeps_hold_and_low_feedback_before_stop(self):
        text = SAFE.read_text()
        hold = text.index('echo "Requesting Human DAgger HOLD')
        lower = text.index('ros2 param set /lift fixed_height 0.0')
        low_check = text.index('"${repo_root}/act/wait_for_safe_height.py"')
        stop = text.rindex('if [[ "$tracked_session" == true ]]; then')
        self.assertLess(hold, lower)
        self.assertLess(lower, low_check)
        self.assertLess(low_check, stop)
        self.assertIn('"$shutdown_all" != 1 && "$tracked_session" == true', text)
        self.assertIn('/serial_port_node([[:space:]]|$)', text)
        for path in (AUTO, SAFE):
            subprocess.run(["bash", "-n", str(path)], check=True)


if __name__ == "__main__":
    unittest.main()
