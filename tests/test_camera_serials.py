"""Exercise camera selection without opening terminals or starting ROS."""
import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "realsense/camera_serials.sh"
ARK1 = ["260422272688", "260422274927", "260522274175"]
ARK2 = ["260422273990", "260422273222", "260422272473"]


def run_shell(script, host="ark-1", overrides=None):
    env = os.environ.copy()
    for key in ("CAMERA_H_SERIAL", "CAMERA_L_SERIAL", "CAMERA_R_SERIAL"):
        env.pop(key, None)
    env.update(overrides or {})
    env["TEST_CAMERA_HOST"] = host
    env["SHELL"] = "/bin/bash"
    result = subprocess.run(
        ["bash", "-c", 'hostname() { printf "%s\\n" "$TEST_CAMERA_HOST"; };\n' + script],
        env=env, text=True, capture_output=True, check=True,
    )
    return result.stdout.splitlines()


class CameraSerialTests(unittest.TestCase):
    def selection(self, host, profile, overrides=None):
        return run_shell(
            f"set -eu; source {shlex.quote(str(CONFIG))}; load_camera_serials {profile}; "
            'printf "%s\\n" "$CAMERA_H_SERIAL" "$CAMERA_L_SERIAL" "$CAMERA_R_SERIAL"',
            host, overrides,
        )

    def test_ark1_matches_connected_devices_for_both_profiles(self):
        for profile in ("standard", "dagger"):
            with self.subTest(profile=profile):
                self.assertEqual(self.selection("ark-1", profile), ARK1)

    def test_ark2_keeps_existing_per_launcher_defaults(self):
        self.assertEqual(self.selection("ark-2", "standard"), ARK2)
        self.assertEqual(self.selection("ark-2", "dagger"), ["260522275257", *ARK2[1:]])

    def test_explicit_overrides_take_precedence_and_empty_uses_default(self):
        for host in ("ark-1", "ark-2"):
            for profile in ("standard", "dagger"):
                with self.subTest(host=host, profile=profile):
                    actual = self.selection(host, profile, {
                        "CAMERA_H_SERIAL": "111", "CAMERA_L_SERIAL": "",
                        "CAMERA_R_SERIAL": "333",
                    })
                    expected_left = ARK1[1] if host == "ark-1" else ARK2[1]
                    self.assertEqual(actual, ["111", expected_left, "333"])

    def test_sourcing_configuration_has_no_launch_side_effects(self):
        self.assertEqual(run_shell(f"source {shlex.quote(str(CONFIG))}"), [])

    def test_both_headless_entrypoints_evaluate_host_configuration(self):
        for filename, profile in (("05_human_dagger.sh", "dagger"),
                                  ("06_collect_filtered.sh", "standard")):
            with self.subTest(filename=filename):
                path = ROOT / "tools" / filename
                text = path.read_text()
                end = text.index(f"load_camera_serials {profile}") + len(f"load_camera_serials {profile}")
                # Execute the actual configuration lines, before hardware code.
                prefix = text[:end]
                result = run_shell(
                    f'script_dir={shlex.quote(str(ROOT / "tools"))}; '
                    f'repo_root={shlex.quote(str(ROOT))};\n' +
                    prefix[prefix.index('source "${repo_root}/realsense/camera_serials.sh"'):] +
                    '\nprintf "%s\\n" "$CAMERA_H_SERIAL" "$CAMERA_L_SERIAL" "$CAMERA_R_SERIAL"',
                )
                self.assertEqual(result, ARK1)
                self.assertNotIn("sed -n 's/^ *\\[", text)

    def test_graphical_launcher_passes_selected_serials_to_ros(self):
        path = ROOT / "realsense/realsense.sh"
        lines = run_shell(
            'gnome-terminal() { printf "%s\\n" "$*"; }; sleep() { :; }; '
            f'source {shlex.quote(str(path))}',
        )
        output = "\n".join(lines)
        for serial in ARK1:
            self.assertIn(f"serial_no:=_{serial}", output)
        for serial in ARK2:
            self.assertNotIn(serial, output)


if __name__ == "__main__":
    unittest.main()
