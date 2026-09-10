"""Exercise model selection without importing ROS or starting robot hardware."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/tau0vla_robot_profile.sh"
ALL_TASKS = {
    "l": "Pick up the L-shaped part and place it in its designated position on the board.",
    "t": "Pick up the T-shaped part and place it in its designated position on the board.",
    "banana": "Pick up the banana and place it in its designated position on the board.",
    "red": "Pick up the red object and place it in its designated position on the board.",
    "blue": "Pick up the blue box and place it in its designated position on the board.",
    "circle": "Pick up the circular part and place it in its designated position on the board.",
}
LEGACY_PROFILES = ("blue-feedback", "t-feedback", "blue-vr", "t-vr")


def load_profile(profile, variant=None, task=None):
    env = dict(os.environ)
    for key in ("MODEL_VARIANT", "TASK_INSTRUCTION"):
        env.pop(key, None)
    env["MODEL_PROFILE"] = profile
    if variant is not None:
        env["MODEL_VARIANT"] = variant
    if task is not None:
        env["TASK_INSTRUCTION"] = task
    return subprocess.run(
        ["bash", "-c", 'set -eu; source "$1"; load_tau0vla_model_profile; '
         'printf "%s\\n" "$MODEL_VARIANT" "$expected_route" "$protocol_version" '
         '"$experiment" "$task"; bash -c \'printf "%s\\n" "$MODEL_VARIANT"\'',
         "profile-test", str(HELPER)],
        env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize("task", ALL_TASKS)
@pytest.mark.parametrize("variant", [None, "0908", "0909"])
def test_all_profiles_preserve_task_and_contract(task, variant):
    result = load_profile(f"all-{task}-feedback", variant)
    if task == "circle" and variant == "0909":
        assert result.returncode != 0
        assert "choose all-cylinder-upper-feedback or all-cylinder-lower-feedback" in result.stderr
        assert not result.stdout
        return
    assert result.returncode == 0, result.stderr
    selected = variant or "0908"
    route = ("arx-lift2s-0909-all-joint-feedback-64g50k-ft" if selected == "0909"
             else "arx-lift2s-0908-all-joint-feedback-ft")
    assert result.stdout.splitlines() == [
        selected, route, "arx-feedback-v4", "joint-feedback", ALL_TASKS[task], selected,
    ]


@pytest.mark.parametrize("position", ["upper", "lower"])
@pytest.mark.parametrize("variant", [None, "0908", "0909"])
def test_cylinder_tasks_are_exact_and_only_supported_for_0909(position, variant):
    result = load_profile(f"all-cylinder-{position}-feedback", variant)
    if variant != "0909":
        assert result.returncode != 0
        assert "requires MODEL_VARIANT=0909" in result.stderr
        assert not result.stdout
        return
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "0909", "arx-lift2s-0909-all-joint-feedback-64g50k-ft", "arx-feedback-v4",
        "joint-feedback",
        f"Pick up the cylindrical part and place it in the {position} hole on the board.",
        "0909",
    ]


@pytest.mark.parametrize("profile", LEGACY_PROFILES)
def test_legacy_profiles_keep_original_route(profile):
    result = load_profile(profile)
    assert result.returncode == 0, result.stderr
    task, mode = profile.split("-")
    assert result.stdout.splitlines() == [
        "0908", f"arx-lift2s-0907-{task}-joint-{mode}-ft", "arx-calibrated-v3",
        f"joint-{mode}", ALL_TASKS[task], "0908",
    ]


@pytest.mark.parametrize("profile", LEGACY_PROFILES)
def test_0909_rejects_legacy_profiles(profile):
    result = load_profile(profile, "0909")
    assert result.returncode != 0
    assert "requires an all-" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("variant", ["0910", "auto", "arx-lift2s-0909-all-joint-feedback-64g50k-ft"])
def test_unknown_variant_is_refused(variant):
    result = load_profile("all-blue-feedback", variant)
    assert result.returncode != 0
    assert "Unknown MODEL_VARIANT" in result.stderr
    assert not result.stdout


def test_prompt_override_remains_explicit():
    result = load_profile("all-t-feedback", "0909", "Operator-provided task.")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[4] == "Operator-provided task."


def test_explicit_task_override_does_not_bypass_variant_mismatch():
    result = load_profile("all-circle-feedback", "0909", "Operator-provided task.")
    assert result.returncode != 0
    assert "no circular-part task" in result.stderr


def test_cylinder_prompt_override_remains_explicit():
    result = load_profile("all-cylinder-upper-feedback", "0909", "Operator-provided task.")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[4] == "Operator-provided task."
