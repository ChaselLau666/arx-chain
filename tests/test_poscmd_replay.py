"""Static wiring checks for filtered PosCmd collection and selectable replay."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
COLLECT = (ROOT / 'act' / 'collect.py').read_text()
REPLAY = (ROOT / 'act' / 'replay.py').read_text()
ROS_OPERATOR = (ROOT / 'act' / 'utils' / 'ros_operator.py').read_text()
FILTERED_LAUNCHER = (ROOT / 'tools' / '06_collect_filtered.sh').read_text()
READY_LAUNCHER = (ROOT / 'tools' / '08_collect_ready_pose.sh').read_text()
REPLAY_LAUNCHER = (ROOT / 'tools' / '05_replay.sh').read_text()


class PosCmdCollectionWiringTests(unittest.TestCase):
    def test_command_has_its_own_dataset(self):
        self.assertIn("'/action_poscmd': []", COLLECT)
        self.assertIn("action_poscmd = deepcopy(action_dict['action_poscmd'])", COLLECT)
        self.assertNotIn("action_poscmd = deepcopy(obs_dict['eef'])", COLLECT)

    def test_filtered_launchers_record_the_topics_consumed_by_vr_slave(self):
        self.assertIn('--poscmd-topics "${ARM_POSE_L}" "${ARM_POSE_R}"', FILTERED_LAUNCHER)
        self.assertIn('--poscmd-topics ${ARM_POSE_L} ${ARM_POSE_R}', READY_LAUNCHER)

    def test_source_topic_metadata_is_written(self):
        self.assertIn("root.attrs['action_poscmd_left_topic']", COLLECT)
        self.assertIn("root.attrs['action_poscmd_right_topic']", COLLECT)


class PosCmdReplayWiringTests(unittest.TestCase):
    def test_mode_is_explicit_and_joint_remains_default(self):
        self.assertIn("choices=['joint', 'poscmd'], default='joint'", REPLAY)

    def test_poscmd_uses_poscmd_publishers(self):
        self.assertIn("args.replay_mode == 'poscmd'", REPLAY)
        self.assertIn('ros_operator.poscmd_publish(left_action, right_action)', REPLAY)
        self.assertIn('self.poscmd_left_publisher', ROS_OPERATOR)
        self.assertIn('self.poscmd_right_publisher', ROS_OPERATOR)

    def test_launcher_selects_the_matching_controller(self):
        self.assertIn('v2_joint_control.launch.py', REPLAY_LAUNCHER)
        self.assertIn('v2_pos_control.launch.py', REPLAY_LAUNCHER)
        self.assertIn('if [[ "${replay_mode}" == joint ]]', REPLAY_LAUNCHER)

    def test_old_episode_is_not_silently_replayed_as_measured_eef(self):
        self.assertIn("actions_poscmd = root.get('/action_poscmd')", REPLAY)
        self.assertNotIn("replay_actions = actions_eefs", REPLAY)


if __name__ == '__main__':
    unittest.main()
