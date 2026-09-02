from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from act_contract import build_act_policy_config, data_contract, effective_actions


def act_args(**overrides):
    values = {
        "lr": 4e-5,
        "lr_backbone": 4e-5,
        "weight_decay": 1e-4,
        "loss_function": "l1",
        "backbone": "resnet18",
        "chunk_size": 30,
        "hidden_dim": 512,
        "camera_names": ["head", "left_wrist", "right_wrist"],
        "position_embedding": "sine",
        "masks": False,
        "dilation": False,
        "use_base": False,
        "use_depth_image": False,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "dropout": 0.1,
        "pre_norm": False,
        "kl_weight": 10,
        "dim_feedforward": 3200,
        "use_qvel": False,
        "use_effort": False,
        "use_eef_states": False,
        "use_eef_action": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ActContractTests(unittest.TestCase):
    def test_official_physical_and_model_dimensions(self):
        config = build_act_policy_config(act_args())
        self.assertEqual(config["states_dim"], 14)
        self.assertEqual(config["physical_action_dim"], 14)
        self.assertEqual(config["auxiliary_action_dim"], 14)
        self.assertEqual(config["action_dim"], 28)

        contract = data_contract(config)
        self.assertEqual(contract["physical_action_dim"], 14)
        self.assertEqual(contract["model_action_dim"], 28)

    def test_t_plus_one_alignment_has_no_fabricated_last_action(self):
        source = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
        result = effective_actions(source)
        self.assertEqual(result.shape, (4, 28))
        np.testing.assert_array_equal(result[:, :14], source[1:])
        np.testing.assert_array_equal(result[:, 14:], np.zeros((4, 14)))

    def test_base_dimensions_remain_officially_aligned(self):
        config = build_act_policy_config(act_args(use_base=True))
        self.assertEqual(config["physical_action_dim"], 24)
        self.assertEqual(config["action_dim"], 48)


if __name__ == "__main__":
    unittest.main()
