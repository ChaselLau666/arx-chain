"""Template for integrating the real tau-0 policy without changing HTTP."""

from __future__ import annotations

import numpy as np

from model_server.policy_adapter import PolicyAdapter


class TauPolicyAdapter(PolicyAdapter):
    """Integration boundary; intentionally fails until the real model is wired."""

    model_id = "tau-policy-unconfigured"

    def load(self) -> None:
        raise NotImplementedError(
            "Connect the real /home/xiangchengliu/code/tau-0-vla policy here; "
            "do not change the HTTP schema."
        )

    def reset(self, session_id: str, task_instruction: str) -> None:
        raise NotImplementedError

    def infer(self, observation_state: np.ndarray, images: dict[str, bytes], task_instruction: str) -> np.ndarray:
        raise NotImplementedError
