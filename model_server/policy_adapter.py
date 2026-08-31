"""Stable policy boundary for the ARX HTTP server."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PolicyAdapter(ABC):
    model_id = "unloaded"

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def reset(self, session_id: str, task_instruction: str) -> None:
        pass

    @abstractmethod
    def infer(self, observation_state: np.ndarray, images: dict[str, bytes], task_instruction: str) -> np.ndarray:
        pass


class MockPolicy(PolicyAdapter):
    """Safe transport-test policy: repeat the observed joint state as a hold chunk."""

    model_id = "mock-hold-v1"

    def __init__(self, horizon: int = 8):
        self.horizon = horizon

    def load(self) -> None:
        return

    def reset(self, session_id: str, task_instruction: str) -> None:
        return

    def infer(self, observation_state: np.ndarray, images: dict[str, bytes], task_instruction: str) -> np.ndarray:
        return np.repeat(observation_state[np.newaxis, :], self.horizon, axis=0).astype(np.float32)
