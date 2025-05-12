import gymnasium as gym
from typing import Any

from .base import PolicyAgent

class RLAgent(PolicyAgent):
    def __init__(
        self,
        env: gym.Env,
        task,
        max_iterations: int = None,
    ):
        super().__init__(
            env=env,
            task=task,
            max_iterations=max_iterations,
        )

    def get_action(
        self, 
        obs: Any,
    ) -> int:
        breakpoint()
        raise Exception("I don't know what to do...")
