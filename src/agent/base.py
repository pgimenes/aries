import gymnasium as gym
from llm import llm, async_llm
import re
import asyncio
from typing import Any

class PolicyAgent:
    def __init__(
        self,
        env: gym.Env,
        task,
        max_iterations: int = None,
    ):
        self.env = env
        self.task = task
        self.max_iterations = max_iterations

    def get_action(
        self, 
        obs: Any,
        *args,
        **kwargs,
    ) -> int:
        raise NotImplementedError("get_action not implemented")

