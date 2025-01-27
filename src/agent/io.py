import gymnasium as gym
from llm import llm, async_llm
import re
import asyncio
from typing import Any
    
class IOAgent:
    def __init__(
        self,
        env: gym.Env,
        task = None,
    ):
        self.env = env
        self.task = task

        self.itr = 0
        self._action_list = [
            "io",
            "score",
            "groundtruth",
        ]

        self._node_list = [
            "0",
            "1",
            "1",
        ]

        self.max_iterations = len(self._action_list)

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        action = {
            "nodes": self._node_list[self.itr],
            "operation": self._action_list[self.itr],
            "explanation": "",
        }
        self.itr += 1
        return action