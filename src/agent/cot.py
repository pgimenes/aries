import gymnasium as gym
from llm import llm, async_llm
import re
import asyncio
from typing import Any
    
class CoTAgent:
    def __init__(
        self,
        env: gym.Env,
        task,
        self_consistency: bool = False,
        num_branches: int = 10,
    ):
        self.env = env
        self.task = task

        self.itr = 0
        if self_consistency:
            self._action_list = ["sort_cot"] * num_branches
            self._action_list += ["score"]
            self._action_list += ["keepbest"]
            self._action_list += ["groundtruth"]

            self._node_list = [
                ["0"] * num_branches,
                [str(i) for i in range(1, num_branches + 1)],
                [str(i) for i in range(1, num_branches + 1)],
                [str(num_branches + 1)],
            ]
        else:
            self._action_list = [
                "cot",
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