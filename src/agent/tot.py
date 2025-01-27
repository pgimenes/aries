import gymnasium as gym
from llm import llm, async_llm
import re
import asyncio
from typing import Any
    
class ToTAgent:
    def __init__(
        self,
        env: gym.Env,
        task,
        model: str,
        problem_definition: str,
        actions: dict[str, str],
        width: int,
        depth: int,
    ):
        self.env = env
        self.task = task 

        self.model = model
        self.problem_definition = problem_definition
        self.actions = actions
        
        self.width = width
        self.depth = depth

        schedule = getattr(task, "_tot_schedule")
        self._actions, self._action_nodes = schedule(
            width=self.width,
            depth=self.depth,
        )
        self.itr = 0

        self.max_iterations = len(self._actions)


    def get_action(self, obs: tuple[int, int, bool]) -> int:
        action, action_nodes = self._actions[self.itr], self._action_nodes[self.itr]
        self.itr += 1
        return {
            "nodes": action_nodes,
            "operation": action,
            "explanation": "",
        }