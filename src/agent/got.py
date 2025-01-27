import gymnasium as gym
from llm import llm, async_llm
import re
import asyncio
from typing import Any

from .base import PolicyAgent

class GoTAgent(PolicyAgent):
    def __init__(
        self,
        env: gym.Env,
        task,
        
        # GoT parameters
        branches:int,
        generate_attempts:int,
        aggregate_attempts:int,
        post_aggregate_keepbest: bool,
        post_aggregate_refine: bool,
        refine_attempts:int,
    ):
        super().__init__(
            env=env,
            task=task,
        )

        # Dynamic action selection
        self.subproblems = {}
        self.last_action = None
        self.decomposition_attempts = 0
        self.aggregation_attempts = 0
        self.old_nodes = set([0])

    def get_action(
        self, 
        obs: Any,
        decompose_attempts: int = 3,
        generate_attempts: int = 1,
        refine_attempts: int = 1,
        aggregate_attempts: int = 1,
    ) -> int:

        # Process last action
        # ==========================

        # First action is always split
        if self.last_action is None:
            self.last_action = {
                "nodes": ["0"],
                "operation": "split",
                "explanation": "",
            }
            self.decomposition_attempts += 1
            return self.last_action
        
        # Register the subproblems after splitting
        elif self.last_action["operation"] == "split":

            decomposed_nodes = set(obs.nodes.keys()) - self.old_nodes
            self.subproblems = {
                node: {
                    "solved": False,
                    "solve_attempts": 0,
                    "refine_attempts": 0,
                } for node in decomposed_nodes
            }

        # Score previous generation attempts
        elif self.last_action["operation"] == "generate":
            self.last_action = {
                "nodes": self.last_action["nodes"],
                "operation": "score",
                "explanation": "",
            }

            return self.last_action

        # Solved a subproblem, so score it
        elif self.last_action["operation"] == "score":
            # Register the subproblem as solved
            for node in self.last_action["nodes"]:
                score = obs.nodes[int(node)].get("score", 0)
                if score > 0:
                    self.subproblems[int(node)]["solved"] = True

        # exhausted this decomposition, start from scratch
        elif self.last_action["operation"] == "aggregate" and self.aggregation_attempts >= aggregate_attempts:

            if self.decomposition_attempts > decompose_attempts:
                raise Exception("Exhausted decomposition attempts")

            else:
                self.last_action = {
                    "nodes": ["0"],
                    "operation": "split",
                    "explanation": "",
                }
                self.decomposition_attempts += 1

                self.old_nodes = set(obs.nodes.keys())
                return self.last_action

            return self.last_action

        # Decide whether to generate or refine
        # ==========================

        to_solve = []
        to_refine = []
        for subproblem, info in self.subproblems.items():
            if info["solved"]:
                continue

            if info["solve_attempts"] < generate_attempts:
                to_solve.append(subproblem)
            elif info["refine_attempts"] < refine_attempts:
                to_refine.append(subproblem)

        # If any subproblems to solve, attempt it
        if to_solve:
            self.last_action = {
                "nodes": [str(node) for node in to_solve],
                "operation": "generate",
                "explanation": "",
            }
            for node in to_solve:
                self.subproblems[node]["solve_attempts"] += 1
            
            return self.last_action

        if to_refine:
            self.last_action = {
                "nodes": [str(node) for node in to_refine],
                "operation": "refine",
                "explanation": "",
            }
            for node in to_refine:
                self.subproblems[node]["refine_attempts"] += 1
            
            return self.last_action
        
        # All subproblems solved, so aggregate
        else:
            self.last_action = {
                "nodes": list(self.subproblems.keys()) + ["0"],
                "operation": "aggregate",
                "explanation": "",
            }
            self.aggregation_attempts += 1
            return self.last_action

        raise Exception("I don't know what to do...")