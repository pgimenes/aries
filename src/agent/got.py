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
        decompose_attempts:int,
        generate_attempts:int,
        refine_attempts:int,
        aggregate_attempts:int,
        post_aggregate_keepbest: bool,
        post_aggregate_refine: bool,
        
        max_iterations: int = None,
    ):
        super().__init__(
            env=env,
            task=task,
            max_iterations=max_iterations,
        )

        # query count
        self.queries = 0

        # Dynamic action selection
        self.subproblems = {}
        self.last_action = None
        self.decomposition_attempts = 0
        self.aggregation_attempts = 0
        self.old_nodes = set([0])

        # GoT parameters
        self.branches = branches
        self.decompose_attempts = decompose_attempts
        self.generate_attempts = generate_attempts
        self.refine_attempts = refine_attempts
        self.aggregate_attempts = aggregate_attempts
        self.post_aggregate_keepbest = post_aggregate_keepbest
        self.post_aggregate_refine = post_aggregate_refine


    def get_action(
        self, 
        obs: Any,
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
        elif self.last_action["operation"] in ["generate", "refine"]:
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
                score = obs.nodes[int(node)].get("score", 100)
                if score == 0:
                    self.subproblems[int(node)]["solved"] = True

        # exhausted this decomposition, start from scratch
        elif self.last_action["operation"] == "aggregate":

            # call groundtruth
            self.last_action = {
                "nodes": [int(max(obs.nodes.keys()))],
                "operation": "groundtruth",
                "explanation": "",
            }
            return self.last_action

        # Decide whether to generate or refine
        # ==========================

        nodes_to_solve = []
        nodes_to_refine = []
        aggregated_nodes_to_refine = []
        
        for subproblem, info in self.subproblems.items():
            if info["solved"]:
                continue

            if info["solve_attempts"] < self.generate_attempts:
                print(f"node {subproblem} needs solving, solve_attempts: {info['solve_attempts']}")
                nodes_to_solve.append(subproblem)
            elif info["refine_attempts"] < self.refine_attempts:
                print(f"node {subproblem} needs refining, refine_attempts: {info['refine_attempts']}")
                nodes_to_refine.append(subproblem)

        print(f"subproblems: {self.subproblems}")
        print(f"nodes_to_solve: {nodes_to_solve}")
        print(f"nodes_to_refine: {nodes_to_refine}")

        # If any subproblems to solve, attempt it
        if nodes_to_solve:
            self.last_action = {
                "nodes": [str(node) for node in nodes_to_solve],
                "operation": "generate",
                "explanation": "",
            }
            for node in nodes_to_solve:
                self.subproblems[node]["solve_attempts"] += 1
            
            return self.last_action

        if nodes_to_refine:
            self.last_action = {
                "nodes": [str(node) for node in nodes_to_refine],
                "operation": "refine",
                "explanation": "",
            }
            for node in nodes_to_refine:
                self.subproblems[node]["refine_attempts"] += 1
            
            return self.last_action
        
        # All subproblems solved, so aggregate
        elif self.aggregation_attempts < self.aggregate_attempts:
            self.last_action = {
                "nodes": list(self.subproblems.keys()) + ["0"],
                "operation": "aggregate",
                "explanation": "",
            }
            self.aggregation_attempts += 1
            return self.last_action

        # Try decompose again
        else:
            
            # Check if too many decomposition attempts
            if self.decomposition_attempts > self.decompose_attempts:
                raise Exception("Exhausted decomposition attempts")

            # Try to decompose from scratch
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


        raise Exception("I don't know what to do...")