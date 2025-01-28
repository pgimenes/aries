from typing import Optional
import numpy as np

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import Graph, Box, Discrete, Sequence, Dict, GraphInstance
import networkx as nx

import logging
import importlib
import asyncio
import json

from tasks import HumanEvalAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

agent_map = {
    "human_eval": HumanEvalAgent,
}

class GoTEnv(gym.Env):

    def __init__(
            self,
            problem: str = "",
            task: str = "",
            max_graph_size: int = 128,
            node_embedding_size: int = 128,
            model: str = "",
            idx: int = 0,
        ):

        self.step_count = 0
        self.max_graph_size = max_graph_size
        self.node_embedding_size = node_embedding_size
        
        self.problem = problem
        
        self.task_name = task
        self.reasoning_agent = agent_map.get(task, None)
        if self.reasoning_agent is None:
            raise ValueError(f"Task {task} not found in agent map")
        self.reasoning_agent = self.reasoning_agent()

        self.model = model

        self.thought_graph = nx.Graph()
        self.thought_graph.add_node(
            0, 
            embedding=np.zeros(node_embedding_size)
        )
        self.idx = idx

        self.observation_space = Graph(
            node_space = Box(
                low=-1, 
                high=1, 
                shape=(node_embedding_size,)
            ),
            edge_space=None,
        )

        self.action_space = Dict(
            {
                # Choice of graph operator
                "operation": Discrete(6),
                
                # Nodes to apply the operation on
                "nodes": Sequence(
                    Box(
                        low=0,
                        high=max_graph_size,
                        dtype=int,
                    ),
                ),
                # "predecessor": gym.spaces.Discrete(6),
            }
        )

    def _get_obs(self):
        graph = GraphInstance(
            nodes = np.array(
                # [self.thought_graph.nodes[node]["embedding"] for node in self.thought_graph.nodes],
                [np.random.randn(self.node_embedding_size) for _ in self.thought_graph.nodes],
                dtype=np.float32,
            ),
            edges=None,
            edge_links=np.array(
                [edge for edge in self.thought_graph.edges],
                dtype=np.int32,
            ),
        )
        return graph
    
    def _get_info(self):
        return {}
    
    def thought_graph_repr(self):
        repr = "Nodes:\n"
        for node in self.thought_graph.nodes:
            content = self.thought_graph.nodes[node]
            content = json.dumps(content, indent=4)
            # content = content.replace("\\n", "\n")
            repr += f"{node}: {content}\n"
        
        repr += "Edges:\n"
        for edge in self.thought_graph.edges:
            repr += f"{edge}: {self.thought_graph.edges[edge]}\n"
        return repr
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reinitalize the thought graph
        self.thought_graph = nx.Graph()
        if self.task_name == "human_eval":
            kwargs = {
                "problem": self.problem,
                "problem_idx": self.idx,
            }
        else:
            kwargs = {
                "thought": self.problem
            }
            
        self.thought_graph.add_node(
            0, 
            **kwargs,
        )

        # observation = self._get_obs()
        info = self._get_info()

        return self.thought_graph, info
    
    def step(
        self, 
        action,
        max_tries: int = 1,
    ):
        operation = action["operation"]
        nodes = action["nodes"]
        multiplicity = action.get("attempts", 1)
        explanation = action.get("explanation", "")

        print(f"\nStep {self.step_count}")
        print(f"========================")
        print(f"Action: {operation}")
        print(f"Nodes: {nodes}")
        print(f"Attempts: {multiplicity}")
        print(f"Explanation: {explanation}\n")

        operator = getattr(self.reasoning_agent, operation, None)
        if operator is None:
            raise ValueError(f"Operation {operation} not found for task {self.reasoning_agent}")

        tries = 1
        success = False

        while tries <= max_tries:
            # try:
            # Currently aggregate operation is the only one that accepts a multiplicity
            if operation == "aggregate":
                kwargs = {
                    "graph": self.thought_graph,
                    "nodes": nodes,
                    "model": self.model,
                    "multiplicity": multiplicity,
                }
                self.thought_graph, terminate = operator(**kwargs)
            else:
                for _ in range(multiplicity):
                    kwargs = {
                        "graph": self.thought_graph,
                        "nodes": nodes,
                        "model": self.model,
                        "multiplicity": multiplicity,
                    }
                    self.thought_graph, terminate = operator(**kwargs)
            truncate = False
            success = True
            break
            # except:
            #     print(f"[{tries}/{max_tries}]: Operation {operation} failed.")
            #     tries += 1

        if not success:
            raise Exception(f"Operation {operation} failed on nodes {nodes} after {max_tries} attempts")

        print("\nGraph state:")
        print(f"------------------------")
        print(self.thought_graph_repr())

        # An environment is completed if there is a ground truth proposal with a score of 1
        reward = 1 if terminate else 0  # the agent is only reached at the end of the episode
        info = self._get_info()

        self.step_count += 1

        if operation == "groundtruth":
            score = self.thought_graph.nodes[int(nodes[0])]["score"]
        else:
            score = None

        info["score"] = score
        
        return reward, terminate, truncate, info

    async def async_step(
        self, 
        action,
        max_tries: int = 1,
    ):
        return await asyncio.to_thread(self.step, action, max_tries)