from typing import Optional
import numpy as np

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import Graph, Box, Discrete, Sequence, Dict, GraphInstance
import networkx as nx

import logging
import tasks.sorting as task

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class GoTEnv(gym.Env):

    def __init__(
            self,
            problem: str = "",
            max_graph_size: int = 128,
            node_embedding_size: int = 128,
        ):

        self.step_count = 0
        self.max_graph_size = max_graph_size
        self.node_embedding_size = node_embedding_size
        self.problem = problem

        self.thought_graph = nx.Graph()
        self.thought_graph.add_node(0, embedding=np.zeros(node_embedding_size))

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
            repr += f"{node}: {self.thought_graph.nodes[node]}\n"
        repr += "Edges:\n"
        for edge in self.thought_graph.edges:
            repr += f"{edge}: {self.thought_graph.edges[edge]}\n"
        return repr
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reinitalize the thought graph
        self.thought_graph = nx.Graph()
        self.thought_graph.add_node(
            0, 
            thought = self.problem,
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        operation = action["operation"]
        nodes = action["nodes"]

        print(f"\nStep {self.step_count}")
        print(f"Operation: {operation}")
        print(f"Nodes: {nodes}\n")

        try:
            operator = getattr(task, operation)
        except AttributeError:
            breakpoint()

        self.thought_graph, terminate = operator(self.thought_graph, nodes)
        truncate = False

        print("\nGraph state:")
        print(self.thought_graph_repr())

        # An environment is completed if there is a ground truth proposal with a score of 1
        reward = 1 if terminate else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        self.step_count += 1

        return observation, reward, terminate, truncate, info