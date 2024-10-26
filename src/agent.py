import gymnasium as gym
from llm import llm
import re

class LLMAgent:
    def __init__(
        self,
        env: gym.Env,
    ):
        self.env = env
        self.start_prompt = """
You are a reasoning agent responsible for solving a number sorting problem by guiding the exploration of a thought graph.
In each iteration, I will provide the current state of the thought graph. You need to choose a subset of the existing nodes to perform an action on, and define which action to perform.

The following actions are available:

- split: Split a sublist into two to decompose the problem.

- sort: Sort a sublist.

- refine: Refine a sublist by fixing any existing mistakes.

- score: count the number of mistakes in the currently sorted sublist.

- keepbest: out of the selected nodes, keep the one with the highest score, and delete the rest. You should only take this action on nodes that have been scored.

- aggregate: merge the sorted sublists of the selected nodes into a single sorted list.

- groundtruth: compare the sorted list in a node with the ground truth.

Here is the current state of the graph:
{}

The starting problem is contained in node 0. If you think one of the nodes contains the correct solution, you can choose the 'groundtruth' operation to compare it with the ground truth.
It's possible this node is already in the graph, or you may need to create it by performing other operations.

Your output should be in the format:

Nodes: [...]
Operation: ...

What nodes do you choose next, and what operation would you like to perform?
"""

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        graph_repr = self.env.thought_graph_repr()
        prompt = self.start_prompt.format(graph_repr)
        res = llm(prompt)

        try:
            match = re.search(r"Nodes: \[(.*)\]\nOperation: (.*)", res[0], re.DOTALL)
            nodes = match.group(1)
            operation = match.group(2)
        except:
            breakpoint()

        action = {
            "nodes": nodes.split(","),
            "operation": operation,
        }

        return action