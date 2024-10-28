import gymnasium as gym
from llm import llm
import re

class LLMAgent:
    def __init__(
        self,
        env: gym.Env,
        model: str,
        problem_definition: str,
        actions: dict[str, str]
    ):
        self.env = env
        self.model = model

        self.problem_definition = problem_definition
        self.actions = actions

        self.prompt = """
<Instruction>
You are a reasoning agent responsible for solving a problem by guiding the exploration of a thought graph.
In each iteration, I will provide the current state of the thought graph. You need to choose a subset of the existing nodes to perform an action on, and define which action to perform.

Problem definition: {problem_definition}

The following actions are available:
{actions}

Here is the current state of the graph:
{graph}

The starting problem is contained in node 0. If you think one of the nodes contains the correct solution, you can choose the 'groundtruth' operation to compare it with the ground truth.
It's possible this node is already in the graph, or you may need to create it by performing other operations.

Your output should be in the format:

Explanation: ...
Nodes: [...]
Operation: ...

</Instruction>

<example>
For example:
Explanation: Nodes 3 and 4 are correctly sorted sublists, so we will aggregate them.
Nodes: [3, 4]
Operation: aggregate
</example>

What nodes do you choose next, and what operation would you like to perform?
"""

    def get_action(self, obs: tuple[int, int, bool]) -> int:        
        graph_repr = self.env.thought_graph_repr()
        prompt = self.prompt.format(
            problem_definition=self.problem_definition,
            actions="\n".join([f"{k}: {v}" for k, v in self.actions.items()]),
            graph=graph_repr
        )
        res = llm(prompt, model=self.model)

        try:
            match = re.search(r"Explanation: (.*?)\n*Nodes: \[(.*)\]\n*Operation: (\w+).*", res[0], re.DOTALL)
            explanation = match.group(1)
            nodes = match.group(2)
            operation = match.group(3)
        except:
            raise ValueError(f"Could not parse the output: {res[0]}")

        action = {
            "nodes": nodes.split(","),
            "operation": operation,
            "explanation": explanation,
        }

        return action