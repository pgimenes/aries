import gymnasium as gym
from llm import llm
import re
    
class IOAgent:
    def __init__(
        self,
        env: gym.Env,
    ):
        self.env = env

        self.itr = 0
        self._action_list = [
            "sort",
            "groundtruth",
        ]
        self._node_list = [
            "0",
            "1",
        ]

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        action = {
            "nodes": self._node_list[self.itr],
            "operation": self._action_list[self.itr],
            "explanation": "",
        }
        self.itr += 1
        return action
    
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
                "sort_cot",
                "groundtruth",
            ]
            self._node_list = [
                "0",
                "1",
            ]

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        action = {
            "nodes": self._node_list[self.itr],
            "operation": self._action_list[self.itr],
            "explanation": "",
        }
        self.itr += 1
        return action
    
class ToTAgent:
    def __init__(
        self,
        env: gym.Env,
        model: str,
        problem_definition: str,
        actions: dict[str, str],
        width: int,
        depth: int,
    ):
        self.env = env
        self.model = model
        self.problem_definition = problem_definition
        self.actions = actions
        self.width = width
        self.depth = depth

        self._actions, self._action_nodes = self._get_action_list()
        self.itr = 0


    def get_action(self, obs: tuple[int, int, bool]) -> int:
        action, action_nodes = self._actions[self.itr], self._action_nodes[self.itr]
        self.itr += 1
        return {
            "nodes": action_nodes,
            "operation": action,
            "explanation": "",
        }

    def _get_action_list(self) -> int:
        actions = []
        action_nodes = []
        keepbest_nodes = []
        last_node = 0

        # Sorting
        actions += ["sort"]
        action_nodes += [["0"] * self.width]
        last_node += self.width

        # Score
        score_nodes = [str(i) for i in range(1, self.width + 1)]
        actions += ["score"]
        action_nodes += [score_nodes]

        # Keep best
        actions += ["keepbest"]
        action_nodes += [[str(i) for i in range(1, self.width + 1)]]
        last_node += 1
        keepbest_nodes.append(str(last_node))

        for i in range(self.depth - 1):
            # Refine
            refine_node = last_node
            actions += ["refine"]
            action_nodes += [[str(refine_node)] * self.width]
            last_node += self.width

            # Score
            score_nodes = [str(j) for j in range(last_node - self.width + 1, last_node + 1)]
            actions += ["score"]
            action_nodes += [score_nodes]

            # Keep best
            actions += ["keepbest"]
            action_nodes += [[str(j) for j in range(last_node - self.width + 1, last_node + 1)]]
            last_node += 1
            keepbest_nodes.append(str(last_node))

        # Keep best
        actions += ["keepbest"]
        action_nodes += [keepbest_nodes]
        last_node += 1

        # Ground truth
        actions += ["groundtruth"]
        action_nodes += [[str(last_node)]]
        last_node += 1
        return actions, action_nodes

class GoTAgent:
    def __init__(
        self,
        env: gym.Env,
        model: str,
        problem_definition: str,
        actions: dict[str, str],
    ):
        self.env = env
        self.model = model

        self._got_action = 0
        self._got_action_list = self._get_action_list()

        self.problem_definition = problem_definition
        self.actions = actions

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        actions, action_nodes = self._got_action_list
        action = {
            "nodes": action_nodes[self._got_action],
            "operation": actions[self._got_action],
            "explanation": "",
        }
        self._got_action += 1
        return action
    
    def _get_action_list(self) -> int:        
        split_branches = 2
        sort_attempts = 5

        # Create two split branches
        actions = ["split"]
        action_nodes = [["0"]]

        last_node = 2
        keepbest_nodes = []
        for split_branch in range(1, split_branches + 1):
            
            # Sorting
            sorted_nodes = []
            actions += ["sort"]
            action_nodes += [[str(split_branch)] * sort_attempts]
            sorted_nodes += list(range(last_node + 1, last_node + 1 + sort_attempts))
            last_node += sort_attempts

            # Scoring
            actions += ["score"]
            action_nodes += [sorted_nodes]

            # Keep best
            actions += ["keepbest"]
            action_nodes += [sorted_nodes]
            last_node += 1
            keepbest_nodes += [str(last_node)]

        # Aggregate
        actions += ["aggregate"]
        action_nodes += [keepbest_nodes]
        last_node += 1

        # Groundtruth
        actions += ["groundtruth"]
        action_nodes += [[str(last_node)]]
        last_node += 1
        return actions, action_nodes
    
class LLMAgent:
    def __init__(
        self,
        env: gym.Env,
        model: str,
        problem_definition: str,
        actions: dict[str, str],
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