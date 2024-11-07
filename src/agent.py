import gymnasium as gym
from llm import llm
import re
    
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

class GoTAgent:
    def __init__(
        self,
        env: gym.Env,
        task,
        model: str,
        problem_definition: str,
        actions: dict[str, str],
        
        # GoT parameters
        branches:int,
        generate_attempts:int,
        aggregate_attempts:int,
        post_aggregate_keepbest: bool,
        post_aggregate_refine: bool,
        refine_attempts:int,
    ):
        self.env = env
        self.task = task
        self.model = model

        schedule = getattr(task, "_got_schedule")
        self._got_action = 0
        self._actions, self._action_nodes = schedule(
            branches,
            generate_attempts,
            aggregate_attempts,
            post_aggregate_keepbest,
            post_aggregate_refine,
            refine_attempts,
        )

        self.problem_definition = problem_definition
        self.actions = actions

        self.max_iterations = len(self._actions)

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        action = {
            "nodes": self._action_nodes[self._got_action],
            "operation": self._actions[self._got_action],
            "explanation": "",
        }
        self._got_action += 1
        return action
    
class LLMAgent:
    def __init__(
        self,
        env: gym.Env,
        task,
        model: str,
        problem_definition: str,
        actions: dict[str, str],
    ):
        self.env = env
        self.task = task

        self.model = model
        self.problem_definition = problem_definition
        self.actions = actions

        self.max_iterations = 100

        self.action_history = []

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

Here is the history of previously chosen actions:
{history}

What nodes do you choose next, and what operation would you like to perform?
"""

    def _get_history(self):
        history = ""
        for idx, action in enumerate(self.action_history):
            history += f"Action {idx}:\n"
            history += f"Operation: {action['operation']}\n\n"
            history += f"Nodes: {action['nodes']}\n"
            history += f"Explanation: {action['explanation']}\n"
        return history

    def get_action(self, obs: tuple[int, int, bool]) -> int:        
        graph_repr = self.env.thought_graph_repr()
        prompt = self.prompt.format(
            problem_definition=self.problem_definition,
            actions="\n".join([f"{k}: {v}" for k, v in self.actions.items()]),
            graph=graph_repr,
            history=self._get_history(),
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

        self.action_history.append(action)

        return action