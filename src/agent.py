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
        self.task_examples = getattr(task, "examples")

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
        self.task_examples = getattr(task, "examples")

        self.max_iterations = 25

        self.action_history = []

        self.prompt = """<Instruction> You are a perspicacious strategy planning agent responsible for solving a problem by guiding the exploration of a thought graph. The starting problem is contained in node 0 of the graph. You must choose a subset of the existing nodes to perform an action on, and define which action to perform.
        
        Your input is:
        1. The history of all previously taken actions, which nodes the actions were taken on, and an explanation of the strategy for each action.
        2. A representation of the current thought graph, including all nodes and edges.

        Your instructions are:
        1. Provide a detailed analysis of the current state of the thought graph and the strategy towards solving the problem.
            a. Provide a detailed description of the action history. Explain the strategy behind each action, and how they contribute to solving the problem.

            b. Provide a detailed description of the nodes and edges in the graph. Explain how each node corresponds to previous actions.
            
            c. Explain whether the strategy outlined in previous actions is successful, unsuccessful, or pending. If the strategy is successful, outline what would be the next steps to reach the solution. If the strategy is unsuccessful, explain why, and outline alternative actions based on this feedback. If still pending, outline which steps are still required to continue exploring the current strategy.
            
        2. Choose the next action to take and which nodes the action should be performed on.

        3. Provide an explanation for the chosen action and nodes. Outline the reasoning behind the choice by reiterating the current strategy to finding a solution to the problem. Explain whether the chosen action is continuing the current strategy, refining it, or exploring a new direction.
        
        Additional instructions:
        - If you think one of the nodes contains the correct solution, you can choose the 'groundtruth' operation to compare it with the ground truth. It's possible this node is already in the graph, or you may need to create it by performing other operations.

Problem definition: {problem_definition}
The following actions are available:

{actions}</Instruction>

{examples}

INPUT:

Previous actions:
{history}
Current graph:
{graph}
OUTPUT:"""

    def _format_action_history(self):
        history = ""
        for idx, action in enumerate(self.action_history):
            history += f"Action {idx}: {action['operation']}\n"
            history += f"Nodes: {[int(node) for node in action['nodes']]}\n"
            history += f"Explanation: {action['explanation']}\n"
        return history

    def _format_action_list(self):
        actions = ""
        for action, obj in self.actions.items():
            actions += f"Action: {action}\n"
            for k, v in obj.items():
                actions += (f"    {k}: {v}\n")
            actions += "\n"

        return actions


    def get_action(self, obs: tuple[int, int, bool]) -> int:        
        graph_repr = self.env.thought_graph_repr()
        prompt = self.prompt.format(
            problem_definition=self.problem_definition,
            actions=self._format_action_list(),
            examples=self.task_examples,
            history=self._format_action_history(),
            graph=graph_repr,
        )

        attempts = 1
        action = None
        
        while True:
            res = llm(prompt, model=self.model)

            if attempts > 5:
                break

            try:
                match = re.search(
                    r"Analysis:\s*(.*?)\s*Next action:\s*(\w+)\s*Nodes:\s*\[([0-9,\s]+)]\s*Explanation:\s*(.*)",
                    res[0],
                    re.DOTALL
                )

                analysis = match.group(1)
                operation = match.group(2)
                nodes = match.group(3)
                explanation = match.group(4)
                
                action = {
                    "nodes": [int(node) for node in nodes.split(",")],
                    "operation": operation,
                    "explanation": explanation,
                    "analysis": analysis,
                }

                break
            except Exception as exc:
                print(f"[{attempts} / 5] Failed to parse LLM output: {exc}")
                attempts += 1
                pass

        if action is None:
            raise Exception("Failed to parse LLM output after 5 attempts")
        
        return action