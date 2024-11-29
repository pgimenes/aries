import gymnasium as gym
from llm import llm, async_llm
import re
import asyncio
    
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
        self._actions, self._action_nodes, self._attempts = schedule(
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
            "attempts": self._attempts[self._got_action],
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
        max_iterations: int = None,
        cot_sc_branches: int = 1,
    ):
        self.env = env
        self.task = task
        self.model = model
        self.max_iterations = max_iterations if max_iterations is not None else 25
        self.cot_sc_branches = cot_sc_branches

        # Things injected into the agent prompt
        self.problem_definition = problem_definition
        self.actions = actions
        self.task_examples = getattr(task, "examples")
        self.additional_instructions = getattr(task, "additional_instructions") if hasattr(task, "additional_instructions") else ""

        self.action_history = []

        self.prompt = """<Instruction> You are a perspicacious strategy planning agent responsible for solving a problem by guiding the exploration of a thought graph. The starting problem is contained in node 0 of the graph. You must choose a subset of the existing nodes to perform an action on, and define which action to perform.
        
Your input is:
1. The history of all previously taken actions, which nodes the actions were taken on, and an explanation of the strategy for each action.
2. A representation of the current thought graph, including all nodes and edges.

Your instructions are:
1. Provide a detailed analysis of the current state of the thought graph and the strategy towards solving the problem.
    a. Provide a detailed description of the action history. Explain the strategy behind each action, and how they contribute to solving the problem.

    b. Provide a detailed description of the nodes and edges in the graph. Explain how each node corresponds to previous actions.
    
    c. Explain whether the strategy outlined in previous actions is successful, unsuccessful, or pending. If the strategy is unsuccessful, explain why, and outline alternative actions based on this feedback. If still pending, outline which steps are still required to continue exploring the current strategy.

    d. Provide a few alternatives of actions that could be taken next, and explain the reasoning behind each alternative.
    
2. Choose the next action to take on the thought graph

3. Choose the node or nodes to perform the action on

4. Specify the number of times to attempt the chosen action on the selected nodes.

5. Provide an explanation for the chosen action and nodes. Outline the reasoning behind the choice by reiterating the current strategy to finding a solution to the problem. Explain whether the chosen action is continuing the current strategy, refining it, or exploring a new direction.

Additional instructions:
- The format of the output should match the examples. The analysis should be wrapped by <analysis> tags. The next action should be wrapped by <next_action> tags. The nodes should be wrapped by <nodes> tags. The number of attempts should be wrapped by <attempts> tags. The explanation should be wrapped by <explanation> tags.

- If you think one of the nodes contains the correct solution, you can choose the 'groundtruth' operation to compare it with the ground truth. It's possible this node is already in the graph, or you may need to create it by performing other operations.

{additional_instructions}

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
            history += f"Explanation: {action['explanation']}\n\n"
        return history

    def _format_action_list(self):
        actions = ""
        for action, obj in self.actions.items():
            actions += f"Action: {action}\n"
            for k, v in obj.items():
                actions += (f"    {k}: {v}\n")
            actions += "\n"

        return actions


    async def _generate_action(self):
        graph_repr = self.env.thought_graph_repr()
        prompt = self.prompt.format(
            problem_definition=self.problem_definition,
            actions=self._format_action_list(),
            examples=self.task_examples,
            history=self._format_action_history(),
            graph=graph_repr,
            additional_instructions=self.additional_instructions,
        )

        get_action_attempts = 1
        action = None
        while get_action_attempts <= 10:
            res = await async_llm(prompt, model=self.model)
            try:
                match = re.search(
                    r"(?i)<analysis>\s*(.*?)\s*</analysis>\s*<next_action>\s*(\w+)\s*</next_action>\s*<nodes>\s*\[([0-9,\s]+)]\s*</nodes>\s*<attempts>\s*([0-9,\s]+)\s*</attempts>\s*<explanation>\s*(.*?)\s*</explanation>",
                    res[0],
                    re.DOTALL
                )

                analysis = match.group(1)
                operation = match.group(2)
                nodes = match.group(3)
                num_attempts = match.group(4)
                explanation = match.group(5)

                action = {
                    "analysis": analysis,
                    "operation": operation,
                    "nodes": [int(node) for node in nodes.split(",")],
                    "attempts": int(num_attempts),
                    "explanation": explanation,
                }
                break
            except Exception as exc:
                print(f"[{get_action_attempts} / 5] Failed to parse LLM output: {exc}")
                get_action_attempts += 1

        if action is None:
            raise Exception("Failed to parse LLM output after 5 attempts")
        
        return action


    async def get_action(self, obs: tuple[int, int, bool]) -> int:        
        graph_repr = self.env.thought_graph_repr()
        prompt = self.prompt.format(
            problem_definition=self.problem_definition,
            actions=self._format_action_list(),
            examples=self.task_examples,
            history=self._format_action_history(),
            graph=graph_repr,
            additional_instructions=self.additional_instructions,
        )

        action_proposals = []
        
        # Gather action proposals, ignoring exceptions
        action_proposals = await asyncio.gather(
            *[self._generate_action() for _ in range(self.cot_sc_branches)], 
            return_exceptions=True
        )
        action_proposals = [result for result in action_proposals if not isinstance(result, Exception)]

        # Take action with highest occurance
        vote_dict = {}
        for action in action_proposals:
            key = (action["operation"], tuple(action["nodes"]), action["attempts"])
            vote_dict[key] = vote_dict.get(key, 0) + 1
        print(f"Action Votes: {vote_dict}")
        highest_vote = max(vote_dict, key=vote_dict.get)

        explanation = ""
        attempts = 1
        for action in action_proposals:
            if (action["operation"], tuple(action["nodes"]), action["attempts"]) == highest_vote:
                explanation = action["explanation"]

        action = {
            "operation": highest_vote[0],
            "nodes": list(highest_vote[1]),
            "attempts": highest_vote[2],
            "explanation": explanation,
        }
        return action