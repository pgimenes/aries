from llm import llm, async_llm
from .common import (
    _common_tot_schedule,
    _common_got_schedule,
    common_keepbest,
    PARSE_OUT_DICT,
)
import asyncio
from copy import copy
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
import regex as re
from traceback import format_exc
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution exceeded time limit of 1 minute.")

problem_definition = "HumanEval: Given a programming problem, provide a solution that passes the test cases."

actions = {
    "split": {
        "description": "Split the problem into subproblems",
        "precondition": "",
        "effects": "A new node is added for each subproblem",
    },
    "generate": {
        "description": "Generate a solution for a problem",
        "precondition": "",
        "effects": "The solution is added to the node",
    },
    "refine": {
        "description": "Refine a solution",
        "precondition": "",
        "effects": "The solution is refined",
    },
    "score": {
        "description": "Score a solution",
        "precondition": "",
        "effects": "A node is scored with 1 if the solution passes the test cases, and 0 otherwise",
    },
    "keepbest": {
        "description": "Out of the selected nodes, keep the one with the highest score, and delete the rest.",
        "preconditions": "The selected nodes must have been scored.",
        "effects": "All selected nodes are deleted, but the one with the highest score is duplicated as a new node.",
    },
    "aggregate": {
        "description": "Aggregate the selected nodes into a single solution",
        "preconditions": "",
        "effects": "The selected nodes are combined into a single node.",
    },
}

examples = [
    """
<example>
INPUT:

Previous actions:
Action 0: generate
Nodes: [0]
Explanation: We are generating a solution for the problem in node 0. This is the first step in solving the problem. We will use the generated solution as a starting point and refine it if necessary.


Current graph:
Nodes:
0: {
    "problem": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "problem_idx": 0,
    "solution": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n\n    return any(abs(numbers[i] - numbers[j]) < threshold for i in range(len(numbers)) for j in range(i+1, len(numbers)))\n",
    "feedback": null,
    "score": null
}
Edges:

OUTPUT:

<analysis>
The current state of the thought graph is that we have generated a solution for the problem in node 0. The previous action, "generate", was taken on node 0 with the strategy of using the generated solution as a starting point and refining it if necessary.

The nodes and edges in the graph are:
- Node 0: This node contains the problem statement, a generated solution, and no feedback or score.
- Edges: There are no edges in the graph, as this is a single node.

The strategy outlined in the previous action is pending, as we do not have any feedback or score on the generated solution. The next step would be to score the solution to determine its accuracy.

Alternative actions that could be taken next are:
- score: Score the solution in node 0 to determine its accuracy.
- refine: Refine the solution in node 0 without scoring it first.
- split: Split the problem into subproblems, but this would not make sense given that we have already generated a solution for the problem.

</analysis>

<next_action>score</next_action>

<nodes>[0]</nodes>

<attempts>1</attempts>

<explanation>
We choose the "score" action on node 0 because we need to determine the accuracy of the generated solution. This is the next logical step in solving the problem, as it will give us feedback on the solution's correctness. If the solution scores high, we can continue refining it. If it scores low, we may need to explore alternative solutions. This action continues the current strategy of using the generated solution as a starting point and refining it if necessary.
</explanation>

</example>  
"""
]


class HumanEvalAgent:
    def __init__(
        self,
        temperature=None,
    ):

        self.temperature = temperature if temperature is not None else 1.0

        self.split_prompt = """<instruction>You are a programming expert. Your role is to outline a skeleton implementation of the function according to the docstring. This skeleton should call functions that are so far not defined. Then, you should list all the functions that need to be defined.

The output should be within <output> ... </output> tags, as shown in the example. You should first output the skeleton in <skeleton> tags. Then, for each function in the skeleton, output its header and docstring in <function> tags. Also include test cases for each function in <testcase> tags.
</instruction>

<example>

<prompt>
def odd_numbers_sum_to_10(lst):
    '''
    Extracts the odd numbers from a given list of numbers
    and returns “Yes” if the sum of the odd numbers is 10, otherwise “No”.
    '''
    pass
</prompt>

<output>

<skeleton>
    odd_numbers = get_odd_numbers(lst)
    sum_odds = sum_numbers(odd_numbers)
    return “Yes” if sum_odds == 10 else “No”
</skeleton>

<function>

<docstring>
def get_odd_numbers(lst):
    '''
    Extracts the odd numbers.
    '''
    pass
</docstring>

<testcase>
assert get_odd_numbers([1, 2, 3, 4, 5]) == [1, 3, 5]
assert get_odd_numbers([2, 4, 6, 8]) == []
</testcase>

</function>

<function>

<docstring>
def sum_numbers(lst):
    '''
    Calculates the sum of numbers.
    '''
    pass
</docstring>

<testcase>
assert sum_numbers([1, 2, 3, 4, 5]) == 15
assert sum_numbers([2, 4, 6, 8]) == 20
</testcase>

</function>

</output>

</example>

Now you go.

<prompt>
{input}
</prompt>
"""

        self.solve_prompt = """<instruction>You are a programming expert. Your role is to complete the function definition according to the docstring.

The output should be within <output> ... </output> tags, as shown in the example. Do not repeat the docstring in the output.
</instruction>

<example>

<prompt>
def truncate_number(number: float) -> float: 
    \"\"\" Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1). Return the decimal part of the number. 
    >>> truncate_number(3.5) 
    0.5 
    \"\"\"
</prompt>

<output>
    return number % 1.0
</output>

</example>

Now you go.

<prompt>
{input}
</prompt>
"""

        self.refine_prompt = """<instruction>You are a programming expert. You are given a programming problem defined in the docstring of a function. Given a candidate solution and the execution output, your role is to refine the solution to pass the test cases.

The output should be within <output> ... </output> tags, as shown in the example. Do not repeat the docstring in the output.
</instruction>

<example>

<candidate>
def celsius_to_fahrenheit(celsius):
    '''
    Convert temperature from Celsius to Fahrenheit.

    Parameters:
    celsius (float): Temperature in Celsius.

    Returns:
    float: Temperature in Fahrenheit.
    '''
    return (celcius * 9/5) + 32

assert celsius_to_fahrenheit(0) == 32
</candidate>

<feedback>
Traceback (most recent call last):
  File "/home/user/find-closest-pair.py", line 13, in <module>
    assert celsius_to_fahrenheit(0) == 32
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/find-closest-pair.py", line 11, in celsius_to_fahrenheit
    return (celcius * 9/5) + 32
            ^^^^^^^
NameError: name 'celcius' is not defined. Did you mean: 'celsius'?
</feedback>

<output>
    return (celsius * 9/5) + 32
</output>

</example>

Now you go.

<candidate>
{candidate}
</candidate>

<feedback>
{feedback}
</feedback>
"""

    def split(
        self,
        graph,
        nodes,
        model="",
        run_async=False,
        multiplicity: int = 1,
    ):

        outs = {
            node: llm(
                self.split_prompt.format(input=graph.nodes[int(node)]["problem"]),
                model=model,
                temperature=self.temperature,
            )[0]
            for node in nodes
        }

        for node in nodes:
            next_thought = outs[node]

            # remove the <output> tags
            next_thought = next_thought.replace("<output>", "")
            next_thought = next_thought.replace("</output>", "")

            # get the skeleton
            skeleton = re.search(r"<skeleton>(.*?)</skeleton>", next_thought, re.DOTALL).group(1)
            graph.nodes[int(node)]["solution"] = graph.nodes[int(node)]["problem"] + skeleton
    
            # find the name of the function
            function_name = re.search(r"def (.*?)\(", graph.nodes[int(node)]["problem"]).group(1)
            graph.nodes[int(node)]["function_name"] = function_name

            # get the content of each <function> tag
            functions = re.findall(
                r"<function>(.*?)</function>", next_thought, re.DOTALL
            )

            for function in functions:

                # get the docstring
                docstring = re.search(
                    r"<docstring>(.*?)</docstring>", function, re.DOTALL
                ).group(1)

                # get the testcases
                testcases = re.search(
                    r"<testcase>(.*?)</testcase>", function, re.DOTALL
                ).group(1)

                idx = max(list(graph.nodes)) + 1
                graph.add_node(
                    idx,
                    problem=docstring,
                    testcases=testcases,
                    score=None,
                )
                graph.add_edge(int(node), idx)

        return graph, False

    async def async_generate(
        self,
        graph,
        nodes,
        model="",
        multiplicity: int = 1,
    ):
        return await asyncio.gather(
            *[
                async_llm(
                    self.solve_prompt.format(input=graph.nodes[int(node)]["problem"]),
                    model=model,
                    temperature=self.temperature,
                )
                for node in nodes
            ],
        )

    def generate(
        self,
        graph,
        nodes,
        model="",
        run_async=True,
        multiplicity: int = 1,
    ):

        # 1. Get LLM responses
        if run_async:
            outs = asyncio.run(self.async_generate(graph, nodes, model=model))
            outs = {nodes[i]: outs[i] for i in range(len(nodes))}
        else:
            outs = {
                node: llm(
                    sort_prompt.format(input=graph.nodes[int(node)]["problem"]),
                    model=model,
                    temperature=self.temperature,
                )[0]
                for node in nodes
            }

        # 2. Update graph
        nodes_to_score = []
        for node in nodes:
            next_thought = outs[node][0]

            # remove the <output> tags
            next_thought = next_thought.replace("<output>", "")
            next_thought = next_thought.replace("</output>", "")

            graph.nodes[int(node)]["solution"] = (
                graph.nodes[int(node)]["problem"].replace("pass", "") + next_thought
            )

            # Reset feedback and score
            graph.nodes[int(node)]["feedback"] = None
            graph.nodes[int(node)]["score"] = None

        return graph, False

    async def async_refine(
        self,
        graph,
        nodes,
        model="",
        multiplicity: int = 1,
    ):
        return await asyncio.gather(
            *[
                async_llm(
                    self.refine_prompt.format(
                        candidate=graph.nodes[int(node)]["solution"],
                        feedback=graph.nodes[int(node)].get("feedback", ""),
                    ),
                    model=model,
                    temperature=self.temperature,
                )
                for node in nodes
            ],
        )

    def refine(
        self,
        graph,
        nodes,
        model="",
        run_async=True,
        multiplicity: int = 1,
    ):
        # 1. Get LLM responses
        if run_async:
            outs = asyncio.run(self.async_refine(graph, nodes, model=model))
            outs = {nodes[i]: outs[i] for i in range(len(nodes))}
        else:
            outs = {
                node: llm(
                    self.refine_prompt.format(
                        candidate=graph.nodes[int(node)]["solution"],
                        feedback=graph.nodes[int(node)].get("feedback", ""),
                    ),
                    model=model,
                    temperature=self.temperature,
                )[0]
                for node in nodes
            }

        # 2. Update graph
        nodes_to_score = []
        for node in nodes:
            sol = outs[node][0]

            # remove the <output> tags
            sol = sol.replace("<output>", "")
            sol = sol.replace("</output>", "")

            graph.nodes[int(node)]["solution"] = graph.nodes[int(node)]["problem"] + sol

            # Reset feedback and score
            graph.nodes[int(node)]["feedback"] = None
            graph.nodes[int(node)]["score"] = None

        return graph, False

    def _score_full_solution(
        self,
        graph,
        node,
    ):
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        problem_idx = graph.nodes[0].get("problem_idx", None)
        if problem_idx is None:
            raise ValueError(
                "Problem index not found in the node: {}".format(graph_node)
            )

        sample = [
            {
                "task_id": f"HumanEval/{problem_idx}",
                "completion": graph_node["solution"],
            }
        ]
        write_jsonl("sample.jsonl", sample)

        out = evaluate_functional_correctness("sample.jsonl", ignore_incomplete=True)

        score = 1 if out["pass@1"] == 1.0 else 0

        graph.nodes[node_idx]["score"] = score

        return graph, score

    def score(
        self,
        graph,
        nodes,
        model="",
        multiplicity: int = 1,
    ):

        any_pass = False
        for node in nodes:

            # # If scoring the full problem, fall back to HumanEval code
            # if node in ["0", 0] or graph.nodes[int(node)].get("is_solution", False):
            #     graph, score = self._score_full_solution(graph, node)
                
            #     if score > 0:
            #         any_pass = True
                
            #     continue

            # Evaluate testcases
            testcases = graph.nodes[int(node)].get("testcases", [])
            
            # testcases is a string when called on a subproblem solution
            if not isinstance(testcases, list):
                try:
                    testcases = re.findall(r'assert (.*)\n', testcases)
                except:
                    testcases = []

            num_failed = 0
            graph.nodes[int(node)]["feedback"] = ""
            for testcase in testcases:

                program = (
                    "from typing import *\n"
                    + graph.nodes[int(node)].get("solution", "")
                    + f"\nassert {testcase}"
                )

                # Set the timeout to 60 seconds
                # This will get triggered if the code requires more inputs than necessary
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # Set alarm for 60 seconds

                try:
                    exec(program, {})
                except Exception as exc:
                    # get traceback
                    num_failed += 1
                    print(f"testcase {testcase} failed with exception: {exc}")
                    graph.nodes[int(node)]["feedback"] += f"\ntestcase {testcase} failed with exception: {exc}"
                    continue

                finally:
                    signal.alarm(0)

            graph.nodes[int(node)]["score"] = num_failed

        return graph, False

    def keepbest(
        self,
        graph,
        nodes,
        model="",
        multiplicity: int = 1,
    ):
        # Score all non-scored nodes
        # graph, _ = score(graph, nodes, model=model)
        return common_keepbest(graph, nodes, model)

    def aggregate(
        self,
        graph,
        nodes,
        model="",
        multiplicity: int = 1,
        run_async: bool = False,
    ):

        # 1. Collect code snippets and testcases
        code = ""
        for node in nodes:
            code += graph.nodes[int(node)]["solution"]

        testcases = ""
        if "0" not in nodes and 0 not in nodes:
            for node in nodes:
                tests = graph.nodes[int(node)].get("testcases", "")
                if isinstance(tests, list):
                    tests = "\n".join(tests)
                testcases += tests
        else:
            testcases = graph.nodes[0].get("testcases", "")
        
        # add new node
        idx = max(list(graph.nodes)) + 1

        graph.add_node(
            idx,
            problem=graph.nodes[0]["problem"],
            solution=code,
            testcases=testcases,
            score=None,
            is_solution=True,
            problem_idx=graph.nodes[0].get("problem_idx", None),
        )

        # score aggregated node
        graph, passed = self.score(graph, [idx], model=model)

        return graph, passed

    # Baselines
    def io(
        self,
        graph,
        nodes,
        model="",
        run_async=False,
        multiplicity: int = 1,
    ):
        return self.generate(graph, nodes, model)

    def cot(
        self,
        graph,
        nodes,
        model="",
    ):
        raise NotImplementedError("Cot not implemented for human eval")

    def groundtruth(
        self,
        graph, 
        nodes,
        model = "",
        run_async = True,
        multiplicity: int = 1,
    ):
        for node in nodes:
            if not graph.nodes[int(node)].get("is_solution", False):
                continue
            
            # Only score if not already scored
            score = graph.nodes[int(node)].get("score", None)
            if score is None:
                _, score = self.score(graph, [node], model=model)

            # Skip checking other nodes if a solution was found
            if score == 0:
                graph.nodes[int(node)]["groundtruth"] = True
                return graph, True

            graph.nodes[int(node)]["groundtruth"] = False

        return graph, False
