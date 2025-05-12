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
from unittest.mock import patch
import io
import contextlib
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

examples = [""]

class CodeContestsAgent:
    def __init__(
        self,
        temperature = None,
    ):

        from datasets import load_dataset
        self.ds = load_dataset("deepmind/code_contests")
        self.temperature = temperature if temperature is not None else 1.0
        self.split_prompt = """<instruction>You are a programming expert. Your role is to outline a skeleton implementation of the function according to the docstring. This skeleton should call functions that are so far not defined. Then, you should list all the functions that need to be defined.

The output should be within <output> ... </output> tags, as shown in the example. You should first output the skeleton in <skeleton> tags. Then, for each function in the skeleton, output its header and docstring in <function> tags. Make sure the docstring for each subfunction contains all required information to solve it independently.  Also include test cases for each function in <testcase> tags.
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

        self.solve_prompt = """<instruction>You are a programming expert taking part in a programming competition. Given the input description, write a Python 3 function that solves the problem.

The output should be within <output> ... </output> tags, as shown in the example. Do not repeat the docstring in the output.
</instruction>

<example>

<prompt>
Vipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the world. Recently he indulged himself in saving the string population so much that he lost his ability for checking brackets (luckily, not permanently ).Being his super-hero friend help him in his time of hardship. Input The first line of the input contains an integer T denoting the number of test cases. The description of T test cases follows. The first line of each test case contains a single string S denoting the string to be checked. Output For each test case, output a single line printing "YES" or "NO" (without " " and in uppercase only) , denoting if the brackets in the given string is balanced or not . Constraints 1 ≤ T ≤ 10 1 ≤ length of S ≤ 60 Example Input: 3 ((())) (())() ()(() Output: YES YES NO   Explanation Example is self-explanatory.
</prompt>

<output>
for item in range(input()):
    try:
        eval(input())
        print 'YES'
    except TypeError:
        print 'YES'
    except:
        print 'NO'
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
  File "/home/pedrogimenes/find-closest-pair.py", line 13, in <module>
    assert celsius_to_fahrenheit(0) == 32
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pedrogimenes/find-closest-pair.py", line 11, in celsius_to_fahrenheit
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
        model = "",
        run_async = False,
        multiplicity: int = 1,
    ):

        outs = {
            node: llm(
                self.split_prompt.format(input=graph.nodes[int(node)]["problem"]),
                model=model,
                temperature=self.temperature,
            )[0] for node in nodes
        }

        for node in nodes:
            next_thought = outs[node]

            # remove the <output> tags
            next_thought = next_thought.replace("<output>", "")
            next_thought = next_thought.replace("</output>", "")

            # get the skeleton
            skeleton = re.search(r"<skeleton>(.*?)</skeleton>", next_thought, re.DOTALL).group(1)
            graph.nodes[int(node)]["solution"] = graph.nodes[int(node)]["problem"] + skeleton

            # get the content of each <function> tag
            functions = re.findall(r"<function>(.*?)</function>", next_thought, re.DOTALL)

            for function in functions:

                # get the docstring
                docstring = re.search(r"<docstring>(.*?)</docstring>", function, re.DOTALL).group(1)

                # get the testcases
                testcases = re.search(r"<testcase>(.*?)</testcase>", function, re.DOTALL).group(1)

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
        node,
        model = "",
        multiplicity: int = 1,
    ):
        return await asyncio.gather(
            *[
                async_llm(
                    self.solve_prompt.format(
                        input=graph.nodes[int(node)]["problem"]
                    ),
                    model=model,
                    temperature=self.temperature,
                ) for _ in range(multiplicity)
            ],
        )

    def generate(
        self,
        graph, 
        nodes,
        model = "",
        run_async = True,
        multiplicity: int = 1,
    ):
        nodes_to_score = []
        for node in nodes:
            
            # 1. Get LLM responses
            if run_async:
                outs = asyncio.run(
                    self.async_generate(
                        graph, 
                        node, 
                        model=model,
                        multiplicity=multiplicity,
                    )
                )
            else:
                outs = {
                    sol: llm(
                        self.solve_prompt.format(
                            input=graph.nodes[int(node)]["problem"]
                        ),
                        model=model,
                        temperature=self.temperature,
                    )[0] for sol in range(multiplicity)
                }

            for idx, out in enumerate(outs):
                next_thought = out[0]

                # remove the <ooututput> tags
                next_thought = next_thought.replace("<output>", "")
                next_thought = next_thought.replace("</output>", "")
                next_thought = next_thought.replace("```python", "")
                next_thought = next_thought.replace("```", "")

                # graph.nodes[int(node)]["solution"] = graph.nodes[int(node)]["problem"].replace("pass", "") + next_thought

                # # Reset feedback and score
                # graph.nodes[int(node)]["feedback"] = None
                # graph.nodes[int(node)]["score"] = None

                # generate new node
                idx = max(list(graph.nodes)) + 1
                kwargs = {
                    "problem": graph.nodes[int(node)]["problem"],
                    "solution": next_thought,
                    "score": None,
                    "feedback": None,
                }
                
                if node in ["0", 0]:
                    kwargs["is_solution"] = True
                
                graph.add_node(
                    idx,
                    **kwargs,
                )
                graph.add_edge(int(node), idx)

                nodes_to_score.append(idx)

            # for node in nodes_to_score:
            #     graph, _ = self.score(graph, [node], model=model)

        return graph, False

    async def async_refine(
        self,
        graph,
        nodes,
        model = "",
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
                ) for node in nodes
            ],
        )

    def refine(
        self,
        graph, 
        nodes,
        model = "",
        run_async = True,
        multiplicity: int = 1,
    ):
        # 1. Get LLM responses
        if run_async:
            outs = asyncio.run(self.async_refine(graph, nodes, model=model))
            outs = {
                nodes[i]: outs[i] for i in range(len(nodes))
            }
        else:
            outs = {
                node: llm(
                    self.refine_prompt.format(
                        candidate=graph.nodes[int(node)]["solution"],
                        feedback=graph.nodes[int(node)].get("feedback", ""),
                    ),
                    model=model,
                    temperature=self.temperature,
                )[0] for node in nodes
            }

        # 2. Update graph
        for node in nodes:
            sol = outs[node][0]

            # remove the <output> tags
            sol = sol.replace("<output>", "")
            sol = sol.replace("</output>", "")

            graph.nodes[int(node)]["solution"] = graph.nodes[int(node)]["problem"] + sol

            # Reset feedback and score
            graph.nodes[int(node)]["feedback"] = None
            graph.nodes[int(node)]["score"] = None

        # 3. Re-score all the refined nodes
        graph, _ = self.score(graph, nodes, model=model)

        return graph, False

    def _score_full_solution(
        self,
        graph,
        node,
    ):
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        # Get problem idx
        problem_idx = graph.nodes[0].get("problem_idx", None)
        if problem_idx is None:
            raise ValueError("Problem index not found in the node: {}".format(graph_node))

        ds_item = self.ds["train"][problem_idx]

        inputs = [inp for inp in ds_item["public_tests"]["input"]]
        inputs += [inp for inp in ds_item["private_tests"]["input"]]

        outputs = [out for out in ds_item["public_tests"]["output"]]
        outputs += [out for out in ds_item["private_tests"]["output"]]

        if not inputs:
            print(f"No testcases for problem {problem_idx}")
            graph_node["score"] = 0
            return graph, False

        # Evaluate the solution
        code = graph_node["solution"]

        outs = []
        successes = []
        failures = []
        for idx, user_input in enumerate(inputs):
            user_input = user_input.split("\n")

            # Setup timeout for 1 minute (60 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # Set alarm for 60 seconds

            output_capture = io.StringIO()
            # remove empty items ''
            with patch("builtins.input", side_effect=user_input):
                with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                    try:
                        exec(code)
                    
                    except TimeoutException:
                        print("Execution exceeded time limit of 1 minute.")

                    except Exception as exc:
                        print(exc)

                    finally: 
                        signal.alarm(0)
                
                captured_out = output_capture.getvalue().strip()
                outs.append(captured_out)

                if captured_out == outputs[idx].strip():
                    successes.append(idx)
                else:
                    failures.append(idx)

        if failures:
            feedback = f"Failed testcases: {failures}"
            score = 0
        elif successes:
            feedback = None
            score = 1
        else:
            raise ValueError("No testcases passed or failed")

        graph_node["score"] = score

        return graph, score

    def score(
        self,
        graph, 
        nodes,
        model = "",
        multiplicity: int = 1,
    ):

        any_pass = False
        for node in nodes:

            # If scoring the full problem, fall back to HumanEval code
            if node in ["0", 0] or graph.nodes[int(node)].get("is_solution", False):
                graph, score = self._score_full_solution(graph, node)
                
                graph.nodes[int(node)]["score"] = score
                if score > 0:
                    any_pass = True
                
                continue

            # Evaluate testcases
            testcases = graph.nodes[int(node)].get("testcases", None)
            if testcases is not None:
                program = (
                    "from typing import *\n"
                    + graph.nodes[int(node)].get("solution", "")
                    + testcases
                )

                try:
                    exec(program, {})
                    graph.nodes[int(node)]["score"] = 1
                except Exception as exc:
                    # get traceback
                    from traceback import format_exc
                    feedback = format_exc()
                    graph.nodes[int(node)]["score"] = 0
                    graph.nodes[int(node)]["feedback"] = feedback
                    continue
            else:
                print("No testcases found for node: ", node)
                graph.nodes[int(node)]["score"] = 0

        return graph, False

    def keepbest(
        self,
        graph, 
        nodes,
        model = "",
        multiplicity: int = 1,
    ):
        # Score all non-scored nodes
        # graph, _ = score(graph, nodes, model=model)
        return common_keepbest(graph, nodes, model)

    def aggregate(
        self,
        graph, 
        nodes,
        model = "",
        multiplicity: int = 1,
        run_async: bool = False,
    ):
        
        # 1. Collect code snippets and testcases
        code = ""
        testcases = ""
        for node in nodes:
            code += graph.nodes[int(node)]["solution"]
            testcases += graph.nodes[int(node)].get("testcases", "")
        
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
        model = "",
        run_async = True,
        multiplicity: int = 1,
    ):
        return self.generate(
            graph, 
            nodes, 
            model,
            run_async,
            multiplicity,
        )

    def cot(
        self,
        graph, 
        nodes,
        model = "",
    ):
        raise NotImplementedError("Cot not implemented for human eval")


    def count(
        self,
        graph, 
        nodes,
        model = "",
        run_async = True,
        multiplicity: int = 1,
    ):
        # Out of the selected nodes, count how many have a score of 1 and print the percentage

        count = 0
        for node in nodes:
            if graph.nodes[int(node)]["score"] == 1:
                count += 1

        rate = count / len(nodes)
        
        # Append to csv file
        problem_idx = graph.nodes[0].get("problem_idx", None)
        with open("code_contests_probabilities.csv", "a") as f:
            f.write(f"{problem_idx},{rate}\n")
        return graph, count > 0

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
                _, score = self._score_full_solution(graph, node)

            # Skip checking other nodes if a solution was found
            if score:
                graph.nodes[int(node)]["groundtruth"] = True
                return graph, True

            graph.nodes[int(node)]["groundtruth"] = False

        return graph, False