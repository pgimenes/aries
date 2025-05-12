"""
This is intended to run with openai human eval provided codex environment

"""

import json
from tqdm import tqdm
import io
import contextlib
from unittest.mock import patch

import datasets
import signal
import pandas as pd

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def _score_code_contests(
        problem_name,
        dataset_df,
        completion,
    ):

        # find the pandas index from the problem name
        problem_idx = dataset_df[dataset_df["name"] == problem_name].index[0]
        ds_item = dataset_df.iloc[problem_idx]

        inputs = [inp for inp in ds_item["public_tests"]["input"]]
        inputs += [inp for inp in ds_item["private_tests"]["input"]]

        outputs = [out for out in ds_item["public_tests"]["output"]]
        outputs += [out for out in ds_item["private_tests"]["output"]]

        if not inputs:
            print(f"No testcases for problem {problem_name}")
            return 0

        # Evaluate the solution
        code = completion

        outs = []
        successes = []
        failures = []

        for idx, user_input in enumerate(inputs):
            user_input = user_input.split("\n")
            
            # Execute
            with patch("builtins.input", side_effect=user_input):
                output_capture = io.StringIO()
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(20)  # Set the timeout to 20 seconds
                
                try:
                    with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                        try:
                            exec(code)
                        except Exception as exc:
                            print(exc)
                    
                    captured_out = output_capture.getvalue().strip()
                    outs.append(captured_out)

                    if captured_out == outputs[idx].strip():
                        successes.append(idx)
                    else:
                        failures.append(idx)
                except TimeoutException:
                    print(f"Timeout for problem {problem_name}")
                    return 0
                finally:
                    signal.alarm(0)  # Disable the alarm

        if failures:
            score = 0
        elif successes:
            score = 1
        else:
            raise ValueError("No testcases passed or failed")

        return score

# load the data
with open("codecontest_result.json", "r") as f:
    input_data = json.load(f)
    
output_data = []
combine_solution_idx = []
combine_solution_list = []  
accuracy_list = []

# we are only evaluating on the test dataset
code_contest_dataset = datasets.load_dataset("deepmind/code_contests")["test"]
code_contest_dataset_df = code_contest_dataset.to_pandas()

for i in tqdm(range(len(input_data))):
    completion = input_data[i]["completions"][0][0]
    # print(completion)
    # remove ```python and ``` from the completion if it exists
    completion = completion.replace("```python", "")
    completion = completion.replace("```", "")
    
    problem_name = input_data[i]["task_id"]
    # work out the problem index from the name from hf dataset
    # combine_solution_idx.append(problem_idx)
    # combine_solution_list.append(completion)
    output_accuracy = _score_code_contests(problem_name, code_contest_dataset_df, completion)
    accuracy_list.append(output_accuracy)
    
accuracy = sum(accuracy_list) / len(accuracy_list)

print(f"Code Contest Accuracy: {accuracy}")