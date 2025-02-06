import json
from tqdm import tqdm

from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


def _score_human_eval(
    problem_idx_list,
    completion_list,
):
    sample = []
    for i in range(len(problem_idx_list)):
        problem_idx = problem_idx_list[i]
        completion = completion_list[i]
        sample.append(
            {
                "task_id": f"HumanEval/{problem_idx}",
                "completion": completion,
            }
        )
    
    write_jsonl("sample.jsonl", sample)

    out = evaluate_functional_correctness("sample.jsonl")
    
    score = out["pass@1"]
    
    return score

# load the data
with open("humaneval_results.json", "r") as f:
    input_data = json.load(f)
    
output_data = []
combine_solution_idx = []
combine_solution_list = []  

for i in tqdm(range(len(input_data))):
    completion = input_data[i]["completions"][0][0]
    
    # remove ```python and ``` from the completion if it exists
    completion = completion.replace("```python", "")
    completion = completion.replace("```", "")
    
    problem_idx = input_data[i]["task_id"].split("/")[-1]
    combine_solution_idx.append(problem_idx)
    combine_solution_list.append(completion)
    
output_accuracy = _score_human_eval(combine_solution_idx, combine_solution_list)

print(f"Human Eval Accuracy: {output_accuracy}")