def _score_human_eval(
    problem_idx,
    completion,
):
    from human_eval.data import write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    sample = [
        {
            "task_id": f"HumanEval/{problem_idx}",
            "completion": completion,
        }
    ]
    write_jsonl("sample.jsonl", sample)

    out = evaluate_functional_correctness("sample.jsonl", ignore_incomplete=True)
    
    score = 1 if out["pass@1"] == 1.0 else 0
    
    return score