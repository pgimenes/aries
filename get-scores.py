import re

def parse_log_and_average_score(log):
    # Split the log into problems
    problems = log.split("=" * 31)
    total_score = 0
    problem_count = 0
    scores = []

    # Iterate through each problem
    for problem in problems:
        if problem.strip() == "":
            continue

        if "Solving problem" in problem:
            last_problem = re.search(r"Solving problem (\d+)/", problem)
            if not last_problem:
                continue
            last_problem = int(last_problem.group(1))
            continue

        # Find all steps in the problem
        steps = problem.split("Step")
        groundtruth_scores = []

        for step in steps:
            # Check if the step contains a groundtruth action
            action_match = re.search(r"Action:\s*(\w+)", step)
            if action_match:
                action = action_match.group(1)
                if action == "groundtruth":  # Replace 'groundtruth' with the desired action keyword
                    # Extract the node involved in the groundtruth action
                    node_match = re.search(r"Nodes:\s*\[([0-9, ]+)\]", step)
                    if node_match:
                        groundtruth_node = node_match.group(1).split(",")[0].strip()

                        # Extract the graph state of the node to get its score
                        graph_state_match = re.search(
                            rf"{groundtruth_node}:\s*\{{.*'score':\s*([^,}}]+).*}}",
                            step,
                        )
                        if graph_state_match:
                            score = graph_state_match.group(1)
                            try:
                                score = float(score)  # Convert score to float
                                groundtruth_scores.append(score)
                            except ValueError:
                                continue

        # Take the lowest score for this problem, if any groundtruth scores were found
        if groundtruth_scores:
            min_score = min(groundtruth_scores)
            total_score += min_score
            problem_count += 1
            scores.append(min_score)

    # Calculate and return the average score
    average_score = total_score / problem_count if problem_count > 0 else None
    return average_score, scores


# Example usage
for task in [
    "sorting32",
    "sorting64",
    "sorting128",
    "set-intersection32",
    "set-intersection64",
    "set-intersection128",
]:
    print(f"Task: {task}")

    with open(f"experiments/logs/llama-3.1-405b/{task}-llm.log", "r") as f:
        log_text = f.read()

    _, scores = parse_log_and_average_score(log_text)
    scs = [sc for sc in scores if sc < 1000]

    if len(scs) == 0:
        print("No scores found")
        exit()

    average_score = sum(scs) / len(scs)
    print(f"Scores found: {len(scs)}")
    print(f"Average Score: {average_score}")
