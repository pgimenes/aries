import optuna
import subprocess
import os
import re
import argparse

from tasks.common import _common_got_schedule

def count_queries(actions, action_nodes):
    query_count = 0
    for idx, action in enumerate(actions):
        if action in ["score", "keepbest", "groundtruth"]:
            continue
        elif action in ["sort", "refine", "split"]:
            query_count += len(action_nodes[idx])
        elif action in ["aggregate"]:
            query_count += len(action_nodes[idx]) // 2
        else:
            raise ValueError(f"Unknown action: {action}")
    
    return query_count

# Define the objective function that will be used in the Optuna study
def objective(trial):
    directory = f'search/{task}'
    os.makedirs(directory, exist_ok=True)
    score_file_path = f'{directory}/trial_{trial.number}'

    # Sample the parameters
    got_branches = branches
    got_post_aggregate_keepbest = trial.suggest_categorical('got_post_aggregate_keepbest', [True, False])  # True or False
    got_post_aggregate_refine = trial.suggest_categorical('got_post_aggregate_refine', [True, False])  # True or False
    
    got_generate_attempts = trial.suggest_categorical('got_generate_attempts', [1, 5, 10, 15, 20])
    got_aggregate_attempts = trial.suggest_categorical('got_aggregate_attempts', [1, 5, 10, 15, 20])
    got_refine_attempts = trial.suggest_categorical('got_refine_attempts', [1, 5, 10, 15, 20])

    if not (got_post_aggregate_keepbest or got_post_aggregate_refine):
        raise optuna.TrialPruned("At least one of post_aggregate_keepbest or post_aggregate_refine must be True.")

    # Prepare the command with the selected parameters
    command = [
        'python', '-u', 'src/main.py',
        '--task', task,
        '--agent', 'got',
        '--start', '1',
        '--end', '10',
        '--got_branches', str(got_branches),
        '--got_generate_attempts', str(got_generate_attempts),
        '--got_aggregate_attempts', str(got_aggregate_attempts),
        '--got_refine_attempts', str(got_refine_attempts),
    ]
    if got_post_aggregate_keepbest:
        command.append('--got_post_aggregate_keepbest')
    if got_post_aggregate_refine:
        command.append('--got_post_aggregate_refine')

    # Count the queries required according to the parameters
    actions, action_nodes = _common_got_schedule(
        branches = got_branches,
        generate_action = "sort",
        generate_attempts = got_generate_attempts,
        aggregate_attempts = got_aggregate_attempts,
        post_aggregate_keepbest = got_post_aggregate_keepbest,
        post_aggregate_refine = got_post_aggregate_refine,
        refine_attempts = got_refine_attempts,
    )
    query_count = count_queries(actions, action_nodes)

    # Dump the parameters
    with open(f"{score_file_path}-spec.log", 'w') as log_file:
        log_file.write(f"========== Trial {trial.number} ==========\n")
        log_file.write(f"alpha: {alpha}\n")
        log_file.write(f"got_branches: {got_branches}\n")
        log_file.write(f"got_post_aggregate_keepbest: {got_post_aggregate_keepbest}\n")
        log_file.write(f"got_post_aggregate_refine: {got_post_aggregate_refine}\n")
        log_file.write(f"got_generate_attempts: {got_generate_attempts}\n")
        log_file.write(f"got_aggregate_attempts: {got_aggregate_attempts}\n")
        log_file.write(f"got_refine_attempts: {got_refine_attempts}\n")
        log_file.write(f"query count: {query_count}\n")

    # Run the command and get the score
    print(f"Running trial {trial.number} with query count: {query_count}")
    with open(f"{score_file_path}.log", 'w') as log_file:
        _ = subprocess.run(command, stdout=log_file, stderr=log_file, text=True)

    with open(f"{score_file_path}.log", 'r') as score_file:
        content = score_file.read()
        score = float(re.search(r'Average score: (\d+\.\d+)', content).group(1))

    # Calculate the cost
    cost = alpha * score + (1 - alpha) * query_count

    # Dump results summary
    with open(f"{score_file_path}-spec.log", 'a') as log_file:
        log_file.write(f"\n========== Results ==========\n")
        log_file.write(f"alpha: {alpha}\n")
        log_file.write(f"query count: {query_count}\n")
        log_file.write(f"score: {score}\n")
        log_file.write(f"cost: {cost}\n")

    return cost

# Create an Optuna study to optimize the objective function
if __name__ == "__main__":

    # read task from command line --task
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", 
        type=str, 
        default="sorting32"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    global task
    global branches
    global alpha

    task = args.task
    alpha = args.alpha

    if task in ["sorting32", "set_intersection32"]:
        branches = 2
    elif task in ["sorting64", "set_intersection64"]:
        branches = 4
    elif task in ["sorting128", "set_intersection128"]:
        branches = 8
    elif task == "keyword_counting":
        branches = 16
    else:
        raise ValueError(f"Unknown task: {task}")
    
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
    )
    study.optimize(objective, n_trials=100)  # Number of trials to run in the random search

    # Print the best trial and its score
    best_trial = study.best_trial

    print(f"Best trial: {best_trial.number}")
    print(f"Best parameters: {best_trial.params}")
    print(f"Best score: {best_trial.value}")
