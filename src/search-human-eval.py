import optuna
import subprocess
import os
import re
import argparse
import sys, pdb, traceback
import random

# Define the objective function that will be used in the Optuna study
def objective(trial):
    directory = f'search/{task}'
    os.makedirs(directory, exist_ok=True)
    score_file_path = f'{directory}/trial_{trial.number}'

    # make the directory if it doesn't exist
    if not os.path.exists(score_file_path):
        os.makedirs(score_file_path)

    search_space = {
        "got_decompose_attempts": [1, 5, 10],
        "got_generate_attempts": [1, 5, 10],
        "got_aggregate_attempts": [1, 5, 10],
        "got_refine_attempts": [1, 5, 10],
    }

    # Sample the parameters
    got_decompose_attempts = trial.suggest_categorical("got_decompose_attempts", search_space["got_decompose_attempts"])
    got_generate_attempts = trial.suggest_categorical("got_generate_attempts", search_space["got_generate_attempts"])
    got_aggregate_attempts = trial.suggest_categorical("got_aggregate_attempts", search_space["got_aggregate_attempts"])
    got_refine_attempts = trial.suggest_categorical("got_refine_attempts", search_space["got_refine_attempts"])

    # Dump the parameters
    print(f"Running trial {trial.number}")
    with open(f"{score_file_path}/spec.log", 'w') as log_file:
        log_file.write(f"========== Trial {trial.number} ==========\n")
        log_file.write(f"got_decompose_attempts: {got_decompose_attempts}\n")
        log_file.write(f"got_generate_attempts: {got_generate_attempts}\n")
        log_file.write(f"got_aggregate_attempts: {got_aggregate_attempts}\n")
        log_file.write(f"got_refine_attempts: {got_refine_attempts}\n")

    scores, queries = [], []
    for idx in range(problems_per_trial):

        # get random num
        problem_idx = random.randint(0, 100)

        # Prepare the command with the selected parameters
        command = [
            'python', '-u', 'src/main.py',
            '--task', task,
            '--agent', 'got',
            '--start', str(problem_idx),
            '--end', str(problem_idx),
            '--got_decompose_attempts', str(got_decompose_attempts),
            '--got_generate_attempts', str(got_generate_attempts),
            '--got_aggregate_attempts', str(got_aggregate_attempts),
            '--got_refine_attempts', str(got_refine_attempts),
        ]
        
        print(f"[{idx}/{problems_per_trial}] command: {' '.join(command)}")

        # Run the command and get the score
        with open(f"{score_file_path}/idx-{idx}.log", 'w') as log_file:
            _ = subprocess.run(command, stdout=log_file, stderr=log_file, text=True)

        # Get the score and queries for this trial
        try:
            with open(f"{score_file_path}/idx-{idx}.log", 'r') as score_file:
                content = score_file.read()
                score = float(re.search(r'Average score: (\d+\.\d+)', content).group(1))
                query_count = int(re.search(r'Average queries: (\d+)', content).group(1))
                
                # Append the results
                scores.append(score)
                queries.append(query_count)
        except:
            pass
        
    print(f"Trial {trial.number} has following scores: {scores}")
    print(f"Trial {trial.number} has following query counts: {queries}")

    # calculate averages
    if len(scores) > 0:
        score = sum(scores) / len(scores)
        query_count = sum(queries) / len(queries)
    else:
        score = 100
        query_count = 1000

    # Dump results summary
    with open(f"{score_file_path}/spec.log", 'a') as log_file:
        log_file.write(f"\n========== Results ==========\n")
        log_file.write(f"query count: {query_count}\n")
        log_file.write(f"score: {score}\n")

    return score, query_count

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
    parser.add_argument(
        "--problems_per_trial",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    global task
    global alpha
    global problems_per_trial
    problems_per_trial = args.problems_per_trial

    task = args.task
    alpha = args.alpha
    
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        directions=['minimize', 'minimize'],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=100)  # Number of trials to run in the random search

    # Print the best trial and its score
    trials = study.best_trials
    breakpoint()

    # print(f"Best trial: {best_trial.number}")
    # print(f"Best parameters: {best_trial.params}")
    # print(f"Best score: {best_trial.value}")
