import traceback
import os
from tqdm import tqdm
import argparse
import importlib
import pandas as pd
import json
import asyncio
import dill
from copy import deepcopy, copy

from environment import GoTEnv
from agent import LLMAgent, GoTAgent, IOAgent, CoTAgent, ToTAgent, get_agent
from cli import get_args
from replay import run_replay

ds_map = {
    "aqua": "deepmind/aqua_rat",
    "math": "hendrycks/competition_math",
    "aime": "qq8933/AIME_1983_2024",
    "mmlu": "cais/mmlu",
    "human_eval": "openai/openai_humaneval",
}

def get_problem(problem, args):
    
    if "sorting" in args.task:
        problem = problem[1]["Unsorted"]
    elif args.task == "game24":
        problem = problem[1]["Puzzles"]
    elif args.task == "crosswords":
        problem, solution = problem
    elif args.task == "keyword_counting":
        problem = problem[1]["Text"]
    elif "set_intersection" in args.task:
        problem = {
            "set1": problem[1]["SET1"],
            "set2": problem[1]["SET2"],
        }
    elif args.task == "human_eval":
        problem = problem["prompt"]
    else:
        raise Exception("Invalid task")

    return problem

def _run_agent_on_problem(
    agent, 
    environment,
    next_action=None,
    idx = None,
    dump_action_tree = True,
):
    done = False
    last_score = None
    itr = 0    
    attempts = 1

    action_tree = []
    
    while not done:
        if itr >= getattr(agent, "max_iterations", float('inf')) or attempts > 5:
            break

        # First action was hardcoded
        if next_action is not None:
            action = next_action
        
        # Sample next action
        else:

            if isinstance(agent, LLMAgent):
                action, prompt, options = asyncio.run(agent.get_action())
            else:
                # get state from environment
                # state = ...
                breakpoint()
                action = agent.get_action()

        try:
            
            reward, terminated, truncated, info = environment.step(action)

            # If just executed the hardcoded action, start sampling in the next round
            next_action = None

        except Exception as exc:
            # LLMAgent can try to recover by fetching another action...
            if isinstance(agent, LLMAgent):
                print(f"[{attempts}/5] Action {action['operation']} failed on nodes {action['nodes']}, trying again. Error: {exc}")
                attempts += 1
                continue

            # Other agents need to give up
            else:
                break
        
        # Only update the action history if it ran successfuly
        if isinstance(agent, LLMAgent):
            agent.action_history.append(action)

        if info.get("score", None) is not None:
            last_score = info["score"]

        success = terminated or truncated
        itr += 1
        attempts = 1

        # Add to action tree, attaching a copy of the agent
        # so the graph state can be reconstructed
        if dump_action_tree:
            tree_level = (prompt, options, deepcopy(agent)) # agent.environment.thought_graph.nodes
            action_tree.append(tree_level)

    return success, last_score, action_tree

def run(args, data):
    # for got datasets
    # iterator = data.iterrows()
    iterator = data

    # Get task name
    taskname = "sorting" if "sorting" in args.task else args.task
    taskname = "set_intersection" if "set_intersection" in taskname else taskname
    task = importlib.import_module(f"tasks.{taskname}")

    # Get agent parameters
    agent_config = {
        # General
        "max_iterations": args.max_iterations,

        # ToT parameters
        "tot_width": args.tot_width,
        "tot_depth": args.tot_depth,

        # GoT parameters
        "got_branches": args.got_branches,
        "got_generate_attempts": args.got_generate_attempts,
        "got_aggregate_attempts": args.got_aggregate_attempts,
        "got_post_aggregate_keepbest": args.got_post_aggregate_keepbest,
        "got_post_aggregate_refine": args.got_post_aggregate_refine,
        "got_refine_attempts": args.got_refine_attempts,
    }

    if args.task == "keyword_counting":
        agent_config["got_branches"] = len(problem.split(".")) - 1

    # Track if each problem was solved
    successes = []
    failures = []
    scores = []
    action_trees = []

    for idx, problem in enumerate(iterator):
        if idx < args.start or idx > args.end:
            continue

        print(f"===============================")
        print(f"Solving problem {idx}/{len(data)}")
        print(f"===============================")

        # DPO data extraction
        # action_tree[i] holds the nodes at depth i of the tree
        action_tree = []

        problem = get_problem(problem, args)

        # Build environment
        environment = GoTEnv(
            problem=problem,
            task= taskname,
            model=args.model,
            idx = idx,
        )
        state, _ = environment.reset()
        
        # Build agent
        agent = get_agent(args.agent, environment, task, args, **agent_config)

        # Run agent
        try:
            success, score, action_tree = _run_agent_on_problem(agent, environment)
        except Exception as exception:
            stack = traceback.format_exc()
            print(f"Could not complete problem {idx}: {exception}")
            print(stack)
            failures.append(problem)
            action_trees.append(None)
            continue

        # Update results
        action_trees.append(action_tree)
        
        if success:
            print(f"Result: success")
            successes.append(problem)
        else:
            print(f"Result: failure")
            failures.append(problem)
        
        if score is not None:
            scores.append(score)
                

    # Print summary
    print(f"===============================")
    print(f"Summary")
    print(f"===============================")

    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    if len (scores) > 0:
        avg_score = sum(scores) / len(scores)
        print(f"Average score: {avg_score}")
    else:
        print(f"Scores for each problem not available")

    # Save results
    # mkdir results if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    with open(f"results/{args.task}_{args.agent}_action_trees.pkl", "wb") as f:
        dill.dump(action_trees, f)

if __name__ == "__main__":

    args = get_args()
    
    if args.task in [
        "human_eval",
    ]:
        from datasets import load_dataset
        ds_name = ds_map[args.task]
        data = load_dataset(ds_name, split="test")
    elif args.task == "crosswords":
        with open("data/crosswords.json") as f:
            data = json.load(f)
    else:
        with open(f"data/{args.task}.csv") as f:
            data = pd.read_csv(f)

    if args.replay:
        asyncio.run(run_replay(args, data))
    else:
        run(args, data)
