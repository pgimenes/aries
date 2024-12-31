from tqdm import tqdm
import argparse
from environment import GoTEnv
from agent import LLMAgent, GoTAgent, IOAgent, CoTAgent, ToTAgent
import importlib
import pandas as pd
import json
import asyncio

import sys, pdb, traceback

ds_map = {
    "aqua": "deepmind/aqua_rat",
    "math": "hendrycks/competition_math",
    "aime": "qq8933/AIME_1983_2024",
    "mmlu": "cais/mmlu",
}

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
# sys.excepthook = excepthook

num_episodes = 1

def get_agent(agent, env, task, args, **kwargs):
    if agent == "io":
        return IOAgent(
            env=env,
            task = task,
        )
    
    if agent == "cot":
        return CoTAgent(
            env=env,
            task=task,
        )
    
    if agent == "tot":
        return ToTAgent(
            env = env,
            task = task,
            model = args.model,
            problem_definition = task.problem_definition,
            actions = task.actions,
            width = kwargs.get("tot_width"),
            depth = kwargs.get("tot_depth"),
        )
    
    if agent == "got":
        return GoTAgent(
            env=env,
            task = task,
            model = args.model,
            problem_definition = task.problem_definition,
            actions = task.actions,
            branches=kwargs.get("got_branches"),
            generate_attempts=kwargs.get("got_generate_attempts"),
            aggregate_attempts=kwargs.get("got_aggregate_attempts"),
            post_aggregate_keepbest=kwargs.get("got_post_aggregate_keepbest"),
            post_aggregate_refine=kwargs.get("got_post_aggregate_refine"),
            refine_attempts=kwargs.get("got_refine_attempts"), 
        )
    
    if agent == "llm":
        return LLMAgent(
            env=env,
            task = task,
            model = args.model,
            problem_definition = task.problem_definition,
            actions = task.actions,
            max_iterations = kwargs.get("max_iterations", 25),
            cot_sc_branches=args.cot_sc_branches,
        )

def run(args, data):
    successes = []
    failures = []
    scores = []

    # for got datasets
    # iterator = data.iterrows()
    iterator = data

    for idx, problem in enumerate(iterator):
        if idx < args.start or idx > args.end:
            continue

        print(f"===============================")
        print(f"Solving problem {idx}/{len(data)}")
        print(f"===============================")

        # try:
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
        elif args.task in [
            "aqua",
            "math",
            "aime",
            "mmlu",
        ]:
            problem = problem["Question"]
        else:
            raise Exception("Invalid task")

        # Build environment
        taskname = "sorting" if "sorting" in args.task else args.task
        taskname = "set_intersection" if "set_intersection" in taskname else taskname
        env = GoTEnv(
            problem=problem,
            task= taskname,
            model=args.model,
        )
        obs, _ = env.reset()
        
        # Build agent
        task = importlib.import_module(f"tasks.{taskname}")
        
        kwargs = {
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
            kwargs["got_branches"] = len(problem.split(".")) - 1

        agent = get_agent(args.agent, env, task, args, **kwargs)

        # Run agent on environment
        done = False
        last_score = None
        itr = 0    
        attempts = 1
        
        while not done:
            if itr >= agent.max_iterations or attempts > 5:
                break

            if isinstance(agent, LLMAgent):
                action = asyncio.run(agent.get_action(obs))
            else:
                action = agent.get_action(obs)

            # try:
            obs, reward, terminated, truncated, info = env.step(action)

            # Only update the action history if it ran successfuly
            if isinstance(agent, LLMAgent):
                agent.action_history.append(action)

            if info.get("score", None) is not None:
                last_score = info["score"]

            done = terminated or truncated
            itr += 1
            attempts = 1
            # except Exception as exc:
            #     # LLMAgent can try to recover by fetching another action...
            #     if isinstance(agent, LLMAgent):
            #         print(f"[{attempts}/5] Action {action['operation']} failed on nodes {action['nodes']}, trying again. Error: {exc}")
            #         attempts += 1

            #     # Other agents need to give up
            #     else:
            #         break

        if done:
            print(f"Result: success")
            successes.append(problem)
        else:
            print(f"Result: failure")
            failures.append(problem)
        
        if last_score is not None:
            scores.append(last_score)
                
        # except Exception as e:
        #     # traceback.print_stack()
        #     print(f"Error: {e}")
        #     failures.append(problem)

    print(f"===============================")
    print(f"Summary")
    print(f"===============================")

    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 1000
    print(f"Average score: {avg_score}")

def argparser():
    # Read args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", 
        type=str, 
        default="tot"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="sorting32"
    )
    parser.add_argument(
        "--start", 
        type=int, 
        default=0,
    )
    parser.add_argument(
        "--end", 
        type=int, 
        default=100,
    )

    # LLM Agent
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--cot_sc_branches",
        type=int,
        default=1,
    )

    # GoT Agent
    parser.add_argument(
        "--got_branches", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--got_generate_attempts", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--got_aggregate_attempts", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--got_post_aggregate_keepbest", 
        action="store_true",
    )
    parser.add_argument(
        "--got_post_aggregate_refine", 
        action="store_true",
    )
    parser.add_argument(
        "--got_refine_attempts", 
        type=int, 
        default=2,
    )

    # ToT Agent
    parser.add_argument(
        "--tot_width", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--tot_depth", 
        type=int, 
        default=2,
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = argparser()
    
    if args.task in [
        "aqua",
        "math",
        "aime",
        "mmlu",
    ]:
        from datasets import load_dataset
        ds_name = ds_map[args.task]
        data = load_dataset(ds_name, split="train")
    elif args.task == "crosswords":
        with open("data/crosswords.json") as f:
            data = json.load(f)
    else:
        with open(f"data/{args.task}.csv") as f:
            data = pd.read_csv(f)

    run(args, data)
