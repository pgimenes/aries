from tqdm import tqdm
import argparse
from environment import GoTEnv
from agent import LLMAgent, GoTAgent, IOAgent, CoTAgent, ToTAgent
import importlib
import pandas as pd
import json

import sys, pdb, traceback

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = excepthook

num_episodes = 1

def get_agent(agent, env, task, **kwargs):
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
            model = "gpt-4",
            problem_definition = task.problem_definition,
            actions = task.actions,
            width = kwargs.get("tot_width"),
            depth = kwargs.get("tot_depth"),
        )
    
    if agent == "got":
        return GoTAgent(
            env=env,
            task = task,
            model = "gpt-4",
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
            model = "gpt-4",
            problem_definition = task.problem_definition,
            actions = task.actions,
        )

def run(args, data):
    successes = []
    failures = []

    if args.task != "crosswords":
        iterator = data.iterrows()
    else:
        iterator = data

    for idx, problem in enumerate(iterator):
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
        else:
            raise Exception("Invalid task")

        # Build environment
        taskname = "sorting" if "sorting" in args.task else args.task
        taskname = "set_intersection" if "set_intersection" in taskname else taskname
        env = GoTEnv(
            problem=problem,
            task= taskname,
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

        # Number of branches depends on number of sentences in the problem
        if "sorting" in args.task:
            kwargs["got_post_aggregate_keepbest"] = True
            kwargs["got_post_aggregate_refine"] = True
        if "set_intersection" in args.task:
            kwargs["got_post_aggregate_keepbest"] = True
            kwargs["got_post_aggregate_refine"] = False
        if args.task == "keyword_counting":
            kwargs["got_branches"] = len(problem.split(".")) - 1
            kwargs["got_post_aggregate_keepbest"] = False
            kwargs["got_post_aggregate_refine"] = True

        agent = get_agent(args.agent, env, task, **kwargs)

        # Run agent on environment
        done = False
        itr = 0    
        max_iterations = agent.max_iterations
        while not done:
            if itr >= max_iterations:
                break

            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            itr += 1

        if done:
            print(f"Result: success")
            successes.append(problem)
        else:
            print(f"Result: failure")
            failures.append(problem)
                
        # except Exception as e:
        #     # traceback.print_stack()
        #     print(f"Error: {e}")
        #     failures.append(problem)

    # summary
    print(f"===============================")
    print(f"Summary")
    print(f"===============================")

    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    breakpoint()

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
        "--max_iterations", 
        type=int, 
        default=100,
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
        type=bool, 
        default=True,
    )
    parser.add_argument(
        "--got_post_aggregate_refine", 
        type=bool, 
        default=True,
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
    
    if args.task == "crosswords":
        with open("data/crosswords.json") as f:
            data = json.load(f)
    else:
        with open(f"data/{args.task}.csv") as f:
            data = pd.read_csv(f)

    data = data[:args.max_iterations]

    run(args, data)
