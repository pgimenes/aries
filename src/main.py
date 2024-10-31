from tqdm import tqdm
import argparse
from environment import GoTEnv
from agent import LLMAgent, GoTAgent, IOAgent, CoTAgent, ToTAgent
import tasks.sorting as task
import pandas as pd
import traceback

num_episodes = 1

def get_agent(method, env, task):
    if method == "io":
        return IOAgent(
            env=env,
        )
    
    if method == "cot":
        return CoTAgent(
            env=env,
            task=task,
        )
    
    if method == "tot":
        return ToTAgent(
            env = env,
            model = "gpt-4",
            problem_definition = task.problem_definition,
            actions = task.actions,
            width = 10,
            depth = 3,
        )
    
    if method == "got":
        return GoTAgent(
            env=env,
            model = "gpt-4",
            problem_definition = task.problem_definition,
            actions = task.actions,
        )
    
    if method == "llm":
        return LLMAgent(
            env=env,
            model = "gpt-4",
            problem_definition = task.problem_definition,
            actions = task.actions,
        )
    
def itr_limit(method):
    if method == "io":
        return 2
    if method == "cot":
        return 100
    if method == "tot":
        return 11
    if method == "got":
        return 9
    if method == "llm":
        return 100

def run(args, data):
    successes = []
    failures = []

    for idx, problem in enumerate(data.iterrows()):
        print(f"===============================")
        print(f"Solving problem {idx}/{len(data)}")
        print(f"===============================")

        try:
            problem = problem[1]["Unsorted"]

            env = GoTEnv(
                problem=problem,
            )
            
            agent = get_agent(args.method, env, task)

            obs, info = env.reset()

            done = False
            itr = 0    
            max_iterations = itr_limit(args.method)
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
                
        except Exception as e:
            # traceback.print_stack()
            print(f"Error: {e}")
            failures.append(problem)

    # summary
    print(f"===============================")
    print(f"Summary")
    print(f"===============================")

    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    breakpoint()

if __name__ == "__main__":
    with open("data/sorting32.csv") as f:
        data = pd.read_csv(f)

    # Read args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", 
        type=str, 
        default="llm"
    )

    args = parser.parse_args()

    run(args, data)
