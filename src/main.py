from tqdm import tqdm

from environment import GoTEnv
from agent import LLMAgent
import tasks.sorting as task
import pandas as pd

num_episodes = 1

if __name__ == "__main__":
    with open("data/sorting32.csv") as f:
        data = pd.read_csv(f)

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
            
            agent = LLMAgent(
                env=env,
                model = "gpt-4",
                problem_definition = task.problem_definition,
                actions = task.actions,
            )

            obs, info = env.reset()

            done = False
            itr = 0
            max_iterations = 100
            while not done:
            
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if itr > max_iterations:
                    break

            if done:
                print(f"Result: success")
                successes.append(problem)
            else:
                print(f"Result: failure")
                failures.append(problem)
                
        except Exception as e:
            print(f"Error: {e}")
            failures.append(problem)

    # summary
    print(f"===============================")
    print(f"Summary")
    print(f"===============================")

    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    breakpoint()