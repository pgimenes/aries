from tqdm import tqdm

from environment import GoTEnv
from agent import LLMAgent
import tasks.sorting as task

num_episodes = 1
problem = "[0, 2, 6, 3, 8, 7, 1, 1, 6, 7, 7, 7, 7, 9, 3, 0, 1, 7, 9, 1, 3, 5, 1, 3, 6, 4, 5, 4, 7, 3, 5, 7]"
    
if __name__ == "__main__":
    env = GoTEnv(
        problem=problem,
    )
    
    agent = LLMAgent(
        env=env,
        model = "gpt-4o",
        problem_definition = task.problem_definition,
        actions = task.actions,
    )

    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated