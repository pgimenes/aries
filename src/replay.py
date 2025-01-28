
import traceback
import os
from tqdm import tqdm
import argparse
import importlib
import pandas as pd
import json
import asyncio
import dill
import sys
from copy import deepcopy, copy

from environment import GoTEnv
from agent import LLMAgent, GoTAgent, IOAgent, CoTAgent, ToTAgent, get_agent
from cli import get_args

async def async_try_n_times(
    fn: callable,
    n: int,
):
    attempts = 1
    while True:
        try:
            res = await fn()
            break
        except:
            attempts += 1

            if attempts > n:
                print(f"Failed to run function {fn} after {n} attempts")
                return None
            else:
                continue

    return res

async def _run_agent_on_problem(
    agent, 
    environment,
    first_action=None,
):
    done = False
    last_score = None
    itr = 0    
    attempts = 1

    action_tree = []
    
    success = False
    last_score = 0

    while not done:
        if itr >= getattr(agent, "max_iterations", float('inf')) or attempts > 5:
            break

        # First action was hardcoded
        if itr == 0:
            action = first_action
        
        # Sample next action
        else:
            action, prompt, options = await agent.get_action()

        try:
            reward, terminated, truncated, info = await environment.async_step(action)

            # Only update the action history if it ran successfuly
            if isinstance(agent, LLMAgent):
                agent.action_history.append(action)

            if info.get("score", None) is not None:
                last_score = info["score"]

            success = terminated or truncated
            itr += 1
            attempts = 1

        except Exception as exc:
            print(f"[{attempts}/5] Action {action['operation']} failed on nodes {action['nodes']}, trying again. Error: {exc}")
            attempts += 1
            continue

    return agent, success, last_score

async def run_replay(args, data):

    # Fetch action trees
    print(f"Loading action trees...")
    with open(f"results/{args.task}_{args.agent}_action_trees.pkl", "rb") as f:
        action_trees = dill.load(f)
    print(f"Done")

    # Estimate how many preferences will be generated
    total_datapoints = 0
    total_replay_points = 0
    for idx, problem in enumerate(data):
        if idx < args.start or idx > args.end:
            continue

        action_tree = action_trees[idx]

        # print(f"Num levels: {len(action_tree)}")
        for level_idx, level in enumerate(action_tree):
            
            _, options, _ = level

            num_actions = len(options)
            # print(f"Level {level_idx}: {num_actions} options")

            total_datapoints += sum(range(num_actions))
            total_replay_points += num_actions

    print(f"Points to be replayed: {total_replay_points}")
    print(f"Total data points to be generated: {total_datapoints}")

    # Run replay
    # ------------------------------------------

    # Replay each problem
    for idx, problem in enumerate(data):
        if idx < args.start or idx > args.end:
            continue

        print(f"===============================")
        print(f"Replaying problem {idx}/{len(data)}")
        print(f"===============================\n")

        action_tree = action_trees[idx]
        problem = problem["prompt"]

        # Replay each level
        for level_idx, level in enumerate(action_tree):
            
            prompt, options, agent = level

            # Override agent parameters inherited from previous run
            setattr(agent, "cot_sc_branches", args.cot_sc_branches)
            setattr(agent, "max_iterations", args.max_iterations)
            agent.environment.reasoning_agent.temperature = args.temperature
            print(f"Setting temperature to {args.temperature}")

            print(f"Level {level_idx}/{len(action_tree)}")
            print(f"--------------------------------------\n")
            print(f"Options to replay: {options.keys()}\n")

            option_metrics = {}

            # Monte Carlo simulation through all the options at this level
            # and record the success rate of each
            for option_idx, item in enumerate(options.items()):

                action, info = item # info: {count, completion}

                print(f"Option {option_idx}/{len(options.keys())}: {action}")
                print(f"------------------------------------------------\n")
                
                # Run agent
                first_action = {
                    "operation": action[0],
                    "nodes": action[1],
                    "attempts": action[2],
                    "explanation": "",
                }

                # Launch using asyncio
                process_lst = []
                for replay_idx in range(args.replays_per_option):
                    # print(f"Replay {replay_idx}/{args.replays_per_option}")
                    process_agent = deepcopy(agent)
                    process_lst.append(
                        _run_agent_on_problem(
                            process_agent, 
                            process_agent.environment,
                            first_action=first_action,
                        )
                    )
                out = await asyncio.gather(*process_lst)

                success_count = 0 # out[0].action_history
                for replay_out in out:
                    if out is None:
                        continue

                    _, success, _ = replay_out
                    
                    if success:
                        success_count += 1

                # Done evaluating option, register success rate
                info["success_count"] = success_count
                print(f"Success count for problem {idx}, level {level_idx}, option {option_idx}: {success_count}")

    # pickle the replayed action trees
    with open(f"results/{args.task}_{args.agent}_replayed_action_trees.pkl", "wb") as f:
        dill.dump(action_trees, f)
                
# [(k["operation"], k["nodes"]) for k in out[0][0].action_history]
#agent.environment.thought_graph_repr()