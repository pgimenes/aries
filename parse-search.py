import os
import re
import sys, pdb, traceback
import matplotlib.pyplot as plt
import numpy as np
import ast

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = excepthook

tasks = [
    "sorting32",
    "sorting64",
    "sorting128",
    "set_intersection32",
    "set_intersection64",
    "set_intersection128",
    "keyword_counting",
]

for task in tasks:

    trials = []
    for i in range(100):
        fname = f'search/{task}/trial_'+str(i)+'-spec.log'

        if not os.path.exists(fname):
            continue

        with open(fname, 'r') as f:
            content = "\n".join(f.readlines())

            if not os.path.exists(fname):
                continue

            got_branches = int(re.findall(r'got_branches: (\d+)', content)[0])
            got_generate_attempts = int(re.findall(r'got_generate_attempts: (\d+)', content)[0])
            got_aggregate_attempts = int(re.findall(r'got_aggregate_attempts: (\d+)', content)[0])
            got_refine_attempts = int(re.findall(r'got_refine_attempts: (\d+)', content)[0])
            got_post_aggregate_keepbest = ast.literal_eval(re.findall(r'got_post_aggregate_keepbest: (True|False)', content)[0])
            got_post_aggregate_refine = ast.literal_eval(re.findall(r'got_post_aggregate_refine: (True|False)', content)[0])

            config = (
                got_branches,
                got_post_aggregate_keepbest,
                got_post_aggregate_refine,
                got_generate_attempts,
                got_aggregate_attempts,
                got_refine_attempts,
            )
            
            try:
                query_count = int(re.findall(r'query count: (\d+)+', content)[0])
                score = float(re.findall(r'score: (\d+\.\d+)', content)[0])
                if score >= 1000:
                    continue
                trials.append((query_count, score, config))
            except:
                continue

    # calculate alpha to balance query count and score
    avg_score = np.mean([t[1] for t in trials])
    avg_query_count = np.mean([t[0] for t in trials])
    alpha = min([avg_query_count / (avg_score + avg_query_count), 0.99])
    print(f"task: {task}")
    print(f"alpha: {alpha}")

    for idx, val in enumerate(trials):
        query_count, score, config = val
        cost = alpha * score + (1 - alpha) * query_count
        trials[idx] = (query_count, score, cost, config)

    # create min filter
    best_trials = []
    argmins = []
    for idx in range(len(trials)):
        costs = [t[2] for t in trials[:idx+1]]
        argmin = np.argmin(costs)
        # eliminate the config from 'best_trials'
        best_trials.append(trials[argmin][:-1])
        argmins.append(argmin)
    best_trials = np.array(best_trials)

    # find last effective trial (first index at which the cost == cost[-1])
    last_trial = len(best_trials) - 1
    idx = len(best_trials) - 1
    while idx >= 0:
        if best_trials[idx][2] == best_trials[-1][2]:
            last_trial = idx
        idx -= 1
    i25 = int(0.25 * last_trial)
    i50 = int(0.50 * last_trial)

    # plot in different plots vertically
    fig, axs = plt.subplots(3)
    fig.suptitle('Search Performance')
    axs[0].plot(best_trials[:, 0], label='query count')
    axs[0].legend()
    axs[1].plot(best_trials[:, 1], label='score')
    axs[1].legend()
    axs[2].plot(best_trials[:, 2], label='cost')
    axs[2].legend()

    # create line in each plot at i25, i50, i75
    for ax in axs:
        ax.axvline(x=i25, color='r', linestyle='--')
        ax.axvline(x=i50, color='r', linestyle='--')
        ax.axvline(x=last_trial, color='g', linestyle='--')

    plt.savefig(f'trial-plot-{task}.png')

    # create pareto plot
    fig, ax = plt.subplots()
    fig.suptitle('Search Performance')
    ax.scatter([t[0] for t in trials], [t[1] for t in trials], c=[t[2] for t in trials])
    ax.set_xlabel('query count')
    ax.set_ylabel('score')
    plt.savefig(f'pareto-plot-{task}.png')
        
    print(f"i25/i50/last_trial/num_trials: {i25}, {i50}, {last_trial}, {len(trials)}")
    
    for item in [(i25, "25"), (i50, "50"), (last_trial, "last")]:
        idx, name = item
        config = trials[idx][3]

        args = ""
        got_branches, got_post_aggregate_keepbest, got_post_aggregate_refine, got_generate_attempts, got_aggregate_attempts, got_refine_attempts = config
        args += f"""                                
                                --got_branches {got_branches} \\
                                --got_generate_attempts {got_generate_attempts} \\
                                --got_aggregate_attempts {got_aggregate_attempts} \\
                                --got_refine_attempts {got_refine_attempts} \\
"""
        if got_post_aggregate_keepbest:
            args += "                                --got_post_aggregate_keepbest \\\n"
        if got_post_aggregate_refine:
            args += "                                --got_post_aggregate_refine \\\n"

        print(f"got{name}: {best_trials[idx]}")
        print(args)