import os
import re
import sys, pdb, traceback
import matplotlib.pyplot as plt
import numpy as np

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = excepthook

task = 'sorting128'
alpha = 0.99

trials = []
for i in range(100):
    fname = f'search/{task}/trial_'+str(i)+'-spec.log'

    # check if exists
    if not os.path.exists(fname):
        continue

    with open(fname, 'r') as f:
        content = "\n".join(f.readlines())

        if not os.path.exists(fname):
            continue

        # parse query count: <NUM>
        try:
            query_count = int(re.findall(r'query count: (\d+)+', content)[0])
            score = float(re.findall(r'score: (\d+\.\d+)', content)[0])
            # cost = float(re.findall(r'cost: (\d+\.\d+)', content)[0])
            cost = alpha * score + (1 - alpha) * query_count
            trials.append((query_count, score, cost))
        except:
            continue

# trials = np.array(trials)

# create min filter
best_trials = []
for idx in range(len(trials)):
    costs = [t[2] for t in trials[:idx+1]]
    # min_costs.append(min(costs))
    argmin = np.argmin(costs)
    best_trials.append(trials[argmin])
best_trials = np.array(best_trials)

# plot in different plots vertically
fig, axs = plt.subplots(3)
fig.suptitle('Search Performance')
axs[0].plot(best_trials[:, 0], label='query count')
axs[0].legend()
axs[1].plot(best_trials[:, 1], label='score')
axs[1].legend()
axs[2].plot(best_trials[:, 2], label='cost')
axs[2].legend()
plt.savefig('trial_plot.png')

# create pareto plot
# fig, ax = plt.subplots()
# fig.suptitle('Search Performance')
# ax.scatter([t[0] for t in trials], [t[1] for t in trials], c=[t[2] for t in trials])
# ax.set_xlabel('query count')
# ax.set_ylabel('score')
# plt.savefig('trial_plot.png')