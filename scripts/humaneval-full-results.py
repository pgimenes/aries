import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing trial_x folders
base_dir = "search/human_eval"

trials = []
scores = []
queries = []

# Read all trials
for trial_dir in sorted(os.listdir(base_dir)):
    if trial_dir.startswith("trial_"):
        trial_path = os.path.join(base_dir, trial_dir, "spec.log")
        if os.path.exists(trial_path):
            with open(trial_path, "r") as f:
                content = f.read()
                
                # Extract values
                query_line = [line for line in content.splitlines() if "query count" in line][0]
                score_line = [line for line in content.splitlines() if "score:" in line][0]
                
                query_value = float(query_line.split(":")[-1].strip())
                score_value = float(score_line.split(":")[-1].strip())
                
                trials.append(trial_dir)
                queries.append(query_value)
                scores.append(score_value)

# Convert to numpy arrays
queries = np.array(queries)
scores = np.array(scores)

# Compute cost function
avg_score = np.mean(scores)
avg_query = np.mean(queries)
alpha = avg_query / (avg_query + avg_score)
costs = alpha * queries + (1 - alpha) * scores

# Compute running minimum cost
best_costs = np.minimum.accumulate(costs)

# Identify GoT100
final_best_cost = best_costs[-1]
GoT100 = np.where(best_costs == final_best_cost)[0][0] + 1  # Convert to 1-based index
GoT25 = int(GoT100 * 0.25)
GoT50 = int(GoT100 * 0.50)

got25_trial = np.where(best_costs == best_costs[GoT25 - 1])[0][0] + 1
got50_trial = np.where(best_costs == best_costs[GoT50 - 1])[0][0] + 1
got100_trial = np.where(best_costs == best_costs[GoT100 - 1])[0][0] + 1

print(f"GoT25: {got25_trial} (Trial {trials[got25_trial - 1]})")
print(f"GoT50: {got50_trial} (Trial {trials[got50_trial - 1]})")
print(f"GoT100: {got100_trial} (Trial {trials[got100_trial - 1]})")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(trials) + 1), best_costs, marker="o", linestyle="-", label="Best Cost")
plt.xlabel("Trial Number")
plt.ylabel("Best Cost So Far")
plt.title("Progression of Best Cost Across Trials")
plt.legend()
plt.grid()

plt.savefig("best_cost_progression.png")