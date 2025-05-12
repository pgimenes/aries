import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_results_from_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()
    
    query_count_match = re.search(r'query count:\s*(\d+\.?\d*)', content)
    score_match = re.search(r'score:\s*(\d+\.?\d*)', content)
    
    if query_count_match and score_match:
        return float(query_count_match.group(1)), float(score_match.group(1))
    return None

def get_results(base_dir):
    results = []
    for trial_dir in sorted(os.listdir(base_dir)):
        if trial_dir.startswith("trial_"):
            log_path = os.path.join(base_dir, trial_dir, "spec.log")
            if os.path.exists(log_path):
                result = extract_results_from_log(log_path)
                if result:
                    results.append((trial_dir, result))
    return results

def pareto_frontier(points):
    """ Compute the Pareto frontier (minimizing both query count and score) """
    sorted_points = sorted(points, key=lambda x: (x[0], x[1]))
    pareto = []
    best_score = np.inf
    for q, s in sorted_points:
        if s < best_score:
            pareto.append((q, s))
            best_score = s
    return pareto

def plot_results(results):
    query_counts, scores = zip(*results)
    pareto_points = pareto_frontier(results)
    pareto_x, pareto_y = zip(*pareto_points)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(query_counts, scores, label='Trials', alpha=0.7)
    plt.plot(pareto_x, pareto_y, 'r-', linewidth=2, label='Pareto Frontier')
    plt.xlabel('Query Count')
    plt.ylabel('Score')
    plt.title('Query Count vs Score with Pareto Frontier')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    base_directory = "search/human_eval"  # Change this to your actual base directory
    
    results = get_results(base_directory)
    trials, scores, queries = [], [], []
    
    for trial, result in results:
        trials.append(trial)
        scores.append(result[1])
        queries.append(result[0])
    
    avg_score = np.mean(scores)
    avg_query = np.mean(queries)

    alpha = avg_query / (avg_query + avg_score)

    costs = {}
    for idx, trial in enumerate(trials):
        costs[trial] = alpha * queries[idx] + (1 - alpha) * scores[idx]

    # sort trials by cost
    sorted_trials = sorted(costs.items(), key=lambda x: x[1])

    print(f"Total queries: {np.sum([10 * i for i in queries]):.2f}")

    print("Trials sorted by cost:")
    for trial, cost in sorted_trials:
        print(f"{trial}: {cost:.2f}")

    # if results:
    #     plot_results(results)
    # else:
    #     print("No valid results found.")
