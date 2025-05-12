import os
import re
from collections import defaultdict

def count_actions_in_logs(base_dir, max_trial):
    action_keywords = {"split", "generate", "refine"}
    results = defaultdict(lambda: defaultdict(int))
    
    for trial_dir in os.listdir(base_dir):
        if trial_dir.startswith("trial_"):
            trial_num = int(trial_dir.split("_")[1])
            if trial_num > max_trial:
                continue
            trial_path = os.path.join(base_dir, trial_dir)
            if os.path.isdir(trial_path):
                for log_file in os.listdir(trial_path):
                    if log_file.startswith("idx-") and log_file.endswith(".log"):
                        log_path = os.path.join(trial_path, log_file)
                        with open(log_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                match = re.match(r"Action:\s*(\w+)", line)
                                if match:
                                    action = match.group(1).lower()
                                    if action in action_keywords:
                                        results[log_path][action] += 1
    
    return results

if __name__ == "__main__":
    base_directory = "search/human_eval"  # Change this to your actual directory path
    
    for max_trial in [22, 43, 85]:
        action_counts = count_actions_in_logs(base_directory, max_trial)
        
        total_queries = 0
        for file, counts in action_counts.items():
            for action, count in counts.items():
                total_queries += count

        print(f"Total queries for trials up to {max_trial}: {total_queries}")
