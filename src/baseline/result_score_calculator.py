import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="humaneval_results.json")
args = parser.parse_args()

with open(args.input_file, "r") as f:
    input_data = json.load(f)

total_score = []
total_idx = 0

for i in tqdm(range(len(input_data))):
    completion = input_data[i]["completions"][0][0]
    problem_idx = input_data[i]["task_id"].split("/")[-1]
    score = input_data[i]["scores"][0]
    
    # flatten the score list
    flatten_score = [item for sublist in score for item in sublist]
    total_score.append(len(flatten_score))
    total_idx += 1
    
print(f"Total Score: {sum(total_score)}")
print(f"average score: {sum(total_score)/total_idx}")