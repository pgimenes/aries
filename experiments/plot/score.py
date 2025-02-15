# import re

# def extract_all_groundtruth_scores(log_file):
#     scores = []
#     with open(log_file, 'r', encoding='utf-8') as file:
#         content = file.read()
        
#         # Find all occurrences of groundtruth action and extract relevant section
#         # action_sections = re.findall(r'(Action: groundtruth\nNodes: \[(\d+).*?Edges:)', content, re.DOTALL)
#         # action_sections = re.findall(r'(Action: groundtruth\nNodes: \[(\d+).*?Result:)', content, re.DOTALL)

#         # action_sections = re.findall(r'(Action: groundtruth\nNodes: \[(\d+)[^]*Result)', content, re.DOTALL)
#         action_sections = re.findall(r'(\nAction: groundtruth\nNodes: \[(\d+)[\s\S]*?Result:)', content, re.DOTALL)
#         print(action_sections)

#         # action_sections = re.findall(r'(Action: groundtruth\nNodes: \[(\d+).*?)(?:(?!Action: ).)*?Result:', content, re.DOTALL)

#         for section, node_id in action_sections:
#             # Extract the score for each node within its action space

#             node_pattern = rf"{node_id}: .*?'score': (\d+)"
#             score_match = re.search(node_pattern, section)
            
#             if score_match:
#                 scores.append(int(score_match.group(1)))
    
#     return scores

# # Example usage:
# log_file_path = "/home/jefferywong/aries/experiments/logs/ablation/llama-3.1-70b/sorting32-em1-llm-test.log"  # Change this to the correct path
# last_score = extract_all_groundtruth_scores(log_file_path)
# print(len(last_score))
# avg_score = sum(last_score) / len(last_score) if len(last_score) > 0 else 1000
# print(f"Score of the last groundtruth node: {last_score}")
# print(f"Average score: {avg_score}")



import re

total_scores = []
# File path
file_path = "/home/jefferywong/aries/experiments/logs/ablation/llama-3.1-70b/sorting32-em10-llm-0-49.log"

# Regex pattern to match the groundtruth action and extract the node number
# groundtruth_pattern = re.compile(r"Action: groundtruth\nNodes: \[(\d+)")
groundtruth_pattern = re.compile(r"Action: groundtruth\nNodes: \[([\d, ]+)\]")

# Regex pattern to match the node's score
score_pattern = re.compile(r"(\d+): \{'thought':.*?'score': (\d+),")

# Read the file content
with open(file_path, "r") as file:
    log_content = file.read()

# Split the log content into individual problems
problems = log_content.split("===============================\nSolving problem")

# Iterate over each problem and find the last groundtruth action's node score
for i, problem in enumerate(problems[1:]):  # Skip the first split as it's not a problem
    # Find all groundtruth actions in the problem
    groundtruth_matches = groundtruth_pattern.findall(problem)
    if not groundtruth_matches:
        print(f"Problem {problems.index(problem)}: No groundtruth action found.")
        continue  # No groundtruth action in this problem
    
    # Get the last groundtruth action's node number
    last_groundtruth_node = groundtruth_matches[-1]
    
    # Find the score of the last groundtruth node
    node_pattern = re.compile(rf"{last_groundtruth_node}: .*?'score': (\d+)")
    score_match = node_pattern.findall(problem)

    if score_match:
        # last_score = int(score_match[-1])
        # last_score = int(score_match[-1])
        last_score = int(max(score_match, key=lambda x: int(x)))
        if last_score >=100000:
            continue
        total_scores.append(last_score)
        print(f"Problem {problems.index(problem)}: Score of the last groundtruth node ({last_groundtruth_node}): {last_score}")
    else:
        print(f"Problem {problems.index(problem)}: No score found for the last groundtruth node ({last_groundtruth_node}).")


avg_score = sum(total_scores) / len(total_scores) if len(total_scores) > 0 else 1000   
print(f"Average score: {avg_score}")