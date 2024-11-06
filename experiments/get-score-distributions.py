import re
import numpy

task = "set-intersection64"
method = "cot"
tot_final_step = 78

def extract_scores(log_content):
    # Regular expression to find 'score' followed by any number

    if method == "tot":
        score_pattern = fr"{tot_final_step}:.*'score':\s*(\d+)"
    else:
        score_pattern = r"'score':\s*(\d+)"
    
    # Find all matches and convert them to integers
    scores = [int(match) for match in re.findall(score_pattern, log_content)]
    
    return scores

# Load your log content from a file (e.g., "log.txt")
with open(f"logs/llama-3.1-70b/{task}-{method}.log", "r") as file:
    log_content = file.read()

# Extract and parse scores
# ------------------------------

scores = extract_scores(log_content)

# if method != "tot":
scores = scores[::2]

scores = numpy.array(scores)

# remove all elements that are 1000000
scores = scores[scores != 1000000]
print("Parsed Scores:", scores)

# Get distribution statistics
# ------------------------------

# get quartiles, min, max, and mean, median
quartiles = numpy.percentile(scores, [25, 50, 75])
data_min = numpy.min(scores)
data_max = numpy.max(scores)
data_mean = numpy.mean(scores)

print(f"Elements without mistakes: {len(scores)}")
print(f"min, Q1, mean, median, Q3, max")
print(f"{data_min}, {quartiles[0]}, {data_mean}, {quartiles[1]}, {quartiles[2]}, {data_max}")