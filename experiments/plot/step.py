# Sample Python code to calculate the average number of steps per problem



file_path = "/home/jefferywong/aries/experiments/logs/ablation/llama-3.1-70b/sorting32-em15-llm-0-49.log"
# Read the log file
with open(file_path, "r") as file:
    log_content = file.read()

# Split the log into individual problems
problems = log_content.split("===============================\nSolving problem")

# Initialize variables to count steps
total_steps = 0
total_problems = 0

# Iterate through each problem
for problem in problems[1:]:  # Skip the first element (it's empty or irrelevant)
    # Count the number of steps in the problem
    steps = problem.count("Step ")
    total_steps += steps
    total_problems += 1

# Calculate the average number of steps per problem
average_steps = total_steps / total_problems

print(f"Total problems: {total_problems}")
print(f"Total steps: {total_steps}")
print(f"Average steps per problem: {average_steps:.2f}")


# cot-em1-llm-0-49.log, 15.29
# cot-em5-llm-0-49.log, 14.69
# cot-em10-llm-0-49.log, 12.03
# cot-em15-llm-0-49.log, 12.80

# no-cot-em1-llm-0-49.log, 23.22
# no-cot-em5-llm-0-49.log, 22.03
# no-cot-em10-llm-0-49.log, 18.97
# no-cot-em15-llm-0-49.log, 21.72