import os
import asyncio
from smolagents import CodeAgent, OpenAIServerModel
from datasets import load_dataset
from tqdm import tqdm
import os
import re

# Setup model and agent
model = OpenAIServerModel(
    model_id="openai/gpt-4o",
    api_base="https://openrouter.ai/api/v1",
    api_key="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",
)

agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["*"],
    add_base_tools=True
)

# Load HumanEval dataset
dataset = load_dataset("openai_humaneval")["test"]

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

def parse_log_file(log_path):
    """Parse a single log file to find if the solution was correct."""
    with open(log_path, "r") as f:
        content = f.read()

    match = re.search(r"Correct:\s*(True|False)", content)
    if match:
        return match.group(1) == "True"
    else:
        print(f"Warning: No correctness info found in {log_path}")
        return False  # Default to incorrect if missing

def compute_accuracy(logs_dir="logs"):
    """Parse all log files and compute accuracy."""
    log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")]

    if not log_files:
        print("No log files found.")
        return

    results = []
    errors = []
    for log_file in log_files:
        is_correct = parse_log_file(log_file)
        results.append(is_correct)

        if not is_correct:
            errors.append(log_file)

    num_correct = sum(results)
    total = len(results)
    accuracy = num_correct / total

    print(f"Parsed {total} problems.")
    print(f"Correct: {num_correct}")
    print(f"Accuracy: {accuracy:.2%}")

def generate_problem_script(sample):
    """Generate a small Python script as a string that solves the problem."""
    prompt = sample["prompt"]
    entry_point = sample["entry_point"]
    test = sample["test"]
    task_id = sample["task_id"]

    full_prompt = (
        f"Write a Python function to complete the following task:\n\n{prompt}\n\n"
        f"Make sure the function is named {entry_point}. "
        "Return a string containing the code for the function and necessary imports."
    )

    script = f"""
import sys
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id="openai/gpt-4o",
    api_base="https://openrouter.ai/api/v1",
    api_key="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",
)

agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["*"],
    add_base_tools=True
)

def evaluate_completion(completion_code: str, test_code: str) -> bool:
    try:
        local_namespace = {{}}
        exec(completion_code, local_namespace)
        exec(test_code, local_namespace)
        return True
    except Exception as e:
        return False

full_prompt = {full_prompt!r}
test_code = {test!r}

try:
    completion = agent.run(full_prompt)
except Exception as e:
    print(f"Agent error: {{e}}", file=sys.stderr)
    completion = ""

is_correct = evaluate_completion(completion, test_code)

print(f"Task: {task_id}")
print(f"Correct: {{is_correct}}")
print(f"Completion:\\n{{completion}}")
"""
    return script

async def solve_problem(sample):
    """Spawn a separate process for each problem."""
    task_id = sample["task_id"]
    script = generate_problem_script(sample)

    # Save temporary script
    script_path = f"temp_scripts/{task_id}.py"
    os.makedirs("temp_scripts", exist_ok=True)
    os.makedirs("temp_scripts/HumanEval", exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)

    # Setup log file
    log_path = f"logs/{task_id.replace('/', '_')}.log"
    log_file = open(log_path, "w")

    # Launch the subprocess
    process = await asyncio.create_subprocess_exec(
        "python", script_path,
        stdout=log_file,
        stderr=log_file,
    )

    await process.communicate()
    log_file.close()

async def main():
    tasks = [solve_problem(sample) for sample in dataset]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating HumanEval"):
        await f

if __name__ == "__main__":
    asyncio.run(main())
    compute_accuracy()
