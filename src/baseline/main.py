import argparse
import logging
import json
from tqdm import tqdm

import torch
from vllm import LLM
from datasets import load_dataset

from beam_search import beam_search
from reward_models import load_prm
from config import Config


HUMAN_EVAL_PROMPT = """
Solve the following coding problem efficiently and clearly using **python**:

- For simple problems (2 steps or fewer):
Provide a concise code solution only without any explination.

- For complex problems (3 steps or more):
Use this step-by-step format:

# Step 1: [Concise description]
[Actual line of code]

# Step 2: [Concise description]
[Actual line of code]

...

Regardless of the approach, always output only code in correct format.
Do not include any explination, only include them in the necessary comments starting with `#`.
always enclose the code in ```python and ``` to make it more readable.
"""

CODE_CONTEST_EVAL_PROMPT = """
Solve the following coding problem efficiently and clearly using **python**:

- For simple problems (2 steps or fewer):
Provide a concise code solution only without any explination.

- For complex problems (3 steps or more):
Use this step-by-step format:

# Step 1: [Concise description]
[Actual line of code]

# Step 2: [Concise description]
[Actual line of code]

...

Regardless of the approach, always output only code in correct format.
Do not include any explination, only include them in the necessary comments starting with `#`.
always enclose the code in ```python and ``` to make it more readable.

Follow the instruction on input and output condition and format, use input() and print() for input and output respectively.
Different variables in same set of input will be seperated by space, different set of input will be seperate by a newline character.
print each set of output on a single line, seperate by newline character.
Always print the output in the same order as the input.
start the code with def main(): and end with main()

Start your code from here:
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="openai/openai_humaneval")
    argparser.add_argument("--output", type=str, default="humaneval_results.json")
    args = argparser.parse_args()

    # initalize config
    config = Config()
    logger.log(logging.INFO, "Loaded Config: %s", config)

    num_gpus = torch.cuda.device_count()

    prm = load_prm(config)
    
    # initalize vllm model
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
        max_model_len=8192,
    )

    # initalize dataset
    if args.dataset:
        if args.dataset == "openai/openai_humaneval":
            system_prompt = HUMAN_EVAL_PROMPT
            dataset = load_dataset("openai/openai_humaneval")
            input_dataset = dataset["test"]
            datas = []
            for i in range(len(input_dataset)):
                example = {}
                example["problem"] = [system_prompt + input_dataset[i]["prompt"]]
                example["task_id"] = input_dataset[i]["task_id"]
                datas.append(example)
        elif args.dataset == "deepmind/code_contests":
            system_prompt = CODE_CONTEST_EVAL_PROMPT
            dataset = load_dataset("deepmind/code_contests")
            input_dataset = dataset["test"]
            datas = []
            for i in range(len(input_dataset)):
                example = {}
                example["problem"] = [system_prompt + input_dataset[i]["description"]]
                example["task_id"] = input_dataset[i]["name"]
                datas.append(example)

    # dumpt the example object
    # with open("example.json", "w") as f:
    #     json.dump(examples, f)

    logger.log(logging.INFO, "Starting beam search")
    for i in tqdm(range(len(datas)), desc="Processing examples"):
        example = datas[i]
        beam_search_result = beam_search(example, llm=llm, prm=prm, config=config)
        datas[i]["completions"] = beam_search_result["completions"]
        datas[i]["scores"] = beam_search_result["scores"]
        datas[i]["pred"] = beam_search_result["pred"]
        datas[i]["completion_tokens"] = beam_search_result["completion_tokens"]

    logger.log(logging.INFO, "Storing result")
    # store results
    with open(args.output, "w") as f:
        json.dump(datas, f)


if __name__ == "__main__":
    main()
