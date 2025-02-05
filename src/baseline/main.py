import argparse
import logging
import json

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
    )

    # initalize dataset
    if args.dataset:
        if args.dataset == "openai/openai_humaneval":
            system_prompt = HUMAN_EVAL_PROMPT
            dataset = load_dataset("openai/openai_humaneval")
            data = dataset["test"]
            examples = {}
            # initalize question
            examples["problem"] = [system_prompt + prompt for prompt in data["prompt"]]
            examples["task_id"] = data["task_id"]

    # dumpt the example object
    with open("example.json", "w") as f:
        json.dump(examples, f)

    logger.log(logging.INFO, "Starting beam search")
    beam_search_results = beam_search(examples, llm=llm, prm=prm, config=config)

    logger.log(logging.INFO, "Storing result")
    # store results
    with open(args.output, "w") as f:
        json.dump(beam_search_results, f)


if __name__ == "__main__":
    main()
