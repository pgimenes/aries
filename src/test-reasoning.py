import asyncio
from datasets import load_dataset
from llm import async_llm
import re
from sympy import sympify

# Load dataset
# ds_name = "openai/gsm8k"
# config = "main"
# split = "test"
# body_key = ""
# question_key = "question"
# answer_key = "answer"

# ds_name = "ChilleD/SVAMP"
# config = ""
# split = "train"
# body_key = "Body"
# question_key = "Question"
# answer_key = "Answer"

# ds_name = "deepmind/aqua_rat"
# config = "raw"
# split = "train"
# body_key = "question"
# question_key = "options"
# answer_key = "correct"

# ds_name = "ChilleD/MultiArith"
# config = ""
# split = "train+test"
# body_key = ""
# question_key = "question"
# answer_key = "final_ans"

# ds_name = "tau/commonsense_qa"
# config = ""
# split = "validation"
# body_key = "question"
# question_key = "choices"
# answer_key = "answerKey"

# ds_name = "ChilleD/StrategyQA"
# config = ""
# split = "train"
# body_key = ""
# question_key = "question"
# answer_key = "answer"

# ds_name = "ChilleD/LastLetterConcat"
# config = ""
# split = "train+test"
# body_key = ""
# question_key = "question"
# answer_key = "answer"

ds_name = "hendrycks/competition_math"
config = ""
split = "test"
body_key = ""
question_key = "problem"
answer_key = "solution"

# ds_name = "qq8933/AIME_1983_2024"
# config = ""
# split = "train"
# body_key = ""
# question_key = "Question"
# answer_key = "Answer"

# ds_name = "hotpotqa/hotpot_qa"
# config = "distractor"
# split = "validation"
# body_key = "question"
# question_key = "context"
# answer_key = "answer"

# ds_name = "cais/mmlu"
# config = "all"
# split = "validation"
# body_key = "question"
# question_key = "choices"
# answer_key = "answer"

# Load the dataset
# ============================

args = [ds_name, config] if config != "" else [ds_name]
kwargs = {"split": split}
ds = load_dataset(*args, **kwargs)
size = len(ds)
max_reqs = 1000 if size > 1000 else size
# max_reqs = 100
# max_reqs = size

# Define the prompt template
# ============================
if ds_name == "hendrycks/competition_math":
    
    prompt = """<instruction>Solve the following question. The final output should be within <answer> tags, with the final numerical value in latex format within a \\boxed{{...}}, as shown in the examples.</instruction>

<example>
<question>Let \\[f(x) = \\left\\{{\n\\begin{{array}}{{cl}} ax+3, &\\text{{ if }}x>2, \\\\\nx-5 &\\text{{ if }} -2 \\le x \\le 2, \\\\\n2x-b &\\text{{ if }} x <-2.\n\\end{{array}}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).</question>

For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{{0}}$.

<answer>$\\boxed{{0}}$</answer>
</example>

<example>
<question>A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?</question>

Let $x$ be the number of band members in each row for the original formation, when two are left over.  Then we can write two equations from the given information: $$rx+2=m$$ $$(r-2)(x+1)=m$$ Setting these equal, we find: $$rx+2=(r-2)(x+1)=rx-2x+r-2$$ $$2=-2x+r-2$$ $$4=r-2x$$ We know that the band has less than 100 members.  Based on the first equation, we must have $rx$ less than 98.  We can guess and check some values of $r$ and $x$ in the last equation.  If $r=18$, then $x=7$, and $rx=126$ which is too big.  If $r=16$, then $x=6$, and $rx=96$, which is less than 98.  Checking back in the second formation, we see that $(16-2)(6+1)=14\\cdot 7=98$ as it should.  This is the best we can do, so the largest number of members the band could have is $\\boxed{{98}}$.'

<answer>$\\boxed{{98}}$</answer>
</example>

<question>
{question}
</question>
"""

else:
    prompt = """<instruction>Solve the following question. The final output should be within <answer> tags, following the examples. If the answer is just numerical, only output the number with no symbols. If there are options, only output the letter with nothing else. If it's a yes or no question, output yes/no. If the answer is a single word, ouput only the word with no other characters.</instruction>

<example>
<question>What is the sum of 2 and 2?</question>
This is a simple question. The answer is 4.
<answer>4</answer>
</example>

<example>
<question>What are the first 5 letters of the alphabet?</question>
<answer>abcde</answer>
</example>

<example>
<question>What is 10 * 10?
A. 100
B. 200
C. 300
D. 400
</question>

<answer>A</answer>
</example>

<example>
<question>Is 8 higher than 4?</question>
<answer>yes</answer>
</example>

<question>
{question}
</question>
"""

# Initialize tracking lists and counters
successes = []
failures = []
finished_cnt = 0

# Async function to process each item
# ======================================

async def process_item(idx, item):
    global finished_cnt
    
    body = item.get(body_key, "")
    question = item.get(question_key, "")

    if isinstance(question, dict):
        # list context for hotpotqa
        if ds_name == "hotpotqa/hotpot_qa":
            q = "\nContext:\n"
            for t, s in zip(question["title"], question["sentences"]):
                q += f"{t}: {''.join(s)}\n"
        
        # list options for commonsense_qa
        elif ds_name == "tau/commonsense_qa":
            q = "\n"
            for l, t in zip(question["label"], question["text"]):
                q += f"{l}. {t}\n"

        question = q

    if isinstance(question, list):
        opts = ["A", "B", "C", "D"]
        q = "\nOptions:"
        for i, quest in enumerate(question):
            q += f"\n{opts[i]}. {quest}"
        question = q

    question = f"{body} {question}"

    if not question:
        print(f"Item {idx} is missing a question.")
        failures.append((idx, "Missing question"))
        return

    # Extract the reference answer
    answer = item[answer_key]
    
    if isinstance(answer, str) and "####" in answer:
        answer = answer.split("#### ")[1]

    if ds_name == "hendrycks/competition_math":
        # extract from \boxed{...}
        answer = answer.replace(" ", "")
        match = re.search(r"\\boxed\{(.+?)\}\$", answer)
        if match:
            answer = f"{match.group(1)}"

    if answer in ["yes", "no"]:
        answer = True if answer == "yes" else False

    if ds_name == "cais/mmlu":
        opts = "ABCD"
        answer = opts[answer]

    try:
        # Call the async LLM function
        res = await async_llm(prompt.format(question=question))
        
        # Extract the LLM-generated answer
        match = re.search(r'<answer>(.*?)</answer>', res[0])
        if not match:
            print(f"LLM response does not contain a valid answer for item {idx}: {res[0]}")
            failures.append((idx, question.split("Context:")[0], answer, "No valid answer"))
            return

        llm_answer = match.group(1)

        if llm_answer in ["yes", "no"]:
            llm_answer = True if llm_answer == "yes" else False

        if ds_name == "ChilleD/LastLetterConcat":
            llm_answer = llm_answer.lower()
            llm_answer = re.sub(r"\s+", "", llm_answer)

        if ds_name == "deepmind/aqua_rat":
            answer = answer.lower()
            llm_answer = llm_answer.lower()

        if ds_name == "hendrycks/competition_math":
            # extract from \boxed{...}
            llm_answer = llm_answer.replace(" ", "")
            match = re.search(r"\\boxed\{(.+?)\}\$", llm_answer)
            if match:
                llm_answer = f"{match.group(1)}"

        # Compare LLM's answer with the reference answer
        if llm_answer == answer:
            print(f"Item {idx} result: success")
            successes.append((idx, question.split("Context:")[0], answer, llm_answer))
        else:
            print(f"Item {idx} result: failure")
            failures.append((idx, question.split("Context:")[0], answer, llm_answer))

    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        failures.append((idx, question.split("Context:")[0], answer, "Processing error"))
    finally:
        finished_cnt += 1
        print(f"Progress: {finished_cnt}/{max_reqs}")

# Main async function
async def main():
    global finished_cnt
    tasks = []

    for idx, item in enumerate(ds):
        if idx >= max_reqs:  # Limit the number of requests
            break
        tasks.append(process_item(idx, item))

    # Run tasks concurrently
    await asyncio.gather(*tasks)

    print("\nFinal Results:")
    print("Successes:", successes)
    print("Failures:", failures)
    print(f"Acc: {len(successes)}/{max_reqs}")

if __name__ == "__main__":
    asyncio.run(main())