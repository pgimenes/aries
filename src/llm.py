import os
import openai
from openai import OpenAI, AsyncOpenAI
from os import getenv
import backoff 
import numpy as np
import asyncio

# AZURE
# ================================================

# openai.api_type = "azure"
# openai.api_version = "2024-10-01-preview"

# api_key = os.getenv("OPENAI_API_KEY", "")
# if api_key != "":
#     openai.api_key = api_key
# else:
#     print("Warning: OPENAI_API_KEY is not set")
    
# api_base = os.getenv("OPENAI_API_BASE", "")
# if api_base != "":
#     print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
#     openai.api_base = api_base

# Local hosting
# ================================================

# import openai
client = OpenAI(
    base_url="http://127.0.0.1:30000/v1", 
    api_key="EMPTY",
)
async_client = AsyncOpenAI(
    base_url="http://127.0.0.1:30000/v1", 
    api_key="EMPTY",
)

# OpenRouter
# ================================================

# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key=getenv("OPENROUTER_API_KEY"),
# )

def get_proposal_perplexities(res):
    interm = []
    proposal_ppl = []

    for idx, token in enumerate(res["choices"][0]["logprobs"]["content"]):
        interm.append(token["logprob"])
        
        if "\n" in token["token"] or idx == len(res["choices"][0]["logprobs"]["content"]) - 1:
            interm = np.array(interm)
            ppl = np.exp(-np.mean(interm))
            proposal_ppl.append(ppl)
            interm = []

    return proposal_ppl

def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def llm(
    prompt,
    model=None, 
    temperature=1, 
    max_tokens=1000000, 
    n=1, 
    stop=None, 
    return_ppl=False
) -> list:
    messages = [{"role": "user", "content": prompt}]    
    outputs = []

    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            n=cnt, 
            stop=stop,
        )

        try:
            msg = [choice.message.content for choice in res.choices]
        except:
            raise ValueError(f"Failed to get messages from response: {res}")

        outputs.extend(msg)
        
        if return_ppl:
            proposals = msg[0].split("\n")
            proposal_ppl = get_proposal_perplexities(res)
            # assert len(proposals) == len(proposal_ppl), f"Found {len(proposals)} proposals but {len(proposal_ppl)} perplexities"
            proposal_ppl = list(zip(proposals, proposal_ppl))

    if return_ppl:
        return outputs, proposal_ppl
    else:
        return outputs

async def async_llm(
    prompt,
    model=None, 
    temperature=1, 
    max_tokens=1000, 
    n=1, 
    stop=None, 
    return_ppl=False
) -> list:
    messages = [{"role": "user", "content": prompt}]    
    outputs = []

    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = await async_client.chat.completions.create(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            n=cnt, 
            stop=stop,
        )

        try:
            msg = [choice.message.content for choice in res.choices]
        except:
            raise ValueError(f"Failed to get messages from response: {res}")

        outputs.extend(msg)
        
        if return_ppl:
            proposals = msg[0].split("\n")
            proposal_ppl = get_proposal_perplexities(res)
            # assert len(proposals) == len(proposal_ppl), f"Found {len(proposals)} proposals but {len(proposal_ppl)} perplexities"
            proposal_ppl = list(zip(proposals, proposal_ppl))

    if return_ppl:
        return outputs, proposal_ppl
    else:
        return outputs