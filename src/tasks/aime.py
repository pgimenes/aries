from llm import llm, async_llm
from .common import (
    _common_tot_schedule, 
    _common_got_schedule, 
    common_keepbest, 
    PARSE_OUT_DICT,
)
import asyncio
from copy import copy

problem_definition = ""

actions = {
    "split": {
    },
    "sort": {
    },
    "aggregate": {
    },
    "refine": {
    },
    "score": {
    },
    "keepbest": {
    },
    "groundtruth": {
    }
}

examples = [""]

# Implementation

def split(
    graph,
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):

    return graph, False

system_prompt = """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem."""

solve_prompt = """
<instruction>
{system_prompt}
</instruction>

<question>
{input}
</question>
"""

async def async_generate(
    graph,
    nodes,
    model = "",
    multiplicity: int = 1,
):
    return await asyncio.gather(
        *[
            async_llm(
                solve_prompt.format(
                    system_prompt=system_prompt,
                    input=graph.nodes[int(node)]["thought"]
                ),
                model=model,
                stop=["\n\n"],
            ) for node in nodes
        ],
    )

def generate(
    graph, 
    nodes,
    model = "",
    run_async = True,
    multiplicity: int = 1,
):
    
    # 1. Get LLM responses
    if run_async:
        outs = asyncio.run(async_generate(graph, nodes, model=model))
        outs = {
            nodes[i]: outs[i] for i in range(len(nodes))
        }
    else:
        outs = {
            node: llm(
                sort_prompt.format(input=graph.nodes[int(node)]["thought"]),
                model=model,
            )[0] for node in nodes
        }

    # 2. Update graph
    nodes_to_score = []
    for node in nodes:
        next_thought = outs[node][0]

        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=next_thought,
            score=None,
            original=graph.nodes[int(node)]["thought"],
        )
        graph.add_edge(int(node), idx)


    # 3. Score all the sorted nodes
    for node in nodes_to_score:
        graph, _ = score(graph, [node], model=model)

    return graph, False


refine_prompt = ""

async def async_refine(
    graph,
    nodes,
    model = "",
    multiplicity: int = 1,
):
    return await asyncio.gather(
        *[
            async_llm(
                refine_prompt.format(
                    input=graph.nodes[int(node)]["original"],
                    incorrectly_sorted=graph.nodes[int(node)]["thought"],
                ),
                model=model,
            ) for node in nodes
        ],
    )

def refine(
    graph, 
    nodes,
    model = "",
    run_async = True,
    multiplicity: int = 1,
):
    # 1. Filter out nodes that are already correct
    nodes_to_refine = []
    refined_nodes = {}
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        # Skip if the node is already correct
        if graph_node.get("score", None) is not None and graph_node["score"] == 0:
            refined_nodes[node] = {
                "thought": graph_node["thought"],
                "score": 0,
                "original": graph_node["original"],
            }
        else:
            nodes_to_refine.append(node)

    if run_async:
        outs = asyncio.run(async_refine(graph, nodes_to_refine, model=model))
        outs = {
            nodes_to_refine[i]: outs[i] for i in range(len(nodes_to_refine))
        }
    else:
        outs = {
            node: llm(
                refine_prompt.format(
                    input=graph.nodes[int(node)]["original"],
                    incorrectly_sorted=graph.nodes[int(node)]["thought"],
                ),
                model=model,
            )[0] for node in nodes_to_refine
        }
    
    # 2. Update the graph
    nodes_to_score = []
    for node in nodes:
        if node in nodes_to_refine:
            original = graph.nodes[int(node)]["original"]
            output = outs[node][0]

            # Parse with regex
            try:
                reason = re.search(r"<reason>(.*?)</reason>", output).group(1)
                output = re.search(r"<output>(.*?)</output>", output).group(1)
            except:
                output = outs[node][0]
        else:
            original, output = refined_nodes[node]["original"], refined_nodes[node]["thought"]

        for k, v in PARSE_OUT_DICT.items():
            output = output.replace(k, v)

        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=output,
            original=original,
            score=None,
        )
        graph.add_edge(node_idx, idx)
        nodes_to_score.append(idx)

    # 3. Score all the refined nodes
    for node in nodes_to_score:
        graph, _ = score(graph, [node], model=model)

    return graph, False

def score(
    graph, 
    nodes,
    model = "",
    multiplicity: int = 1,
):

    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        # score = ...
        breakpoint()
        
        graph.nodes[node_idx]["score"] = score

    return graph, False

def keepbest(
    graph, 
    nodes,
    model = "",
    multiplicity: int = 1,
):
    # Score all non-scored nodes
    graph, _ = score(graph, nodes, model=model)
    return common_keepbest(graph, nodes, model)

aggregate_prompt = ""

async def async_aggregate(
    graph,
    nodes,
    model = "",
    multiplicity: int = 1,
):
    
    prompt = aggregate_prompt.format(
        inputs=[graph.nodes[int(node)]["thought"] for node in nodes],
    )

    return await asyncio.gather(
        *[
            async_llm(
                prompt,
                model=model
            ) for _ in range(multiplicity)
        ],
    )

def aggregate(
    graph, 
    nodes,
    model = "",
    multiplicity: int = 1,
    run_async: bool = True,
):
    # 1. Run the aggregate attempts
    if run_async:
        outs = asyncio.run(
            async_aggregate(
                graph,
                nodes,
                model=model,
                multiplicity=multiplicity,
            )
        )
        outs = [out[0] for out in outs]
    
    else:
        outs = [
            llm(
                aggregate_prompt.format(
                    input1=graph.nodes[int(nodes[0])]["thought"],
                    input2=graph.nodes[int(nodes[1])]["thought"]
                ),
                model=model
            )[0] for _ in range(multiplicity)
        ]

    # 2. Extract aggregation original
    combined_list = []
    for node in nodes:
        if isinstance(graph.nodes[int(node)]["thought"], list):
            thought = graph.nodes[int(node)]["thought"]
        else:
            thought = ast.literal_eval(graph.nodes[int(node)]["thought"])

        combined_list += thought

    # 3. Update the graph
    for out in outs:
        for k, v in PARSE_OUT_DICT.items():
            out = out.replace(k, v)
        
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx, 
            thought=out,
            score=None,
            original = combined_list,
        )

        for node in nodes:
            node_idx = int(node)
            graph.add_edge(node_idx, idx)

    # Score the aggregated node
    graph, _ = score(graph, [idx], model)

    return graph, False

def groundtruth(
    graph, 
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    # read from dataset...

    return graph, False

# Baselines
def io(
    graph,
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    return generate(graph, nodes, model)

def _tot_schedule(
        width: int,
        depth: int,
) -> int:
    return _common_tot_schedule(
        width=width,
        depth=depth,
        generate_action="solve",
        refine_action="refine",
    )

def _got_schedule(
    branches:int,
    generate_attempts:int,
    aggregate_attempts:int,
    post_aggregate_keepbest: bool,
    post_aggregate_refine: bool,
    refine_attempts:int,
) -> int:
    return _common_got_schedule(
        branches=branches,
        generate_action="solve",
        generate_attempts=generate_attempts,
        aggregate_attempts=aggregate_attempts,
        post_aggregate_keepbest=post_aggregate_keepbest,
        post_aggregate_refine=post_aggregate_refine,
        refine_attempts=refine_attempts,
    )

cot_prompt = ""

def cot(
    graph, 
    nodes,
    model = "",
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(cot_prompt.format(input=graph_node["thought"]), model=model)[0]
        
        # Parse the result
        for k, v in PARSE_OUT_DICT.items():
            out = out.replace(k, v)
        
        # 2. Update the graph
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx, 
            thought=out, 
            score=None,
            original = graph_node["thought"],
        )
        graph.add_edge(node_idx, idx)

    return graph, False