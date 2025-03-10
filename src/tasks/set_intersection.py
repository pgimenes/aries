from llm import llm, async_llm
import ast
import asyncio
from .common import (
    _common_tot_schedule, 
    _common_got_schedule, 
    common_keepbest, 
    PARSE_OUT_DICT,
)

problem_definition = "Find the intersection of two sets of numbers."

actions = {
    "split": {
        "description": "Split set 2 into smaller subsets.",
        "preconditions": "",
        "effects": "New nodes are created, containing copies of set 1, and subsets of set2.",
    },
    "intersect": {
        "description": "Find the intersection of the subsets in the selected nodes.",
        "preconditions": "",
        "effects": "For each selected node, a new node is created containing the intersection of set 1 and set2.",
    },
    "aggregate": {
        "description": "Merge the intersected subsets of the selected nodes into a single set intersection.",
        "preconditions": "",
        "effects": "A new node is created, connected to the selected nodes.",
    },
    # "refine": {
    #     "description": "Refine an intersection by fixing any existing mistakes.",
    #     "preconditions": "The node should have a non-zero score.",
    #     "effects": "A new node is created with a refined sorting of the selected node, connected to the selected node.",
    # },
    # "score": {
    #     "description": "Count the number of mistakes in the node.",
    #     "preconditions": "",
    #     "effects": "The node is annotated with a score, which is the number of mistakes. The node may also be annotated with a feedback dictionary. The missing_elements key indicates the number of elements that are missing from the sorted list. The extra_elements key indicates the number of elements that are in the sorted list but not in the original list.",
    # },
    "keepbest": {
        "description": "Out of the selected nodes, keep the one with the highest score, and delete the rest.",
        "preconditions": "The selected nodes must have been scored.",
        "effects": "All selected nodes are deleted, but the one with the highest score is duplicated as a new node.",
    },
    "groundtruth": {
        "description": "Compare a node to the ground truth intersection of the sets in node 0.",
        "preconditions": "",
        "effects": "The node is annotated with 'matches_ground_truth: True' or 'False'.",
    }
}

examples = [
    """<example>
INPUT:
Previous actions:

Current graph:

Nodes:
0: {'set1': "[11, 60, 1, 49, 21, 33, 14, 56, 54, 15, 23, 40, 45, 22, 7, 28, 20, 46, 51, 6, 34, 37, 3, 50, 17, 8, 25, 0, 35, 47, 18, 19]", "set2": "[31, 11, 4, 63, 38, 58, 59, 24, 61, 14, 32, 39, 27, 46, 48, 19, 52, 57, 50, 56, 3, 2, 53, 29, 5, 37, 62, 41, 36, 12, 49, 16]"}

Edges:

OUTPUT:

<analysis>
A. Action history: No actions have been taken yet. 

B. Graph state: The graph currently has 1 node and 0 edges. Node 0 contains the initial problem. 

C. Strategy analysis: The strategy for solving the problem has not been determined yet.

D. Next action options
    1. Attempt to intersect the sets directly. This may be effective if the sets are small and the intersection is simple.

    2. Decompose the sets by splitting them into smaller sets. This may be necessary if the sets are too large to intersect directly.
</analysis>

<next_action>
split
</next_action>

<nodes>
[0]
</nodes>

<attempts>
1
</attempts>

<explanation>
The sets are too large to intersect directly, so we need to split them into smaller sets first.
</explanation>

</example>""",
]

# Implementation

split_prompt32 = """<Instruction> Split the following list of 32 numbers into 2 lists of 16 numbers each, the first list should contain the first 16 numbers and the second list the second 16 numbers.
Only output the 2 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [13, 16, 30, 6, 21, 7, 31, ...],
    "List 2": [25, 24, 10, 4, 27, 0, 14, ...]
}} </Instruction>

<Example>
Input: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Output:
{{
    "List 1": [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22],
    "List 2": [33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
}}
</Example>

Input: {input}
Output: """

split_prompt64 = """<Instruction> Split the following list of 64 numbers into 4 lists of 16 numbers each, the first list should contain the first 16 numbers, the second list the second 16 numbers, the third list the third 16 numbers and the fourth list the fourth 16 numbers.
Only output the 4 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [13, 35, 20, 96, 34, 18, 47, ...],
    "List 2": [127, 126, 9, 21, 16, 77, 22, ...],
    "List 3": [111, 122, 85, 73, 42, 105, 123, ...],
    "List 4": [15, 33, 59, 67, 57, 104, 8, ...]
}} </Instruction>

<Example>
Input: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Output:
{{
    "List 1": [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1],
    "List 2": [118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29],
    "List 3": [58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98],
    "List 4": [52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
}}
</Example>

Input: {input}
Output: """

split_prompt128 = """<Instruction> Split the following list of 128 numbers into 8 lists of 16 numbers each, the first list should contain the first 16 numbers, the second list the second 16 numbers, the third list the third 16 numbers and the fourth list the fourth 16 numbers, the fifth list the fifth 16 numbers and so on.
Only output the 8 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [13, 35, 20, 96, 34, 18, 47, ...],
    "List 2": [127, 126, 9, 21, 16, 77, 22, ...],
    "List 3": [111, 122, 85, 73, 42, 105, 123, ...],
    "List 4": [15, 33, 59, 67, 57, 104, 8, ...],
    "List 5": [30, 89, 76, 12, 65, 84, 32, ...],
    "List 6": [40, 7, 100, 108, 50, 14, 28, ...],
    "List 7": [24, 53, 90, 17, 91, 81, 124, ...],
    "List 8": [63, 5, 46, 125, 93, 49, 66, ...]
}} </Instruction>

<Example>
Input: [124, 100, 176, 65, 28, 214, 201, 67, 127, 40, 193, 26, 212, 196, 12, 205, 39, 162, 190, 152, 186, 182, 125, 229, 114, 70, 189, 242, 155, 32, 47, 113, 99, 177, 57, 103, 156, 107, 226, 188, 31, 25, 49, 171, 17, 64, 254, 217, 35, 22, 248, 88, 251, 219, 84, 82, 167, 14, 116, 92, 105, 148, 119, 58, 13, 160, 81, 120, 74, 94, 173, 38, 101, 104, 91, 135, 208, 118, 136, 87, 75, 41, 197, 76, 249, 240, 86, 246, 231, 44, 29, 141, 241, 227, 239, 142, 222, 115, 129, 255, 216, 153, 0, 110, 36, 130, 6, 102, 209, 37, 143, 30, 51, 146, 71, 27, 181, 183, 2, 122, 77, 184, 56, 225, 108, 83, 106, 15]
Output:
{{
    "List 1": [124, 100, 176, 65, 28, 214, 201, 67, 127, 40, 193, 26, 212, 196, 12, 205],
    "List 2": [39, 162, 190, 152, 186, 182, 125, 229, 114, 70, 189, 242, 155, 32, 47, 113],
    "List 3": [99, 177, 57, 103, 156, 107, 226, 188, 31, 25, 49, 171, 17, 64, 254, 217],
    "List 4": [35, 22, 248, 88, 251, 219, 84, 82, 167, 14, 116, 92, 105, 148, 119, 58],
    "List 5": [13, 160, 81, 120, 74, 94, 173, 38, 101, 104, 91, 135, 208, 118, 136, 87],
    "List 6": [75, 41, 197, 76, 249, 240, 86, 246, 231, 44, 29, 141, 241, 227, 239, 142],
    "List 7": [222, 115, 129, 255, 216, 153, 0, 110, 36, 130, 6, 102, 209, 37, 143, 30],
    "List 8": [51, 146, 71, 27, 181, 183, 2, 122, 77, 184, 56, 225, 108, 83, 106, 15]
}}
</Example>

Input: {input}
Output: """

splitx = """<Instruction> Split the following list of numbers into 2 lists. The first list should contain the first half of the numbers and the second list should contain the second half.
Only output the 2 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [13, 16, 30, 6, 21, 7, 31, ...],
    "List 2": [25, 24, 10, 4, 27, 0, 14, ...]
}} </Instruction>

<Example>
Input: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Output:
{{
    "List 1": [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22],
    "List 2": [33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
}}
</Example>

Input: {input}
Output: """

def split(
    graph, 
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        num_elems = len(ast.literal_eval(graph_node["thought"]["set2"]))
        if num_elems == 32:
            prompt = split_prompt32
        elif num_elems == 64:
            prompt = split_prompt64
        elif num_elems == 128:
            prompt = split_prompt128
        else:
            prompt = splitx
        
        out = llm(
            prompt.format(
                input=graph_node["thought"]["set2"]
            ), 
            model=model
        )
        
        # Parse the result
        as_dict = ast.literal_eval(out[0].replace("\n", "").replace("Output:", ""))

        # 2. Update the graph
        for i in range(1, (num_elems // 16) + 1):
            idx = max(list(graph.nodes)) + 1
            graph.add_node(
                idx, 
                thought={
                    "set1": graph_node["thought"]["set1"],
                    "set2": str(as_dict[f"List {i}"]),
                }, 
                score=None
            )
            graph.add_edge(node_idx, idx)

    return graph, False

intersect_prompt = """<Instruction>Find the intersection of two sets of numbers. Output only the set of numbers that are present in both sets, no additional text. </Instruction>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Output: [24, 10, 20, 8]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Output: [40, 15, 5, 24, 35, 59, 16, 63]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Output: [115, 35, 90, 117, 63, 40, 30, 33, 15, 89, 50, 12, 2, 84, 7, 57, 96]
</Examples>

Input Set 1: {set1}
Input Set 2: {set2}
Output:"""

async def async_intersect(
    graph,
    nodes,
    model = "",
    multiplicity: int = 1,
):
    return await asyncio.gather(
        *[
            async_llm(
                intersect_prompt.format(
                    set1=graph.nodes[int(node)]["thought"]["set1"],
                    set2=graph.nodes[int(node)]["thought"]["set2"],
                ),
                model=model,
            ) for node in nodes
        ],
    )

def intersect(
    graph, 
    nodes,
    model = "",
    run_async = True,
    multiplicity: int = 1,
):
    # 1. Get LLM responses
    if run_async:
        outs = asyncio.run(async_intersect(graph, nodes, model=model, multiplicity=multiplicity))
        outs = {
            node: out[0] for node, out in zip(nodes, outs)
        }
    else:
        outs = {
            node: llm(
                intersect_prompt.format(
                    set1=graph.nodes[int(node)]["thought"]["set1"],
                    set2=graph.nodes[int(node)]["thought"]["set2"],
                ), 
                model=model
            )[0] for node in nodes
        }

    # 2. Update the graph
    intersected_nodes = []
    for node in nodes:
        node_idx = int(node)
        out = outs[node]

        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=out,
            set1=graph.nodes[int(node)]["thought"]["set1"],
            set2=graph.nodes[int(node)]["thought"]["set2"],
        )
        graph.add_edge(node_idx, idx)
        intersected_nodes.append(idx)

    # 3. Score all intersected nodes
    for node in intersected_nodes:
        graph, _ = score(graph, [node], model=model)

    return graph, False

def score(
    graph,
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        thought = graph_node["thought"]

        errors = 0
        feedback = {
            "missing_elements": 0,
            "extra_elements": 0,
            "duplicated_elements": 0,
        }
        try:
            # Extract set 1
            if isinstance(graph_node["set1"], list):
                set1 = graph_node["set1"]
            else:
                set1 = ast.literal_eval(graph_node["set1"])
            set1 = set(set1)

            # Extract set 2
            if isinstance(graph_node["set2"], list):
                set2 = graph_node["set2"]
            else:
                set2 = ast.literal_eval(graph_node["set2"])
            set2 = set(set2)
            
            # Parse the expression
            intersection = list(set1.intersection(set2))
            intersection.sort()

            if isinstance(thought, list):
                thought.sort()
            else:
                thought = ast.literal_eval(thought)
                thought.sort()

            # X_1: elements in C that aren't supposed to be there
            for i in thought:
                if i not in intersection:
                    errors += 1
                    feedback["extra_elements"] += 1

            # X_2: elements in C that are missing
            for i in intersection:
                if i not in thought:
                    errors += 1
                    feedback["missing_elements"] += 1

            # X_d: duplicated elements
            for i in set(thought):
                if thought.count(i) > 1:
                    errors += 1
                    feedback["duplicated_elements"] += 1
        except:
            # If the parsing fails, the LLM output is incorrectly formulated
            # so we assign a high error value
            errors = 1000000

        graph_node["score"] = errors
        if errors > 0:
            graph_node["feedback"] = feedback

    return graph, False

def keepbest(
    graph, 
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    # Score all non-scored nodes
    graph, _ = score(graph, nodes, model=model)
    return common_keepbest(graph, nodes, model=model)

refine_prompt = """<Instruction> The following three sets represent two sets and an intersection set of those two sets. The intersection set is not correct. Fix the intersection set so that it is correct.
Make sure that the numbers in the intersection set can be found in both input sets. Only output in the format following the examples, with no additional text.</Instruction>

<Approach>
To fix the incorrectly intersection set follow these steps:
1. Check for each number in the incorrect intersection set, whether it can be found in both input sets. If not, remove that number from the intersection set.
2. Iterate through the second input set and check whether each number is already in the incorrect intersection set and if not, check whether that number can also be found in the first input set. If so, add that number to the intersection set.
</Approach>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Incorrect Intersection Set: [24, 20, 25]
Reason: The incorrect intersection set contains the number 25, which is not present in the first input set and is missing the numbers 10 and 8.
Output: [24, 10, 20, 8]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Incorrect Intersection Set: [57, 16, 15, 24, 35, 10, 40]
Reason: The incorrect intersection set contains the numbers 57, which is not present in the second input set, and 10, which is not present in the first input set, and is missing the numbers 5, 63 and 59.
Output: [16, 15, 5, 24, 35, 63, 40, 59]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Incorrect Intersection Set: [35, 96, 44, 15, 33, 57, 30, 50, 90, 119, 123, 63, 117, 115, 2]
Reason: The incorrect intersection set contains the numbers 44 and 119, which are not present in the second input set, and 123, which is not present in the first input set, and is missing the numbers 89, 12, 84, 40 and 7.
Output: [35, 96, 15, 33, 57, 30, 89, 12, 84, 40, 7, 50, 90, 63, 117, 115, 2]
</Examples>

Input Set 1: {set1}
Input Set 2: {set2}
Incorrect Intersection Set: {incorrect_intersection}"""

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
                    set1=graph.nodes[int(node)]["set1"],
                    set2=graph.nodes[int(node)]["set2"],
                    incorrect_intersection=graph.nodes[int(node)]["thought"],
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
                "set1": graph_node["set1"],
                "set2": graph_node["set2"],
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
                    set1=graph.nodes[int(node)]["set1"],
                    set2=graph.nodes[int(node)]["set2"],
                    incorrect_intersection=graph.nodes[int(node)]["thought"],
                ),
                model=model,
            )[0] for node in nodes_to_refine
        }
    
    # 2. Update the graph
    nodes_to_score = []
    for node in nodes:
        if node in nodes_to_refine:
            output = outs[node][0]
            set1 = graph.nodes[int(node)]["set1"]
            set2 = graph.nodes[int(node)]["set2"]
        else:
            output = refined_nodes[node]["thought"]
            set1 = refined_nodes[node]["set1"]
            set2 = refined_nodes[node]["set2"]

        for k, v in PARSE_OUT_DICT.items():
            output = output.replace(k, v)

        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=output,
            set1 = set1,
            set2 = set2,
            score=None,
        )
        graph.add_edge(node_idx, idx)
        nodes_to_score.append(idx)

    # 3. Score all the refined nodes
    for node in nodes_to_score:
        graph, _ = score(graph, [node], model=model)

    return graph, False

aggregate_prompt = """<Instruction> Merge the following lists into one list by appending them together.
Only output the final list without any additional text or thoughts! </Instruction>

Lists: {inputs}
"""

async def async_aggregate(
    graph,
    nodes,
    model = "",
    multiplicity: int = 1,
):
    return await asyncio.gather(
        *[
            async_llm(
                aggregate_prompt.format(
                    inputs=[graph.nodes[int(node)]["thought"] for node in nodes],
                ),
                model=model
            ) for _ in range(multiplicity)
        ],
    )

def aggregate( 
    graph, 
    nodes,
    model = "",
    run_async: bool = True,
    multiplicity: int = 1,
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
                    inputs=[graph.nodes[int(node)]["thought"] for node in nodes],
                ),
                model=model
            )[0] for _ in range(multiplicity)
        ]
    
    # 2a. Find combined score
    if any(
        graph.nodes[int(node)].get("score", None) is None for node in nodes
    ):
        newscore = None
    else:
        newscore = sum(graph.nodes[int(node)]["score"] for node in nodes)

    # 2b. Find combined set2
    combined_set2 = []
    for node in nodes:
        if isinstance(graph.nodes[int(node)]["set2"], list):
            set2 = graph.nodes[int(node)]["set2"]
        else: 
            set2 = ast.literal_eval(graph.nodes[int(node)]["set2"])

        combined_set2 += set2

    # 3. Update the graph
    for out in outs:
        
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx, 
            thought=out,
            score=newscore,
            set1=graph.nodes[int(nodes[0])]["set1"],
            set2=combined_set2,
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
    # 1. Solve the original problem
    original = graph.nodes[0]["thought"]
    set1 = set(ast.literal_eval(original["set1"]))
    set2 = set(ast.literal_eval(original["set2"]))
    intersection = list(set1.intersection(set2))
    intersection = sorted(intersection)

    # 2. Set the "original" field to the original problem
    for node in nodes:
        graph.nodes[int(node)]["set1"] = graph.nodes[0]["thought"]["set1"]
        graph.nodes[int(node)]["set2"] = graph.nodes[0]["thought"]["set2"]

    graph, _ = score(
        graph,
        nodes,
        model
    )

    # 3. Check against ground truth
    any_match = False
    for node in nodes:
        node_idx = int(node)

        try:
            thought = graph.nodes[node_idx]["thought"]
            if isinstance(thought, list):
                pass
            else:
                thought = ast.literal_eval(thought)
            thought.sort()

            if graph.nodes[int(node)]["score"] == 0 and (intersection == thought):
                graph.nodes[node_idx]["matches_ground_truth"] = True
                any_match = True
            else:
                graph.nodes[node_idx]["matches_ground_truth"] = False
        except:
            graph.nodes[node_idx]["matches_ground_truth"] = False

    return graph, any_match

# Baselines

def io(
    graph, 
    nodes,
    model = "",
):
    return intersect(graph, nodes, model=model)

def _tot_schedule(
    width: int,
    depth: int,
) -> int:
    return _common_tot_schedule(
        width, 
        depth, 
        "intersect", 
        "refine"
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
        generate_action="intersect",
        generate_attempts=generate_attempts,
        aggregate_attempts=aggregate_attempts,
        post_aggregate_keepbest=post_aggregate_keepbest,
        post_aggregate_refine=post_aggregate_refine,
        refine_attempts=refine_attempts,
    )

cot_prompt = """<Instruction> Find the intersection of two sets of numbers. You can generate any intermediate solutions, but the final output should be the set of numbers that are present in both sets, prefixed with "Output: ". </Instruction>

<Approach>
To find the intersection of the two sets follow these steps:
1. Split the second input set of numbers into two to four subsets, each containing an equal number of elements from the original set (make sure they don't overlap).
2. For each subset find the set of numbers that are present in the subset and the first input set.
3. Merge the resulting sets into a single output set.
</Approach>

<Example>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Subsets of Input Set 2:
[25, 24, 10, 4, 27, 0, 14, 12]
[8, 2, 29, 20, 17, 19, 26, 23]
Intersected Subsets with Input Set 1:
[24, 10]
[8, 20]
Output: [24, 10, 8, 20]
</Example>

<Example>
Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Subsets of Input Set 2:
[16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53]
[58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Intersected Subsets with Input Set 1:
[16, 15, 5, 24]
[35, 63, 40, 59]
Output: [16, 15, 5, 24, 35, 63, 40, 59]
</Example>

Input Set 1: {set1}
Input Set 2: {set2}"""

def cot(
    graph, 
    nodes,
    model = "",
):
    for node in nodes:
        node_idx = int(node)

        out = llm(
            cot_prompt.format(
                set1=graph.nodes[int(node)]["thought"]["set1"],
                set2=graph.nodes[int(node)]["thought"]["set2"],
            ), model=model)[0]

        # Extract steps and answer
        output = out.split("Output:")[1]
        output = ast.literal_eval(output)
        
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=output,
            set1=graph.nodes[int(node)]["thought"]["set1"],
            set2=graph.nodes[int(node)]["thought"]["set2"],
        )
        graph.add_edge(node_idx, idx)

    return graph, False
