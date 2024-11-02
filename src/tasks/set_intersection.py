from llm import llm
import ast
import pandas as pd

model = "gpt-4"

problem_definition = "Count the frequency of how many times each country is explicitly named in the input text."

# IO

io_prompt = """<Instruction> Find the intersection of two sets of numbers. Output only the set of numbers that are present in both sets, no additional text. </Instruction>

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

_io_action_list = [
    "io",
    "groundtruth",
]

_io_node_list = [
    "0",
    "1",
]

def io(
    graph, 
    nodes,
):
    for node in nodes:
        out = llm(
            io_prompt.format(
                set1=graph.nodes[int(node)]["thought"]["set1"],
                set2=graph.nodes[int(node)]["thought"]["set2"],
            ), 
            model=model
        )[0]

        graph.add_node(
            1,
            thought=out,
        )
        graph.add_edge(0, 1)

    return graph, False
        

cot_prompt = """<Instruction> Find the intersection of two sets of numbers. You can generate any intermediate solutions, but the final output should be the set of numbers that are present in both sets, prefixed with "Output: ". </Instruction>

<Approach>
To find the intersection of the two sets follow these steps:
1. Split the second input set of numbers into two to four subsets, each containing an equal number of elements from the original set (make sure they don't overlap).
2. For each subset find the set of numbers that are present in the subset and the first input set.
3. Merge the resulting sets into a single output set.
</Approach>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Subsets of Input Set 2:
[25, 24, 10, 4, 27, 0, 14, 12]
[8, 2, 29, 20, 17, 19, 26, 23]
Intersected Subsets with Input Set 1:
[24, 10]
[8, 20]
Output: [24, 10, 8, 20]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Subsets of Input Set 2:
[16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53]
[58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Intersected Subsets with Input Set 1:
[16, 15, 5, 24]
[35, 63, 40, 59]
Output: [16, 15, 5, 24, 35, 63, 40, 59]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Subsets of Input Set 2:
[13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122]
[85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12]
[65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81]
[124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Intersected Subsets with Input Set 1:
[35, 96]
[15, 33, 57, 30, 89, 12]
[84, 40, 7, 50, 90]
[63, 117, 115, 2]
Output: [35, 96, 15, 33, 57, 30, 89, 12, 84, 40, 7, 50, 90, 63, 117, 115, 2]
</Examples>

Input Set 1: {set1}
Input Set 2: {set2}"""


_cot_action_list = [
    "cot",
    "groundtruth",
]
_cot_node_list = [
    "0",
    "1",
]

def cot(
    graph, 
    nodes,
):
    for node in nodes:
        out = llm(
            cot_prompt.format(
                set1=graph.nodes[int(node)]["thought"]["set1"],
                set2=graph.nodes[int(node)]["thought"]["set2"],
            ), model=model)[0]

        # Extract steps and answer
        output = out.split("Output: ")[1]
        output = ast.literal_eval(output)
        graph.add_node(
            1,
            thought=output,
        )
        graph.add_edge(0, 1)

    return graph, False

actions = {
    "propose": "",
    "score": "",
    "keepbestn": "",
    "validate": "",
    "groundtruth": "",
}

propose_prompt = """<Instruction> Use numbers and basic arithmetic operations (+ - * /) to obtain 24. </Instruction>

<Example>
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
</Example>

Input: {input}"""

def propose(
    graph, 
    nodes,
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(propose_prompt.format(input=graph_node["left"]), model=model)
        
        # Parse the result
        pass

    return graph, False


score_prompt = """<Instruction>Evaluate if given numbers can reach 24 (sure/likely/impossible)</Instruction>

<Example>
Input: 11 12
Attempts:
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
Output: impossible
</Example>

<Example>
Input: 4 4 10
Attempts:
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
Output: sure
</Example>

<Example>
Input: 5 7 8
Attempts: 
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
Output: likely
</Example>

Input: {input}
"""

def score(
    graph, 
    nodes,
):
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        pass

    return graph, False

def get_parent_nodes(graph, node):
    parent_nodes = []
    for edge in graph.edges:
        if edge[1] == node:
            parent_nodes.append(edge[0])
    return parent_nodes

def keepbestn(
    graph, 
    nodes,
):
    min_score = 1000000
    best_node_idx = None
    
    # Node id for the new node
    # (decide before deleting nodes)
    new_idx = max(list(graph.nodes)) + 1

    # Find node with highest score
    for idx, node in enumerate(nodes):
        graph_node = graph.nodes[int(node)]
        
        if graph_node["score"] < min_score:
            min_score = graph_node["score"]
            best_node_idx = node

    # Delete all other nodes
    for _, node in enumerate(nodes):
        node_idx = int(node)
        
        # Duplicate the best node
        if node == best_node_idx:
            graph.add_node(
                new_idx, 
                thought=graph.nodes[int(best_node_idx)]["thought"], 
                score=min_score,
            )
            graph.add_edge(get_parent_nodes(graph, node_idx)[0], new_idx)
        
        graph.remove_node(node_idx)

    return graph, False

validate_prompt = """<Instruction>Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.</Instruction>

<Example>
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure
</Example>

<Example>
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure
</Example>

<Example>
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure
</Example>

<Example>
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
</Example>

<Example>
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
</Example>

<Example>
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
</Example>

Input: {input}
Answer: {answer}
Judge:"""

def validate(
    graph, 
    nodes,
):
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(validate_prompt.format(input=graph_node["left"], answer=graph_node["thought"]), model=model)
        
        pass

    return graph, False

def groundtruth(
    graph, 
    nodes,
):
    original = graph.nodes[0]["thought"]

    any_match = False
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        # Parse the expression
        set1 = set(ast.literal_eval(original["set1"]))
        set2 = set(ast.literal_eval(original["set2"]))
        intersection = list(set1.intersection(set2))
        intersection.sort()

        if isinstance(graph_node["thought"], list):
            thought = graph_node["thought"]
        else:
            thought = ast.literal_eval()
        thought.sort()

        if intersection == thought:
            any_match = True

    return graph, any_match