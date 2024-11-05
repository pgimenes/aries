from llm import llm
import ast
import re

model = "gpt-4"

problem_definition = "Sort a list of numbers in ascending order."

actions = {
    "split": "Split a sublist into two to decompose the problem.",
    "sort": " Sort a sublist.",
    "score": "Count the number of mistakes in the currently sorted sublist.",
    "refine": "Refine a sublist by fixing any existing mistakes. This action should only be called on nodes that have already been scored.",
    "keepbest": "Out of the selected nodes, keep the one with the highest score, and delete the rest. This action should only be called on nodes that have already been scored.",
    "aggregate": "Merge the sorted sublists of the selected nodes into a single sorted list. You can only aggregate two nodes at a time.",
    "groundtruth": "Compare the sorted list in a node with the ground truth. When a node doesn't match the ground truth, it will be marked with 'matches_ground_truth: False'."
}

# IO

_io_action_list = [
    "sort",
    "score",
    "groundtruth",
]

_io_node_list = [
    "0",
    "1",
    "1",
]

# CoT

_cot_action_list = [
    "sort_cot",
    "score",
    "groundtruth",
]
_cot_node_list = [
    "0",
    "1",
    "1",
]

# ToT

def _tot_schedule(
        width: int,
        depth: int,
) -> int:
    actions = []
    action_nodes = []
    keepbest_nodes = []
    last_node = 0

    # Sorting
    actions += ["sort"]
    action_nodes += [["0"] * width]
    last_node += width

    # Score
    score_nodes = [str(i) for i in range(1, width + 1)]
    actions += ["score"]
    action_nodes += [score_nodes]

    # Keep best
    actions += ["keepbest"]
    action_nodes += [[str(i) for i in range(1, width + 1)]]
    last_node += 1
    keepbest_nodes.append(str(last_node))

    for i in range(depth - 1):
        # Refine
        refine_node = last_node
        actions += ["refine"]
        action_nodes += [[str(refine_node)] * width]
        last_node += width

        # Score
        score_nodes = [str(j) for j in range(last_node - width + 1, last_node + 1)]
        actions += ["score"]
        action_nodes += [score_nodes]

        # Keep best
        actions += ["keepbest"]
        action_nodes += [[str(j) for j in range(last_node - width + 1, last_node + 1)]]
        last_node += 1
        keepbest_nodes.append(str(last_node))

    # Keep best
    actions += ["keepbest"]
    action_nodes += [keepbest_nodes]
    last_node += 1

    # Ground truth
    actions += ["groundtruth"]
    action_nodes += [[str(last_node)]]
    last_node += 1
    return actions, action_nodes

# GoT

def _got_schedule(    
    split_branches:int,
    sort_attempts:int,
) -> int:        
    # Create two split branches
    actions = ["split"]
    action_nodes = [["0"]]

    last_node = 2
    keepbest_nodes = []
    for split_branch in range(1, split_branches + 1):
        
        # Sorting
        sorted_nodes = []
        actions += ["sort"]
        action_nodes += [[str(split_branch)] * sort_attempts]
        sorted_nodes += list(range(last_node + 1, last_node + 1 + sort_attempts))
        last_node += sort_attempts

        # Scoring
        actions += ["score"]
        action_nodes += [sorted_nodes]

        # Keep best
        actions += ["keepbest"]
        action_nodes += [sorted_nodes]
        last_node += 1
        keepbest_nodes += [str(last_node)]

    # Aggregate
    actions += ["aggregate"]
    action_nodes += [keepbest_nodes]
    last_node += 1

    # Groundtruth
    actions += ["groundtruth"]
    action_nodes += [[str(last_node)]]
    last_node += 1
    return actions, action_nodes

split_prompt = """<Instruction> Split the following list of numbers into 2 lists, the first list should contain the first half of the numbers and the second list the second half of the numbers.
Only output the final 2 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, ...],
    "List 2": [2, 9, 2, 4, 7, 1, 5, ...]
}} </Instruction>

<Example>
Input: [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4, 5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
Output: 
{{
    List 1: [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4],
    List 2: [5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
}}
</Example>

Input: {input}"""

def split(
    graph, 
    nodes,
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(split_prompt.format(input=graph_node["thought"]), model=model)
        
        # Parse the result
        as_dict = ast.literal_eval(out[0].replace("\n", ""))

        # 2. Update the graph
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx, 
            thought=as_dict["List 1"], 
            score=None
        )
        graph.add_edge(node_idx, idx)

        idx = max(list(graph.nodes)) + 1
        graph.add_node(idx, thought=as_dict["List 2"], score=None)
        graph.add_edge(node_idx, idx)

    return graph, False

sort_prompt = """<Instruction> Sort the following list of numbers in ascending order. Output only the sorted list of numbers, no additional text. </Instruction>

<Examples>
Input: [5, 1, 0, 1, 2, 0, 4, 8, 1, 9, 5, 1, 3, 3, 9, 7]
Output: [0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 5, 5, 7, 8, 9, 9]

Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9, 4, 3, 5, 6, 6, 4, 4, 5, 2, 0, 9, 3, 3, 9, 2, 1]
Output: [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9]

Input: [4, 4, 9, 7, 9, 7, 0, 0, 4, 9, 1, 7, 9, 5, 8, 7, 5, 6, 3, 8, 6, 7, 5, 8, 5, 0, 6, 3, 7, 0, 5, 3, 7, 5, 2, 4, 4, 9, 0, 7, 8, 2, 7, 7, 7, 2, 1, 3, 9, 9, 7, 9, 6, 6, 4, 5, 4, 2, 0, 8, 9, 0, 2, 2]
Output: [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
</Examples>

Input: {input}"""

def sort(
    graph, 
    nodes,
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(sort_prompt.format(input=graph_node["thought"]), model=model)
        # out = [ast.literal_eval(graph_node["thought"])]
        
        # 2. Update the graph
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx, 
            thought=out[0], 
            score=None,
            original = graph_node["thought"],
        )
        graph.add_edge(node_idx, idx)

    return graph, False

sort_cot_prompt = """<Instruction> Sort the following list of numbers in ascending order. You can generate any intermediate lists, but the final output should be the sorted list of numbers, prefixed with "Output: ". </Instruction>

<Approach>
To sort the list of numbers follow these steps:
1. Split the list of numbers into two to four unsorted sublists, each containing an equal number of elements from the original list (make sure they don't overlap).
2. Sort each of the unsorted sublists.
3. Merge the sorted sublists into a single sorted list using the merging algorithm from merge sort.
</Approach>

<Examples>
Input: [4, 5, 3, 3, 7, 3, 0, 5, 0, 2, 8, 0, 2, 1, 6, 9]
Unsorted Subarrays:
[4, 5, 3, 3, 7, 3, 0, 5]
[0, 2, 8, 0, 2, 1, 6, 9]
Sorted Subarrays:
[0, 3, 3, 3, 4, 5, 5, 7]
[0, 0, 1, 2, 2, 6, 8, 9]
Output: [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 8, 9]

Input: [6, 4, 5, 7, 5, 6, 9, 7, 6, 9, 4, 6, 9, 8, 1, 9, 2, 4, 9, 0, 7, 6, 5, 6, 6, 2, 8, 3, 9, 5, 6, 1]
Unsorted Subarrays:
[6, 4, 5, 7, 5, 6, 9, 7, 6, 9, 4, 6, 9, 8, 1, 9]
[2, 4, 9, 0, 7, 6, 5, 6, 6, 2, 8, 3, 9, 5, 6, 1]
Sorted Subarrays:
[1, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 9, 9, 9]
[0, 1, 2, 2, 3, 4, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9]
Output: [0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9]

Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9, 4, 3, 5, 6, 6, 4, 4, 5, 2, 0, 9, 3, 3, 9, 2, 1, 9, 3, 1, 8, 1, 8, 6, 0, 1, 6, 1, 7, 4, 4, 6, 3, 3, 7, 9, 3, 6, 0, 3, 4, 5, 6, 6, 9, 9, 9, 7, 3]
Unsorted Subarrays:
[3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9]
[4, 3, 5, 6, 6, 4, 4, 5, 2, 0, 9, 3, 3, 9, 2, 1]
[9, 3, 1, 8, 1, 8, 6, 0, 1, 6, 1, 7, 4, 4, 6, 3]
[3, 7, 9, 3, 6, 0, 3, 4, 5, 6, 6, 9, 9, 9, 7, 3]
Sorted Subarrays:
[0, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9]
[0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 9, 9]
[0, 1, 1, 1, 1, 3, 3, 4, 4, 6, 6, 6, 7, 8, 8, 9]
[0, 3, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 9, 9, 9, 9]
Output: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]
</Examples>

Input: {input}"""

def sort_cot(
    graph, 
    nodes,
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(sort_cot_prompt.format(input=graph_node["thought"]), model=model)
        output = out[0].split("Output: ")[-1]
        
        # 2. Update the graph
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx, 
            thought=output, 
            score=None,
            original = graph_node["thought"],
        )
        graph.add_edge(node_idx, idx)

    return graph, False


refine_prompt = """<Instruction> The following two lists represent an unsorted list of numbers and a sorted variant of that list. The sorted variant is not correct. Fix the sorted variant so that it is correct.
Make sure that the output list is sorted in ascending order, has the same number of elements as the input list, and contains the same elements as the input list. Only output in the described format as in the example, with no additional text. </Instruction>

<Approach>
To fix the incorrectly sorted list follow these steps:
1. For each number from 0 to 9, compare the frequency of that number in the incorrectly sorted list to the frequency of that number in the input list.
2. Iterate through the incorrectly sorted list and add or remove numbers as needed to make the frequency of each number in the incorrectly sorted list match the frequency of that number in the input list.
</Approach>

Your output should be in the following format:
<example>
Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9]
Incorrectly Sorted: [0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 7, 7, 8, 8, 9, 9, 9, 9]
Reason: The incorrectly sorted list contains four extra 0s, two extra 4s and three extra 9s and is missing two 2s.
Output: [0, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9]
</example>

Input: {input}
Incorrectly Sorted: {incorrectly_sorted}
"""

def refine(
    graph, 
    nodes,
):
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        original = graph.nodes[0]["thought"]
        sorted = graph_node["thought"]

        prompt = refine_prompt.format(
            input=original,
            incorrectly_sorted=sorted
        )

        out = llm(prompt, model=model)

        # Find reason and output
        output = out[0].split("Output: ")[-1]
        # output = [original]

        # Update the graph
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=output,
            score=None,
            original=original,
        )
        graph.add_edge(node_idx, idx)

    return graph, False



def score(
    graph, 
    nodes,
):

    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        # Extract thought
        try:
            if isinstance(graph_node["thought"], list):
                thought = graph_node["thought"]
            else:
                thought = ast.literal_eval(graph_node["thought"])
        
            # Extract original for comparison
            if "original" in graph_node.keys():
                if isinstance(graph_node["original"], list):
                    original = graph_node["original"]
                else:
                    original = ast.literal_eval(graph_node["original"])
            else:
                original = ast.literal_eval(graph.nodes[0]["thought"])
            
            # Sorting errors term
            errors = 0
            for i in range(1, len(thought)):
                if thought[i] < thought[i - 1]:
                    errors += 1

            # Frequency difference term
            real_freq_dict = {}
            thought_freq_dict = {}
            for i in range(0, 10):
                real_freq_dict[i] = original.count(i)
                thought_freq_dict[i] = thought.count(i)
                diff = abs(real_freq_dict[i] - thought_freq_dict[i])
                errors += diff
        
        except:
            errors = 1000000

        graph.nodes[node_idx]["score"] = errors

    return graph, False

def get_parent_nodes(graph, node):
    parent_nodes = []
    for edge in graph.edges:
        if edge[1] == node:
            parent_nodes.append(edge[0])
    return parent_nodes

def keepbest(
    graph, 
    nodes,
):
    min_score = 1000000
    best_node_idx = nodes[0] # if all nodes have the same score, keep the first one
    
    # Node id for the new node
    # (decide before deleting nodes)
    new_idx = max(list(graph.nodes)) + 1
    
    if new_idx in graph.nodes:
        raise ValueError(f"new_idx {new_idx} already exists in graph")

    # Find node with highest score
    for node in nodes:
        graph_node = graph.nodes[int(node)]
        
        if graph_node["score"] < min_score:
            min_score = graph_node["score"]
            best_node_idx = node

    # Duplicate the best node
    added = False
    nodes_to_remove = []
    for _, node in enumerate(nodes):
        node_idx = int(node)
        
        if node == best_node_idx:
            added = True
            graph.add_node(
                new_idx, 
                thought=graph.nodes[int(best_node_idx)]["thought"], 
                score=min_score,
            )

            parent_node = get_parent_nodes(graph, node_idx)[0]
            graph.add_edge(parent_node, new_idx)
        
        # Flag node to remove
        nodes_to_remove.append(node_idx)

    if not added:
        breakpoint()

    # Remove the other nodes
    for node in nodes_to_remove:
        graph.remove_node(node)

    return graph, False


aggregate_prompt = """<Instruction> Merge the following 2 sorted lists of length X and Y into one sorted list of length X + Y using a merge sort style approach.
Only output the final merged list without any additional text or thoughts!:</Instruction>

<Approach>
To merge the two lists in a merge-sort style approach, follow these steps:
1. Compare the first element of both lists.
2. Append the smaller element to the merged list and move to the next element in the list from which the smaller element came.
3. Repeat steps 1 and 2 until one of the lists is empty.
4. Append the remaining elements of the non-empty list to the merged list.
</Approach>

Merge the following two lists into one sorted list:
1: {input1}
2: {input2}

Merged list:
"""

def aggregate(
    graph, 
    nodes,
):
    if len(nodes) != 2:
        raise ValueError("aggregate action requires exactly 2 nodes to be selected")
    
    # 1. Send the prompt
    out = llm(
        aggregate_prompt.format(
            input1=graph.nodes[int(nodes[0])]["thought"],
            input2=graph.nodes[int(nodes[1])]["thought"]
        ),
        model=model
    )
    
    # 2. Update the graph
    idx = max(list(graph.nodes)) + 1
    graph.add_node(idx, thought=out[0], score=None)

    for node in nodes:
        node_idx = int(node)
        graph.add_edge(node_idx, idx)

    return graph, False

def groundtruth(
    graph, 
    nodes,
):
    problem = graph.nodes[0]["thought"]
    sorted_problem = ast.literal_eval(problem)
    sorted_problem.sort()

    any_match = False
    for node in nodes:
        node_idx = int(node)
        thought = graph.nodes[node_idx]["thought"]
        
        try:
            if isinstance(thought, list):
                thought = thought
            else:
                thought = ast.literal_eval(thought)
            
            if thought == sorted_problem:
                graph.nodes[node_idx]["matches_ground_truth"] = True
                any_match = True
            else:
                graph.nodes[node_idx]["matches_ground_truth"] = False
        except:
            graph.nodes[node_idx]["matches_ground_truth"] = False
            pass


        
    return graph, any_match