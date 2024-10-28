from llm import llm
import ast
import re

model = "gpt-4"

problem_definition = "Sort a list of numbers in ascending order."

actions = {
    "split": "Split a sublist into two to decompose the problem.",
    "sort": " Sort a sublist.",
    "refine": "Refine a sublist by fixing any existing mistakes. This action should only be called on nodes that have already been scored.",
    "score": "Count the number of mistakes in the currently sorted sublist.",
    "keepbest": "Out of the selected nodes, keep the one with the highest score, and delete the rest. This action should only be called on nodes that have already been scored.",
    "aggregate": "Merge the sorted sublists of the selected nodes into a single sorted list. You can only aggregate two nodes at a time.",
    "groundtruth": "Compare the sorted list in a node with the ground truth. When a node doesn't match the ground truth, it will be marked with 'matches_ground_truth: False'."
}

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
        idx = graph.number_of_nodes()
        graph.add_node(
            idx, 
            thought=as_dict["List 1"], 
            score=None
        )
        graph.add_edge(node_idx, idx)

        idx = graph.number_of_nodes()
        graph.add_node(graph.number_of_nodes(), thought=as_dict["List 2"], score=None)
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
        
        # 2. Update the graph
        idx = graph.number_of_nodes()
        graph.add_node(
            idx, 
            thought=out[0], 
            score=None,
            original = graph_node["thought"],
        )
        graph.add_edge(node_idx, idx)

    return graph, False

refine_prompt = """<Instruction> The following two lists represent an unsorted list of numbers and a sorted variant of that list. The sorted variant is not correct. Fix the sorted variant so that it is correct.
Make sure that the output list is sorted in ascending order, has the same number of elements as the input list, and contains the same elements as the input list. </Instruction>

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

        # Update the graph
        idx = graph.number_of_nodes()
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
        
        try:
            if type(graph_node["thought"]) == list:
                thought = graph_node["thought"]
            else:
                thought = ast.literal_eval(graph_node["thought"])
        except:
            raise ValueError("score action requires the original and sorted lists to be in a list format")

        # Count elementwise errors
        errors = 0
        for i in range(1, len(thought)):
            if thought[i] < thought[i - 1]:
                errors += 1

        graph.nodes[node_idx]["score"] = errors

    return graph, False

def keepbest(
    graph, 
    nodes,
):
    max_score = 0
    best_node_idx = 0
    
    # Find node with highest score
    for idx, node in enumerate(nodes):
        graph_node = graph.nodes[int(node)]
        
        if graph_node["score"] > max_score:
            max_score = graph_node["score"]
            best_node_idx = idx

    # Delete all other nodes
    for idx, node in enumerate(nodes):
        if idx != best_node_idx:
            node_idx = int(node)
            graph.remove_node(node_idx)

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
    idx = graph.number_of_nodes()
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

    for node in nodes:
        node_idx = int(node)
        thought = graph.nodes[node_idx]["thought"]
        thought = ast.literal_eval(thought)

        if thought == sorted_problem:
            graph.nodes[node_idx]["matches_ground_truth"] = True
            return graph, True
        else:
            graph.nodes[node_idx]["matches_ground_truth"] = False
        
    return graph, False