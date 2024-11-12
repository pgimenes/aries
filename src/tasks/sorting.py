from llm import llm
import ast
from .common import (
    _common_tot_schedule, 
    _common_got_schedule, 
    common_keepbest, 
)

problem_definition = "Sort a list of numbers in ascending order."

actions = {
    "split": {
        "description": "Split a sublist into two to decompose the problem.",
        "preconditions": "",
        "effects": "Two new nodes are created, each containing a sublist of the original list.",
    },
    "sort": {
        "description": "Sort a list or sublist. You should attempt sorting any node up to 10 times.",
        "preconditions": "",
        "effects": "A new node is created with the sorted sublist, connected to the original node.",
    },
    "aggregate": {
        "description": "Merge the sorted sublists of the selected nodes into a single sorted list.",
        "preconditions": "Only two nodes must be selected.",
        "effects": "A new node is created with the merged sorted list, connected to the two selected nodes.",
    },
    "refine": {
        "description": "Refine the sorting of a list or sublist.",
        "preconditions": "The selected node must have been scored.",
        "effects": "A new node is created with a refined sorting of the selected node, connected to the selected node.",
    },
    "score": {
        "description": "Count the number of mistakes in the node.",
        "preconditions": "",
        "effects": "The node is annotated with a score, which is the number of mistakes. The node may also be annotated with a feedback dictionary. The missing_elements key indicates the number of elements that are missing from the sorted list. The extra_elements key indicates the number of elements that are in the sorted list but not in the original list.",
    },
    "keepbest": {
        "description": "Out of the selected nodes, keep the one with the highest score, and delete the rest.",
        "preconditions": "The selected nodes must have been scored.",
        "effects": "All selected nodes are deleted, but the one with the highest score is duplicated as a new node.",
    },
    "groundtruth": {
        "description": "Compare a node to the ground truth sorting of node 0.",
        "preconditions": "",
        "effects": "The node is annotated with 'matches_ground_truth: True' or 'False'.",
    }
}

example_list = [
    # Sort directly
    """<example>
INPUT:
Previous actions:

Current graph:

Nodes:
0: {'thought': '[0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1, 3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]'}

Edges:

OUTPUT:
Analysis: 

A. Action history: No actions have been taken yet. 

B. Graph state: The graph currently has 1 node and 0 edges. Node 0 contains the initial problem. 

C. Strategy analysis: The strategy for solving the problem has not been determined yet.

D. Next action options
    1. Attempt to sort the entire list directly without decomposing it. This may work if the list is already mostly sorted.

    2. Decompose the list into smaller sublists, sort them individually, then merge the results to produce a solution. This may be more effective if the list is not already mostly sorted.

Next action: sort
Nodes: [0]

Explanation: The list in node 0 appears easy to sort since many elements are already in order, so we can attempt to sort the entire list directly, then score the sort the sorting attempt. If there are many mistakes, we can decompose the list into two smaller sublists, sort them individually, then merge the results to produce a solution.
</example>""",
    
    # Decompose, sort, merge strategy
    """<example>
INPUT:
Previous actions:

Step 1: split
Nodes: [0]
Explanation: To solve this problem, we should decompose node 0 into smaller sublists and sort them individually, since sorting smaller sublists is less likely to introduce errors. The strategy will be to decompose into two lists, sort them individually, then score each sorted sublist to identify whether any mistakes were introduced. Once we find successfully sorted sublists, we can start merging them to find the sorted version of node 0. 

Step 2: sort
Nodes: [1, 2]
Explanation: We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. The next step is to sort each of these sublists individually. We will start by attempting to sort each sublist once. Next, we will score each sorting attempt to identify any errors. If any errors are found, we may need to perform additional sorting attempts until a correct sorting is achieved.

Step 3: score
Nodes: [3, 4]
Explanation: We are continuing the strategy outlined in step 1. Nodes 3 and 4 are sorted versions of nodes 1 and 2, which are sublists of node 0. The nodes are marked with a score of None, so the next step is to check if the sublists were correctly sorted. If any mistakes are found, we may need to perform additional sorting attempts until a correct sorting is achieved. Otherwise, we can merge nodes 3 and 4 to find the sorted version of node 0.

Step 4: aggregate
Nodes: [3, 4]
Explanation: We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. Nodes 3 and 4 are sorted correctly sorted versions of nodes 1 and 2, so we can now aggregate them to find the sorted version of node 0. We will start by performing a single aggregation attempt.If any mistakes are found, we may need to perform additional aggregation attempts until a correct solution is achieved.

Current graph:

Nodes:
0: {'thought': '[0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1, 3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]'}
1: {'thought': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1], 'score': None}
2: {'thought': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9], 'score': None}
3: {'thought': '[0, 0, 0, 0, 1, 1, 1, 1, 2, 5, 6, 7, 9, 9, 9, 9]', 'score': 0, 'original': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1]}
4: {'thought': '[0, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 9, 9]', 'score': 0, 'original': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]}
5: {'thought': '[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 9, 9, 9, 9, 9, 9]', 'score': None}

Edges:
(0, 1): {}
(0, 2): {}
(1, 3): {}
(2, 4): {}
(3, 5): {}
(4, 5): {}

OUTPUT:
Analysis: 
A. Action history: The current strategy is to decompose the initial list into two smaller sublists, sort them individually, then merge them to produce a solution to the original problem. In step 1, the list was decomposed into two sublists. In step 2, the sublists were sorted. In step 3, the sorted sublists were scored, showing that both sublists were correctly sorted. In step 4, the correctly sorted sublists were aggregated. 

B. Graph state: The graph currently has 6 nodes and 6 edges. Node 0 has the initial problem. Nodes 1 and 2 represent sublists of the original list, obtained in step 1. Nodes 3 and 4 were obtained in step 2, and they represent attempted sortings of the sublists in nodes 1 and 2. Nodes 3 and 4 both have a score of 0, marked in step 3. Node 5 is the result of aggregating nodes 3 and 4, obtained in step 4. The score of node 5 is None, indicating it has not been scored yet. 

C. Strategy analysis: The strategy so far is partially successful, since the sublists in nodes 1 and 2 were correctly sorted into nodes 3 and 4. However, the strategy is still pending since we don't know yet if the aggregation was successful.

D. Next action options
    1. Directly compare the current aggregation attempt to the ground truth. Scoring is not necessary if the solution is correct, however if it is incorrect we will have no information on the number of mistakes.

    2. Score node 5 to check if the aggregation was successful. If the sorted list in node 5 matches the ground truth, the score will be 0, and we can then call the groundtruth action to finish the problem.

Next action: groundtruth
Nodes: [5]

Explanation: We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. Nodes 3 and 4 are sorted sublists derived from nodes 1 and 2, respectively. Their score is 0, meaning their sorting is correct. Node 5 is the result of aggregating nodes 3 and 4. We should compare the sorted list in node 5 with the ground truth to determine if the aggregation was successful. If the sorted list in node 5 matches the ground truth, we have successfully solved the problem. Otherwise, we may need to perform additional aggregation attempts until a correct solution is achieved.
</example>""",

    # Aggregation error, retry 10 times, periodically scoring
    """<example>
INPUT:
Previous actions:

Step 1: split
Nodes: [0]
Explanation: To solve this problem, we should decompose the full list into smaller sublists and sort them individually, since sorting smaller sublists is less likely to introduce errors. The strategy will be to decompose into two lists, sort them individually, then score each sorted sublist to identify whether any mistakes were introduced. Once we find successfully sorted sublists, we can start merging them to produce a solution to the final problem. 

Step 2: sort
Nodes: [1, 2]
Explanation:  We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. The next step is to sort each of these sublists individually. We will start by attempting to sort each sublist once. Next, we will score each sorting attempt to identify any errors. If any errors are found, we may need to perform additional sorting attempts until a correct sorting is achieved.

Step 3: score
Nodes: [3, 4]
Explanation:  We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. Nodes 3 and 4 are sorted sublists derived from nodes 1 and 2, but they haven't been scored yet. If any mistakes are found, we may need to perform additional sorting attempts until a correct sorting is achieved. Ultimately, after finding correctly sorted sublists, we can start merging them to produce a solution to the original problem.

Step 4: aggregate
Nodes: [3, 4]
Explanation:  We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. Nodes 3 and 4 are correctly sorted sublists derived from nodes 1 and 2. We can now aggregate them to find the solution to sorting node 0. We will perform 10 aggregation attempts. Each aggregation attempt will be scored to identify any errors, and if any mistakes are found, we may need to perform additional aggregation attempts until a correct solution is achieved.

Step 5: score
Nodes: [3, 4]
Explanation:  We are continuing the strategy outlined in step 1. Node 0 was split into nodes 1 and 2. Nodes 3 and 4 are correctly sorted sublists derived from nodes 1 and 2. They were aggregated in node 5, and we will now check if the aggregation was correct.

Step 6: aggregate
Nodes: [3, 4]
Explanation: The aggregation of nodes 3 and 4 was not performed correctly and the error count is high, so we will perform another 10 attempts. We will score each attempt to identify any errors, and if any mistakes are found, we may need to attempt refining the best aggregation attempts.

Step 7: aggregate
Nodes: [3, 4]
Explanation: As explained in step 6, we are reattempting the aggregation due to errors in node 5. We have so far performed 1/10 aggregation attempts.

Current graph:

Nodes:
0: {'thought': '[0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1, 3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]'}
1: {'thought': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1], 'score': None}
2: {'thought': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9], 'score': None}
3: {'thought': '[0, 0, 0, 0, 1, 1, 1, 1, 2, 5, 6, 7, 9, 9, 9, 9]', 'score': 0, 'original': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1]}
4: {'thought': '[0, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 9, 9]', 'score': 0, 'original': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]}
5: {'thought': '[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 9, 9, 9, 9]', 'score': 3}
6: {'thought': '[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 9, 9, 9, 9]', 'score': 4}
7: {'thought': '[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 9, 9, 9, 9]', 'score': 3}

Edges:
(0, 1): {}
(0, 2): {}
(1, 3): {}
(2, 4): {}
(3, 5): {}
(4, 5): {}
(3, 6): {}
(4, 6): {}
(3, 7): {}
(4, 7): {}

OUTPUT:
Analysis: 
A. Action history: the initial strategy was to decompose the initial list into two smaller sublists, sort them individually, then merge them to produce a solution to the original problem. In step 1, the list was decomposed into two sublists. In step 2, the sublists were sorted. In step 3, the sorted sublists were scored, showing that both sublists were correctly sorted. In step 4, the correctly sorted sublists were aggregated. In step 5, the aggregation was scored, showing 3 mistakes. In steps 6 and 7, another 2 aggregation attempts were performed.

B. Graph state: the graph currently has 11 nodes and 16 edges. Node 0 has the initial problem. Nodes 1 and 2 represent sublists of the original list, obtained in step 1. Nodes 3 and 4 were obtained in step 2, and they represent attempted sortings of the sublists in nodes 1 and 2. Nodes 3 and 4 both have a score of 0, marked in step 3. Nodes 5, 6, 7 are attempts at aggregating nodes 3 and 4. They all have aggregation errors.

C. Strategy analysis: the current strategy is unsuccessful, as we have not yet found a correct aggregation of nodes 3 and 4. However, there are still 8 aggregation attempts pending, so we may find a correct solution in the next steps.

D. Next action options
    1. Keep the best aggregation attempt out of nodes 5 to 10 and delete the rest. This would then allow refining the best aggregation attempt.

    2. Continue aggregating nodes 3 and 4. If no correct solution is found, switch to refining the best aggregation attempt.

Next action: aggregate
Nodes: [3, 4]
Explanation: As explained in step 6, we are reattempting the aggregation due to errors in node 5. We have so far performed 2/10 aggregation attempts.
</example>""",

    # Sorting error, split further
    """<example>
INPUT:
Previous actions:

Step 1: split
Nodes: [0]
Explanation: To solve this problem, we should decompose node 0 into smaller sublists and sort them individually, since sorting smaller sublists is less likely to introduce errors. The strategy will be to decompose into two lists, sort them individually, then score each sorted sublist to identify whether any mistakes were introduced. Once we find successfully sorted sublists, we can start merging them to find the sorted version of node 0. 

Step 2: sort
Nodes: [1, 2]
Explanation: We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. The next step is to sort each of these sublists individually. We will start by attempting to sort each sublist once. Next, we will score each sorting attempt to identify any errors. If any errors are found, we may need to perform additional sorting attempts until a correct sorting is achieved.

Step 3: score
Nodes: [3, 4]
Explanation: We are continuing the strategy outlined in step 1. Nodes 3 and 4 are sorted versions of nodes 1 and 2, which are sublists of node 0. The nodes are marked with a score of None, so the next step is to check if the sublists were correctly sorted. If any mistakes are found, we may need to perform additional sorting attempts until a correct sorting is achieved. Otherwise, we can merge nodes 3 and 4 to find the sorted version of node 0.

Step 4: split
Nodes: [2]
Explanation: We have found a mistake, so we are adapting the current strategy. The sublist in node 2 was incorrectly sorted, as indicated by the score of 1. To correct this mistake, we will decompose the problem further by splitting the sublist in node 2 into two smaller sublists. We will leave the sublist in node 1 as it is already correctly sorted in node 3. Once the new sublists are correctly sorted, we can merge them to correct the error in node 4. Then, we can continue with the strategy of merging the sorted sublists to produce a solution to the original problem.

Step 5: sort
Nodes: [5, 6]
Explanation: We are continuing with the new strategy outlined in step 4. We split the sublist in node 2 into two smaller sublists in nodes 5 and 6, since sublist 2 was incorrectly sorted in node 4. The next step is to sort each of these sublists individually. We will start by attempting to sort each sublist once. Next, we will score each sorting attempt to identify any errors. If any errors are found, we may need to perform additional sorting attempts until a correct sorting is achieved.

Current graph:

Nodes:
0: {'thought': '[0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1, 3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]'}
1: {'thought': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1], 'score': None}
2: {'thought': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9], 'score': None}
3: {'thought': '[0, 0, 0, 0, 1, 1, 1, 1, 2, 5, 6, 7, 9, 9, 9, 9]', 'score': 0, 'original': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1]}
4: {'thought': '[0, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 9]', 'score': 1, 'original': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]}
5: {'thought': [3, 5, 2, 3, 5, 6, 0, 2], 'score': None}
6: {'thought': [7, 4, 6, 2, 9, 7, 5, 9], 'score': None}

Edges:
(0, 1): {}
(0, 2): {}
(1, 3): {}
(2, 4): {}
(2, 5): {}
(2, 6): {}

OUTPUT:
Analysis: 
A. Action history:
The starting strategy was to decompose node 0 into two smaller sublists, sort them individually, then merge them to produce a solution to the original problem. In step 1, the list was decomposed into two sublists. In step 2, the sublists were sorted. In step 3, the sorted sublists were scored, showing there was a mistake in node 4. Therefore the strategy was changed to decompose node 2 further into smaller sublists and sort them individually before merging to correct the mistake in node 4. Node 2 was split in step 4, and the new sorting attempt was made in step 5.

B. Graph state:
The graph currently has 7 nodes and 6 edges. Node 0 has the initial problem. Nodes 1 and 2 represent sublists of the original list, obtained in step 1. Nodes 3 and 4 were obtained in step 2, and they represent attempted sortings of the sublists in nodes 1 and 2. Node 3 has a score of 0, marked in step 3. Node 4 has a score of 1, marked in step 3. Nodes 5 and 6 are the new sublists obtained by splitting node 2 in step 4. 

C. Strategy analysis:
The current strategy is pending, as we need to identify whether the new sorting attempts in nodes 5 and 6 are successful.

D. Next action options:
    1. Score nodes 5 and 6 to check if the new sorting attempts were successful. If the sorting is correct, we can proceed with aggregating the sublists to produce a corrected version of node 4. If any mistakes are found, we may need to perform additional sorting attempts until a correct sorting is achieved.

    2. Merge the sorted sublists in nodes 5 and 6 to produce a corrected version of node 4. If the sorting is correct, we can proceed with aggregating the sublists to produce a corrected version of node 4. However, the merging will likely be incorrect if either of the sorted sublists have errors in them.

Next action: score
Nodes: [5, 6]

Explanation: We are continuing with the new strategy outlined in step 4. We split the sublist in node 2 into two smaller sublists in nodes 5 and 6, since sublist 2 was incorrectly sorted in node 4. The new sublists were sorted, and we need to score them to check if the sorting was successful. If the sorting is correct, we can proceed with aggregating the sublists to produce a corrected version of node 4. If any mistakes are found, we may need to perform additional sorting attempts until a correct sorting is achieved. Otherwise, we can continue with the strategy of merging the sorted sublists to produce a sorted version of node 0.
</example>""",
]

minimal_examples = [
    """<example>
INPUT:
Previous actions:

Step 1: split
Nodes: [0]
Explanation: To solve this problem, we should decompose node 0 into smaller sublists and sort them individually, since sorting smaller sublists is less likely to introduce errors. The strategy will be to decompose into two lists, sort them individually, then score each sorted sublist to identify whether any mistakes were introduced. Once we find successfully sorted sublists, we can start merging them to find the sorted version of node 0. 

Step 2: sort
Nodes: [1, 2]
Explanation: We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. The next step is to sort each of these sublists individually. We will start by attempting to sort each sublist once. Next, we will score each sorting attempt to identify any errors. If any errors are found, we may need to perform additional sorting attempts until a correct sorting is achieved.

Step 3: score
Nodes: [3, 4]
Explanation: We are continuing the strategy outlined in step 1. Nodes 3 and 4 are sorted versions of nodes 1 and 2, which are sublists of node 0. The nodes are marked with a score of None, so the next step is to check if the sublists were correctly sorted. If any mistakes are found, we may need to perform additional sorting attempts until a correct sorting is achieved. Otherwise, we can merge nodes 3 and 4 to find the sorted version of node 0.

Step 4: aggregate
Nodes: [3, 4]
Explanation: We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. Nodes 3 and 4 are sorted correctly sorted versions of nodes 1 and 2, so we can now aggregate them to find the sorted version of node 0. We will start by performing a single aggregation attempt.If any mistakes are found, we may need to perform additional aggregation attempts until a correct solution is achieved.

Current graph:

Nodes:
0: {'thought': '[0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1, 3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]'}
1: {'thought': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1], 'score': None}
2: {'thought': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9], 'score': None}
3: {'thought': '[0, 0, 0, 0, 1, 1, 1, 1, 2, 5, 6, 7, 9, 9, 9, 9]', 'score': 0, 'original': [0, 0, 5, 9, 0, 7, 9, 9, 1, 2, 6, 1, 1, 9, 0, 1]}
4: {'thought': '[0, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 9, 9]', 'score': 0, 'original': [3, 5, 2, 3, 5, 6, 0, 2, 7, 4, 6, 2, 9, 7, 5, 9]}
5: {'thought': '[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 9, 9, 9, 9, 9, 9]', 'score': None}

Edges:
(0, 1): {}
(0, 2): {}
(1, 3): {}
(2, 4): {}
(3, 5): {}
(4, 5): {}

OUTPUT:

<analysis>
A. Action history: The current strategy is to decompose the initial list into two smaller sublists, sort them individually, then merge them to produce a solution to the original problem. In step 1, the list was decomposed into two sublists. In step 2, the sublists were sorted. In step 3, the sorted sublists were scored, showing that both sublists were correctly sorted. In step 4, the correctly sorted sublists were aggregated. 

B. Graph state: The graph currently has 6 nodes and 6 edges. Node 0 has the initial problem. Nodes 1 and 2 represent sublists of the original list, obtained in step 1. Nodes 3 and 4 were obtained in step 2, and they represent attempted sortings of the sublists in nodes 1 and 2. Nodes 3 and 4 both have a score of 0, marked in step 3. Node 5 is the result of aggregating nodes 3 and 4, obtained in step 4. The score of node 5 is None, indicating it has not been scored yet. 

C. Strategy analysis: The strategy so far is partially successful, since the sublists in nodes 1 and 2 were correctly sorted into nodes 3 and 4. However, the strategy is still pending since we don't know yet if the aggregation was successful.

D. Next action options
    1. Directly compare the current aggregation attempt to the ground truth. Scoring is not necessary if the solution is correct, however if it is incorrect we will have no information on the number of mistakes.

    2. Score node 5 to check if the aggregation was successful. If the sorted list in node 5 matches the ground truth, the score will be 0, and we can then call the groundtruth action to finish the problem.
</analysis>

<next_action>
groundtruth
</next_action>

<nodes>
[5]
</nodes>

<explanation>
We are continuing the strategy outlined in step 1. Currently, the list in node 0 has been split into two sublists in nodes 1 and 2. Nodes 3 and 4 are sorted sublists derived from nodes 1 and 2, respectively. Their score is 0, meaning their sorting is correct. Node 5 is the result of aggregating nodes 3 and 4. We should compare the sorted list in node 5 with the ground truth to determine if the aggregation was successful. If the sorted list in node 5 matches the ground truth, we have successfully solved the problem. Otherwise, we may need to perform additional aggregation attempts until a correct solution is achieved.
</explanation>

</example>""",
]

examples = "\n".join(minimal_examples)

additional_instructions = """- Only two nodes can be aggregated at a time, so if there are more than two nodes to aggregate, the aggregation must be done following a tree reduction strategy. E.g. if there are 4 nodes to aggregate, aggregate nodes 1 and 2, then aggregate nodes 3 and 4, and finally aggregate the results of the previous aggregations."""

PARSE_OUT_DICT = {
    "Output:": "",
    "json": "",
    "`": "",
    "\n": "",
}

# Implementation

splitx = """<Instruction> Split the following list of numbers into 2 lists, the first list should contain the first half of the numbers and the second list the second half of the numbers.
Only output the final 2 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, ...],
    "List 2": [2, 9, 2, 4, 7, 1, 5, ...]
}} </Instruction>

<Example>
Input: [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4, 5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
Output: 
{{
    "List 1": [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4],
    "List 2": [5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
}}
</Example>

Input: {input}"""

split32 = """<Instruction> Split the following list of 32 numbers into 2 lists of 16 numbers each, the first list should contain the first 16 numbers and the second list the second 16 numbers.
Only output the final 2 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, ...],
    "List 2": [2, 9, 2, 4, 7, 1, 5, ...]
}} </Instruction>

<Example>
Input: [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4, 5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
Output: 
{{
    "List 1": [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4],
    "List 2": [5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
}}
</Example>

Input: {input}"""

split64 = """<Instruction> Split the following list of 64 numbers into 4 lists of 16 numbers each, the first list should contain the first 16 numbers, the second list the second 16 numbers, the third list the third 16 numbers and the fourth list the fourth 16 numbers.
Only output the final 4 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, ...],
    "List 2": [2, 9, 2, 4, 7, 1, 5, ...],
    "List 3": [6, 9, 8, 1, 9, 2, 4, ...],
    "List 4": [9, 0, 7, 6, 5, 6, 6, ...]
}} </Instruction>

<Example>
Input: [3, 1, 9, 3, 7, 5, 5, 4, 8, 1, 5, 3, 3, 2, 3, 0, 9, 7, 2, 2, 4, 4, 8, 5, 0, 8, 7, 3, 3, 8, 7, 0, 9, 5, 1, 6, 7, 6, 8, 9, 0, 3, 0, 6, 3, 4, 8, 0, 6, 9, 8, 4, 1, 2, 9, 0, 4, 8, 8, 9, 9, 8, 5, 9]
Output: 
{{
    "List 1": [3, 1, 9, 3, 7, 5, 5, 4, 8, 1, 5, 3, 3, 2, 3, 0],
    "List 2": [9, 7, 2, 2, 4, 4, 8, 5, 0, 8, 7, 3, 3, 8, 7, 0],
    "List 3": [9, 5, 1, 6, 7, 6, 8, 9, 0, 3, 0, 6, 3, 4, 8, 0],
    "List 4": [6, 9, 8, 4, 1, 2, 9, 0, 4, 8, 8, 9, 9, 8, 5, 9]
}}
</Example>

Input: {input}"""

split128 = """<Instruction> Split the following list of 128 numbers into 8 lists of 16 numbers each, the first list should contain the first 16 numbers, the second list the second 16 numbers, the third list the third 16 numbers, the fourth list the fourth 16 numbers, the fifth list the fifth 16 numbers and so on.
Only output the final 8 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, ...],
    "List 2": [2, 9, 2, 4, 7, 1, 5, ...],
    "List 3": [6, 9, 8, 1, 9, 2, 4, ...],
    "List 4": [9, 0, 7, 6, 5, 6, 6, ...],
    "List 5": [7, 9, 4, 1, 1, 8, 1, ...],
    "List 6": [1, 9, 0, 4, 3, 3, 5, ...],
    "List 7": [2, 4, 3, 5, 8, 2, 2, ...],
    "List 8": [4, 2, 1, 2, 7, 6, 8, ...]
}} </Instruction>

<Example>
Input: [6, 0, 2, 3, 8, 3, 0, 2, 4, 5, 4, 1, 3, 6, 9, 8, 3, 1, 2, 6, 5, 3, 9, 8, 9, 1, 6, 1, 0, 2, 8, 9, 5, 3, 1, 2, 7, 9, 4, 8, 8, 9, 3, 2, 8, 4, 7, 4, 3, 8, 7, 3, 6, 4, 0, 0, 6, 8, 1, 5, 8, 7, 5, 1, 4, 0, 8, 6, 1, 3, 6, 1, 7, 6, 8, 7, 3, 7, 8, 2, 0, 8, 2, 6, 0, 0, 9, 9, 8, 6, 9, 4, 8, 5, 5, 0, 0, 9, 3, 9, 4, 0, 5, 6, 2, 4, 6, 7, 7, 7, 8, 0, 4, 9, 1, 4, 8, 5, 1, 4, 4, 7, 4, 9, 3, 9, 6, 7]
Output: 
{{
    "List 1": [6, 0, 2, 3, 8, 3, 0, 2, 4, 5, 4, 1, 3, 6, 9, 8],
    "List 2": [3, 1, 2, 6, 5, 3, 9, 8, 9, 1, 6, 1, 0, 2, 8, 9],
    "List 3": [5, 3, 1, 2, 7, 9, 4, 8, 8, 9, 3, 2, 8, 4, 7, 4],
    "List 4": [3, 8, 7, 3, 6, 4, 0, 0, 6, 8, 1, 5, 8, 7, 5, 1],
    "List 5": [4, 0, 8, 6, 1, 3, 6, 1, 7, 6, 8, 7, 3, 7, 8, 2],
    "List 6": [0, 8, 2, 6, 0, 0, 9, 9, 8, 6, 9, 4, 8, 5, 5, 0],
    "List 7": [0, 9, 3, 9, 4, 0, 5, 6, 2, 4, 6, 7, 7, 7, 8, 0],
    "List 8": [4, 9, 1, 4, 8, 5, 1, 4, 4, 7, 4, 9, 3, 9, 6, 7]
}}
</Example>

Input: {input}
Output: """

def split(
    graph, 
    nodes,
    model = "",
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        if isinstance(graph_node["thought"], list):
            num_elems = len(graph_node["thought"])
        else:
            num_elems = len(ast.literal_eval(graph_node["thought"]))

        if num_elems == 32:
            split_prompt = split32
        elif num_elems == 64:
            split_prompt = split64
        elif num_elems == 128:
            split_prompt = split128
        else:
            split_prompt = splitx

        out = llm(
            split_prompt.format(
                input=graph_node["thought"]
            ), 
            model=model
        )[0]
        
        # Parse the result
        for k, v in PARSE_OUT_DICT.items():
            out = out.replace(k, v)
        as_dict = ast.literal_eval(out)

        # 2. Update the graph
        for k, v in as_dict.items():
            idx = max(list(graph.nodes)) + 1
            graph.add_node(
                idx, 
                thought=v, 
                score=None
            )
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

Input: {input}
Output: """

def sort(
    graph, 
    nodes,
    model = "",
):
    sorted_nodes = []
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(sort_prompt.format(input=graph_node["thought"]), model=model)[0]
        
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
        sorted_nodes.append(idx)

    # Score all the sorted nodes
    for node in sorted_nodes:
        graph, _ = score(graph, [node], model=model)

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
    model = "",
):
    refined_nodes = []
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        # Skip if the node is already correct
        if graph_node.get("score", None) is not None and graph_node["score"] == 0:
            original = graph_node["original"]
            output = graph_node["thought"]

        else:
            original = graph_node["original"]

            prompt = refine_prompt.format(
                input=original,
                incorrectly_sorted=graph_node["thought"]
            )

            output = llm(prompt, model=model)[0]

            for k, v in PARSE_OUT_DICT.items():
                output = output.replace(k, v)

        # Update the graph
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=output,
            score=None,
            original=original,
        )
        graph.add_edge(node_idx, idx)
        refined_nodes.append(idx)

    # Score all the refined nodes
    for node in refined_nodes:
        graph, _ = score(graph, [node], model=model)

    return graph, False



def score(
    graph, 
    nodes,
    model = "",
):

    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        # Skip scoring if already scored
        if "score" in graph_node.keys() and graph_node["score"] is not None:
            continue
        
        feedback = {}
        try:
            # Extract thought
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

            # scoring feedback
            feedback = {
                "missing_elements": 0,
                "extra_elements": 0,
            }
            
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
                diff = real_freq_dict[i] - thought_freq_dict[i]
                errors += abs(diff)

                if diff > 0:
                    feedback["missing_elements"] += abs(diff)
                if diff < 0:
                    feedback["extra_elements"] += abs(diff)
        
        # Assign a large error if scoring fails
        except:
            errors = 1000000

        graph.nodes[node_idx]["score"] = errors
        if errors > 0:
            graph.nodes[node_idx]["feedback"] = feedback

    return graph, False

def keepbest(
    graph, 
    nodes,
    model = "",
):
    # Score all non-scored nodes
    graph, _ = score(graph, nodes, model=model)
    return common_keepbest(graph, nodes, model)


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
    model = "",
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
    )[0]

    for k, v in PARSE_OUT_DICT.items():
        out = out.replace(k, v)
    
    # 2. Update the graph
    if isinstance(graph.nodes[int(nodes[0])]["thought"], list):
        node1 = graph.nodes[int(nodes[0])]["thought"]
    else:
        node1 = ast.literal_eval(graph.nodes[int(nodes[0])]["thought"])

    if isinstance(graph.nodes[int(nodes[1])]["thought"], list):
        node2 = graph.nodes[int(nodes[1])]["thought"]
    else:
        node2 = ast.literal_eval(graph.nodes[int(nodes[1])]["thought"])


    combined_list = str(node1 + node2)
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

# Baselines

def io(
    graph,
    nodes,
    model = "",
):
    return sort(graph, nodes, model)

def _tot_schedule(
        width: int,
        depth: int,
) -> int:
    return _common_tot_schedule(
        width=width,
        depth=depth,
        generate_action="sort",
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
        generate_action="sort",
        generate_attempts=generate_attempts,
        aggregate_attempts=aggregate_attempts,
        post_aggregate_keepbest=post_aggregate_keepbest,
        post_aggregate_refine=post_aggregate_refine,
        refine_attempts=refine_attempts,
    )

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

def cot(
    graph, 
    nodes,
    model = "",
):
    for node in nodes:
        # 1. Send the prompt
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        out = llm(sort_cot_prompt.format(input=graph_node["thought"]), model=model)[0]
        
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