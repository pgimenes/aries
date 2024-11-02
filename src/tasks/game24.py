from llm import llm
import ast
import operator as op

model = "gpt-4"

problem_definition = "Use numbers and basic arithmetic operations (+ - * /) to obtain 24."

# IO

io_prompt = """<Instruction>Use numbers and basic arithmetic operations (+ - * /) to obtain 24.</Instruction>

<Example>
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
</Example>

<Example>
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
</Example>

<Example>
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
</Example>

</Example>
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
</Example>

<Example>
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
</Example>

Input: {input}
Answer: """

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
        out = llm(io_prompt.format(input=graph.nodes[int(node)]["thought"]), model=model)[0]
        graph.add_node(
            1,
            thought=out,
        )
        graph.add_edge(0, 1)

    return graph, False
        

cot_prompt = """<Instruction>Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.</Instruction>

<Example>
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
</Example>

<Example>
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
</Example>

<Example>
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
</Example>

<Example>
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
</Example>

<Example>
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
</Example>

Input: {input}
"""


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
        out = llm(cot_prompt.format(input=graph.nodes[int(node)]["thought"]), model=model)[0]

        # Extract steps and answer
        steps = out.split("Steps:")[1].split("Answer:")[0].strip()
        answer = out.split("Answer:")[1].strip()

        graph.add_node(
            1,
            thought=answer,
            steps=steps,
        )
        graph.add_edge(0, 1)

    return graph, False

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

        breakpoint()
        
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

# parse the arithmetic expression
# example: '(6 / (1 - 1/4)) = 24'
ops = {
    ast.Add: op.add, 
    ast.Sub: op.sub, 
    ast.Mult: op.mul, 
    ast.Div: op.truediv,
}

def eval_expr(node):
    if isinstance(node, ast.Expression):
        return eval_expr(node.body)
    if isinstance(node, ast.BinOp):
        return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
    if isinstance(node, ast.Constant):
        return node.value
    raise TypeError(node)

def groundtruth(
    graph, 
    nodes,
):
    original = graph.nodes[0]["thought"]
    original = [int(i) for i in original.split(" ")]

    any_match = False
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        expression = graph_node["thought"].split(' =')[0]
        matches = True
        
        # rule 1: the utilized digits should match the original set
        numbers = [int(s) for s in expression if s.isdigit()]
        numbers.sort()
        if numbers != original:
            matches = False

        # rule 2: the expression should evaluate to 24
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree)
        if result != 24:
            matches = False

        if matches:
            graph.nodes[node_idx]["matches_ground_truth"] = True
            any_match = True
        else:
            graph.nodes[node_idx]["matches_ground_truth"] = False

    return graph, any_match