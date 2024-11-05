import ast

_common_io_action_list = [
    "io",
    "score",
    "groundtruth",
]

_common_io_node_list = [
    "0",
    "1",
    "1",
]

_common_cot_action_list = [
    "cot",
    "score",
    "groundtruth",
]
_common_cot_node_list = [
    "0",
    "1",
    "1",
]

def get_parent_nodes(graph, node):
    parent_nodes = []
    for edge in graph.edges:
        if edge[1] == node:
            parent_nodes.append(edge[0])
    return parent_nodes

def common_keepbest(
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
    for idx, node in enumerate(nodes):
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

            for key, value in graph_node.items():
                if key == "thought":
                    continue
                graph.nodes[new_idx][key] = value

            parent_node = get_parent_nodes(graph, node_idx)[0]
            graph.add_edge(parent_node, new_idx)
        
        # Flag node to remove
        nodes_to_remove.append(node_idx)

    if not added:
        raise ValueError(f"Best node {best_node_idx} not found in nodes {nodes}")

    # Remove the other nodes
    for node in nodes_to_remove:
        graph.remove_node(node)

    return graph, False

# ToT

def _common_tot_schedule(
        width: int,
        depth: int,
        generate_action: str = "generate",
        refine_action: str = "refine",
) -> int:
    actions = []
    action_nodes = []
    keepbest_nodes = []
    last_node = 0

    # Sorting
    actions += [generate_action]
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
        actions += [refine_action]
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

def _common_got_schedule(    
    split_branches:int,
    sort_attempts:int,
    split_action:str = "split",
    generate_action:str = "generate",
) -> int:        
    # Create two split branches
    actions = [split_action]
    action_nodes = [["0"]]

    last_node = 2
    keepbest_nodes = []
    for split_branch in range(1, split_branches + 1):
        
        # Sorting
        sorted_nodes = []
        actions += [generate_action]
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