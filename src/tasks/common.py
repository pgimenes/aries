from copy import copy

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
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        if graph_node["score"] < min_score:
            min_score = graph_node["score"]
            best_node_idx = node_idx

    # Duplicate the best node
    added = False
    nodes_to_remove = []
    for _, node in enumerate(nodes):
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]
        
        if node_idx == best_node_idx:
            added = True

            graph.add_node(
                new_idx, 
                thought=graph.nodes[int(best_node_idx)]["thought"], 
                score=min_score,
            )

            # Copy attributes from parent node
            for key, value in graph_node.items():
                if key in ["thought", "score"]:
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
    branch_heads = []
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
    branch_heads.append(str(last_node))

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
        branch_heads.append(str(last_node))

    # Keep best
    actions += ["keepbest"]
    action_nodes += [branch_heads]
    last_node += 1

    # Ground truth
    actions += ["groundtruth"]
    action_nodes += [[str(last_node)]]
    last_node += 1
    return actions, action_nodes

# GoT

def _common_got_schedule(
    branches:int,
    generate_action:str,
    generate_attempts:int,
    aggregate_attempts:int,
    post_aggregate_keepbest: bool,
    post_aggregate_refine: bool,
    refine_attempts:int,
) -> int:        
    assert (post_aggregate_keepbest or post_aggregate_refine), "At least one of post_aggregate_keepbest or post_aggregate_refine must be True"

    # Create split branches
    actions = ["split"]
    action_nodes = [["0"]]

    last_node = branches
    keepbest_nodes = []
    for split_branch in range(1, branches + 1):
        
        # Generate action e.g. sorting, intersection
        generate_nodes = []
        actions += [generate_action]
        action_nodes += [[str(split_branch)] * generate_attempts]
        generate_nodes += list(range(last_node + 1, last_node + 1 + generate_attempts))
        last_node += generate_attempts

        # Scoring
        actions += ["score"]
        action_nodes += [generate_nodes]

        # Keep best
        actions += ["keepbest"]
        action_nodes += [generate_nodes]
        last_node += 1
        keepbest_nodes += [str(last_node)]

    # Aggregate
    # Perform tree reduction
    branch_heads = copy(keepbest_nodes)
    while len(branch_heads) > 1:
        nodepairs = [branch_heads[i:i + 2] for i in range(0, len(branch_heads), 2)]

        for pair in nodepairs:
            if len(pair) == 1:
                continue

            # Aggregate
            actions += ["aggregate"] * aggregate_attempts
            action_nodes += [pair] * aggregate_attempts
            aggregate_nodes = list(range(last_node + 1, last_node + 1 + aggregate_attempts))
            last_node += aggregate_attempts

            # Score
            actions += ["score"]
            action_nodes += [aggregate_nodes]

            # Keep best
            if post_aggregate_keepbest:
                actions += ["keepbest"]
                action_nodes += [aggregate_nodes]
                last_node += 1
                keepbest_residual = last_node

            # Refine
            if post_aggregate_refine:
                actions += ["refine"]
                
                if post_aggregate_keepbest:
                    action_nodes += [[last_node] * refine_attempts] 
                    refine_nodes = list(range(last_node + 1, last_node + 1 + refine_attempts))
                    last_node += refine_attempts
                else:
                    action_nodes += [aggregate_nodes * refine_attempts]
                    refine_nodes = list(range(last_node + 1, last_node + 1 + refine_attempts * len(aggregate_nodes)))
                    last_node += refine_attempts * len(aggregate_nodes)
                
                # Score
                actions += ["score"]
                action_nodes += [refine_nodes]

                # Keep best
                actions += ["keepbest"]
                if post_aggregate_keepbest:
                    action_nodes += [refine_nodes + [keepbest_residual]]
                else:
                    action_nodes += [refine_nodes + aggregate_nodes]
                last_node += 1

            # Update keepbest list
            branch_heads.remove(pair[0])
            branch_heads.remove(pair[1])
            branch_heads.append(str(last_node))

    # Groundtruth
    actions += ["groundtruth"]
    action_nodes += [[str(last_node)]]
    last_node += 1
    return actions, action_nodes