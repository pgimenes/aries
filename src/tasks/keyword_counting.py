from llm import llm, async_llm
import ast
import pandas as pd
import asyncio
from .common import (
    _common_tot_schedule, 
    _common_got_schedule, 
    common_keepbest, 
    PARSE_OUT_DICT,
)

from .countries import COUNTRIES

problem_definition = "Count the frequency of how many times each country is explicitly named in the input text."

actions = {
    "split": {
        "description": "Split the input text into paragraphs to decompose the problem. ",
        "preconditions": "",
        "effects": "This creates new nodes connected to the original node.",
    },
    "count": {
        "description": "Count the frequency of each country that appears at least once in the input text.",
        "preconditions": "",
        "effects": "This creates a new node connected to the original node.",
    },
    "aggregate": {
        "description": "Merge the frequency counts of the selected nodes into a single count.",
        "preconditions": "",
        "effects": "This action creates a new node connected to the selected nodes.",
    },
    # "refine": {
    #     "description": "Refine a count by fixing any existing mistakes.",
    #     "preconditions": "This action should only be called on nodes that have already been scored and contain mistakes (i.e. non-zero scores).",
    #     "effects": "This creates a new node connected to the original node.",
    # },
    # "score": {
    #     "description": "Count the number of mistakes in the node.",
    #     "preconditions": "",
    #     "effects": "The error count is annotated in the metadata of each node, and no new nodes are created.",
    # },
    "keepbest": {
        "description": "Out of the selected nodes, keep the one with the highest score, and delete the rest.",
        "preconditions": "The selected nodes must have been scored.",
        "effects": "All selected nodes are deleted, but the one with the highest score is duplicated as a new node.",
    },
    "groundtruth": {
        "description": "Compare the sorted list in a node with the ground truth.",
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
0: {'thought': "One evening, Sarah, an archaeologist from Norway made a surprising discovery about ancient trade routes between Sweden and Norway. As per her research, the artifacts that were found in Norway were identical to those in Sweden, indicating a deep-rooted cultural connection between Sweden and Norway. This piqued the interest of her colleague, James, who was from Canada. He had been researching the indigenous tribes of Canada and found many similarities with tribes from his neighboring country, the United States. James had always been interested in the historical ties between Canada and United States, and his study further confirmed the age-old connections between the two countries. Upon hearing James's story, Sarah shared a fascinating anecdote from her travels in Portugal. She recalled how locals loved to tell the tale of the shared history between Spain and Portugal. Her anecdotes about Spain and Portugal echoed the same sense of shared culture and past, just like in the case of Norway and Sweden, and Canada and United States. Their conversation reminded James of his stay in South Korea, where he had learned about the close relationship between North Korea and South Korea, despite their current political divide. He recalled stories about the shared history of North Korea and South Korea, whose deep-seated cultural ties transcended political boundaries. Sarah, who had been to Australia, reciprocated with her own experiences of the bond between Australia and New Zealand. She described how, despite geographical separation, Australia and New Zealand shared a unique camaraderie and close historical ties. As they exchanged stories, their conversation moved to South Africa and its various connections with its neighbouring country, Zimbabwe. Sarah shared stories she had heard about the intricate bond between South Africa and Zimbabwe, showcasing the age-old interactions between these two nations. It left them both reflecting on the timeless bonds that connect nations across the world, from Norway to Australia, Canada to Zimbabwe, and all the countries in between.",

Edges:

OUTPUT:

<analysis>
A. Action history: No actions have been taken yet. 

B. Graph state: The graph currently has 1 node and 0 edges. Node 0 contains the initial problem. 

C. Strategy analysis: The strategy for solving the problem has not been determined yet.

D. Next action options
    1. Attempt to count the keywords on the entire text. This may be effective if the text is not too long.

    2. Decompose the text into sentences to make it easier to count the keywords. This may be effective if the text is too long to count directly.
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
The text is long and contains multiple sentences. Splitting the text into sentences will make it easier to count the keywords.
</explanation>
</example>""",
]

examples = "\n".join(examples)

# additional_instructions = """- The count action may have a low probability of success. If previous counts were unsuccessful, it is recommended to repeat counting with higher attempt counts to increase the likelihood of finding a correct solution.
# - If counting is not successful after many attempts, it is recommended to split the text into smaller segments to make counting easier, then aggregate the counts from each segment."""

# Implementation

split2_prompt = """<Instruction> Split the following input text into 2 paragraphs of approximately same length.
Only output the final 4 paragraphs in the following format without any additional text or thoughts:
{{
    "Paragraph 1": "Some paragraph text ...",
    "Paragraph 2": "Some paragraph text ...",
}} </Instruction>

<Example>
Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Output: 
{{
    "Paragraph 1": "Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away.",
    "Paragraph 2": "She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.",
}}
</Example>

Input:
{input}
"""

split4_prompt = """<Instruction> Split the following input text into 4 paragraphs of approximately same length.
Only output the final 4 paragraphs in the following format without any additional text or thoughts:
{{
    "Paragraph 1": "Some paragraph text ...",
    "Paragraph 2": "Some paragraph text ...",
    "Paragraph 3": "Some paragraph text ...",
    "Paragraph 4": "Some paragraph text ..."
}} </Instruction>

<Example>
Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Output: 
{{
    "Paragraph 1": "Journeying westward, she admired the art in Italy and sipped coffee in France. ",
    "Paragraph 2": "The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away.",
    "Paragraph 3": "She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.",
    "Paragraph 4": "Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit."
}}
</Example>

Input:
{input}
"""

splitx_prompt = """<Instruction> Split the following input text into individual sentences.
Output each sentence in the following format without any additional text or thoughts:
{{
    "Sentence 1": "Some sentence text ...",
    "Sentence 2": "Some sentence text ...",
    "Sentence 3": "Some sentence text ...",
    ...
}} </Instruction>

<Example>
Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Output: 
{{
    "Sentence 1": "Journeying westward, she admired the art in Italy and sipped coffee in France. ",
    "Sentence 2": "The music of Spain and the history of Greece deepened her love for Europe. "
    "Sentence 3": "The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away.",
    "Sentence 4": "She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.",
    "Sentence 5": "Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit."
}}
</Example>

Input:
{input}
Output:
"""

def split(
    graph, 
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[int(node)]
        out = llm(splitx_prompt.format(input=graph.nodes[int(node)]["thought"]), model=model)[0]

        as_dict = ast.literal_eval(out)

        for value in as_dict.values():
            idx = max(list(graph.nodes)) + 1
            graph.add_node(
                idx,
                thought=value,
            )
            graph.add_edge(node_idx, idx)

    return graph, False

count_prompt = """<Instruction> Count the frequency of how many times each country is explicitly named in the input text. Output only the frequency of each country that appears at least once in the following json format; make sure to keep the same spelling and output no additional text:
{{
    "country1": frequency1,
    "country2": frequency2,
    ...
}}
</Instruction>

<Approach>
To count the frequency for each country follow these steps:
1. Create an empty dictionary.
2. Iterate through the text word by word.
3. If the word corresponds to a country, add the country to the dictionary and set its value to 1 if it is not already in the dictionary. If the word is already in the dictionary, increment its value by 1.
</Approach>

<Examples>
Input:
Alexandra explored the rainforests of Brazil and danced the tango in Argentina.
Output: 
{{
    "Brazil": 1,
    "Argentina": 1    
}}

Input:
In Norway she found stones that were identical to those in Sweden, indicating a deep-rooted cultural connection between Sweden and Norway.
Output:
{{
    "Norway": 2,
    "Sweden": 2
}}

Input:
A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Output: 
{{
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Peru": 1,
    "Brazil": 1,
    "Mexico": 1
}}

Input:
Italy, Sweden, Sweden and Germany will always stay her favourite destinations to visit.
Output:
{{
    "Italy": 1,
    "Sweden": 2,
    "Germany": 1
}}
</Examples>

Input:
{input}
Output:
"""

async def async_count(
    graph,
    nodes,
    model = "",
    multiplicity: int = 1,
):
    return await asyncio.gather(
        *[
            async_llm(
                count_prompt.format(
                    input=graph.nodes[int(node)]["thought"]
                ), 
                model=model
            ) for node in nodes
        ],
    )

def count(
    graph, 
    nodes,
    model = "",
    run_async = True,
    multiplicity: int = 1,
):
    
    # 1. Get LLM responses
    if run_async:
        outs = asyncio.run(async_count(graph, nodes, model=model, multiplicity=multiplicity))
        outs = {
            node: out[0] for node, out in zip(nodes, outs)
        }
    else:
        outs = {
            node: llm(
                count_prompt.format(
                    input=graph.nodes[int(node)]["thought"]
                ), 
                model=model
            )[0] for node in nodes
        }

    # 2. Update the graph
    counted_nodes = []
    for node in nodes:
        node_idx = int(node)
        out = outs[node]

        for key in PARSE_OUT_DICT:
            out = out.replace(key, PARSE_OUT_DICT[key])

        as_dict = ast.literal_eval(out)

        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=as_dict,
            original=graph.nodes[int(node)]["thought"],
        )
        graph.add_edge(node_idx, idx)
        counted_nodes.append(idx)

    # Score
    graph, _ = score(graph, counted_nodes, model=model)

    return graph, False

refine_prompt = """<Instruction> The following two inputs represent an initial input text and a dictionary of countries and their frequencies of explicit appearance in the input text. The dictionary is incorrect and might not contain all countries, extra countries or incorrect frequencies.
Fix the dictionary such that it has the correct frequencies for each country that appears at least once in the input text. Only output the reason why it's incorrect and the correct output, as shown in the examples, and nothing else.</Instruction>

<Approach>
To fix the incorrect list of countries follow these steps:
1. Iterate through the input text and find all countries that are explicitly mentioned.
2. Count the frequency of each country in the input text.
3. Compare the frequency of each country in the input text with the frequency of the country in the incorrect dictionary and update the frequency in the incorrect dictionary if they are different.

</Approach>

<Example>
Input:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Incorrect Dictionary:
{{
    "Canada": 1,
    "Mexico": 1,
    "Argentina": 1
}}
Reason: The input text names Brasil once but the incorrect dictionary does not contain Brasil at all, the remaining countries are correct.
Output: 
{{
    "Canada": 1,
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1
}}
</Example>

<Example>
Input:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Incorrect Dictionary:
{{
    "Peru": 3,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1,
    "Argentina": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}
Reason: The input text names Peru twice, but the incorrect dictionary lists it with a frequency of 3 instead of 2. The incorrect dictionary also contains Argentina which does not appear in the input text.
Output: 
{{
    "Peru": 2,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}
</Example>

<Example>
Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Incorrect Dictionary:
{{
    "Italy": 1,
    "France": 1,
    "Spain": 1,
    "Greece": 1,
    "Norway": 1,
    "Sweden": 1,
    "Finland": 1,
    "Denmark": 1,
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 1,
    "Russia": 1
}}
Reason: The input text names Italy, Norway, Sweden and Germany twice each, but the incorrect dictionary lists them with a frequency of 1 each instead of 2.
Output: 
{{
    "Italy": 2,
    "France": 1,
    "Spain": 1,
    "Greece": 1,
    "Norway": 2,
    "Sweden": 2,
    "Finland": 1,
    "Denmark": 1,
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 2,
    "Russia": 1
}}
</Example>

Input: 
{input}
Incorrect Dictionary: 
{incorrect_dict}
Reason:"""

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
                    incorrect_dict=graph.nodes[int(node)]["thought"], 
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
                    incorrect_dict=graph.nodes[int(node)]["thought"], 
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

            try:
                # Extract the refined count as a dict
                output = output.split("{")[1].split("}")[0]
                output = ast.literal_eval("{" + output + "}")
            except:
                # If not successful, keep the original count
                output = graph.nodes[int(node)]["thought"]
        else:
            original, output = refined_nodes[node]["original"], refined_nodes[node]["thought"]


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

with open("data/keyword_counting.csv", "r") as f:
    data = pd.read_csv(f)
        
def get_ground_truth(text):
    count = {}
    for country in COUNTRIES:
        cnt = text.count(country)
        if cnt > 0:
            count[country] = cnt
    return count
        
def score(
    graph, 
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    for node in nodes:
        graph_node = graph.nodes[int(node)]
        
        # Skip scoring if already scored
        if "score" in graph_node.keys() and graph_node["score"] is not None:
            continue
        
        try:
            original = graph_node["original"]
            real_count = get_ground_truth(original)
            error = 0

            # scoring feedback
            feedback = {
                "missing_elements": 0,
                "extra_elements": 0,
            }

            for country in COUNTRIES:
                diff = real_count.get(
                    country,
                    0,
                ) - graph.nodes[int(node)]["thought"].get(
                    country, 
                    0,
                )

                if diff > 0:
                    feedback["missing_elements"] += abs(diff)
                if diff < 0:
                    feedback["extra_elements"] += abs(diff)
                
                error += abs(diff)
        
        # Assign a large error if scoring fails
        except:
            errors = 1000000

        graph_node["score"] = error
        if error > 0:
            graph_node["feedback"] = feedback
            
    return graph, False

def keepbest(
    graph, 
    nodes,
    model = "",
    run_async = False,
    multiplicity: int = 1,
):
    # Score all nodes first
    graph, _ = score(graph, nodes, model=model)
    return common_keepbest(graph, nodes, model)

def aggregate(
    graph, 
    nodes,
    model="",
    run_async = False,
    multiplicity: int = 1,
):
    # Initialize merged thoughts and score
    merged_thought = {}
    merged_score = 0
    merged_original = []

    # Iterate over all specified nodes to merge their thoughts and scores
    for node_id in nodes:
        node = graph.nodes[int(node_id)]
        thought = node["thought"]
        merged_original.append(node["original"])

        # Merge thoughts by summing values
        for k in thought:
            merged_thought[k] = merged_thought.get(k, 0) + thought.get(k, 0)

        # Aggregate score if available
        if node.get("score") is not None:
            merged_score += node["score"]
        else:
            merged_score = None  # Set to None if any node lacks a score

    for _ in range(multiplicity):
        # Generate new node index
        idx = max(graph.nodes) + 1

        # Add new aggregated node
        graph.add_node(
            idx,
            thought=merged_thought,
            score=merged_score,
            original=" ".join(merged_original),
        )

        # Add edges from all specified nodes to the new aggregated node
        for node_id in nodes:
            graph.add_edge(int(node_id), idx)

    # Score the aggregation
    graph, _ = score(graph, [idx], model=model)

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
    real_count = get_ground_truth(original)

    # 2. Set the "original" field to the original problem in node 0
    # and score each node against the original to override
    # any incorrectly scored values
    for node in nodes:
        graph.nodes[int(node)]["original"] = original
        
    graph, _ = score(
        graph,
        nodes,
        model
    )

    any_match = False
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        if graph_node["score"] == 0:
            graph.nodes[node_idx]["matches_ground_truth"] = True
            any_match = True
        else:
            graph.nodes[node_idx]["matches_ground_truth"] = False

    return graph, any_match

# Baselines

def io(
    graph, 
    nodes,
):
    return count(graph, nodes)

def _tot_schedule(
    width: int,
    depth: int,
) -> int:
    return _common_tot_schedule(
        width, 
        depth,
        generate_action="count",
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
        generate_action="count",
        generate_attempts=generate_attempts,
        aggregate_attempts=aggregate_attempts,
        post_aggregate_keepbest=post_aggregate_keepbest,
        post_aggregate_refine=post_aggregate_refine,
        refine_attempts=refine_attempts,
    )

cot_prompt = """<Instruction> Count the frequency of how many times each country is explicitly named in the input text. You can generate any intermedate lists and states, but the final output should only contain the frequency of each country that appears at least once in the following json format, prefixed with "Output: " (make sure to keep the same spelling for each country in the output as in the input text):
{{
    "country1": frequency1,
    "country2": frequency2,
    ...
}}
</Instruction>

<Approach>
To count the frequency for each country follow these steps:
1. Split the input passage into four paragraphs of similar length.
2. Count the frequency of each country in each paragraph.
3. Combine the frequencies of each country from each paragraph by adding them together.
</Approach>

<Examples>
Input:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Paragraphs:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. 

Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Sublist frequencies:
{{
    "Canada": 1
}}

{{
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1
}}
Output: 
{{
    "Canada": 1,
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1
}}

Input:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Paragraphs:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. 

A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
Sublists:
{{
    "Peru": 1,
    "Chile": 1
}}

{{
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Peru": 1,
    "Brazil": 1,
    "Mexico": 1
}}
Output: 
{{
    "Peru": 2,
    "Chile": 1,
    "Uruguay": 1,
    "Paraguay": 1,
    "Canada": 1,
    "Brazil": 1,
    "Mexico": 1
}}

Input:
Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Paragraphs:
Journeying westward, she admired the art in Italy and sipped coffee in France. 

The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. 

She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. 

Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.
Sublists:
{{
    "Italy": 1,
    "France": 1
}}

{{
    "Spain": 1,
    "Greece": 1,
    "Norway": 1,
    "Sweden": 1,
    "Finland": 1,
    "Denmark": 1
}}

{{
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 1,
    "Russia": 1
}}

{{
    "Italy": 1,
    "Norway": 1,
    "Sweden": 1,
    "Germany": 1
}}
Output: 
{{
    "Italy": 2,
    "France": 1,
    "Spain": 1,
    "Greece": 1,
    "Norway": 2,
    "Sweden": 2,
    "Finland": 1,
    "Denmark": 1,
    "Ireland": 1,
    "Scotland": 1,
    "Germany": 2,
    "Russia": 1
}}
</Examples>

Input:
{input}
Paragraphs:"""

def cot(
    graph, 
    nodes,
    model = "",
):
    for node in nodes:
        node_idx = int(node)
        out = llm(cot_prompt.format(input=graph.nodes[int(node)]["thought"]), model=model)[0]

        # Extract steps and answer
        output = out.split("Output:")[1]
        output = ast.literal_eval(output)
        
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=output,
            original=graph.nodes[int(node)]["thought"],
        )
        graph.add_edge(node_idx, idx)

    return graph, False
