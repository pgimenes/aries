from llm import llm
import ast
import pandas as pd
from .common import (
    _common_tot_schedule, 
    _common_got_schedule, 
    common_keepbest, 
)

from .countries import COUNTRIES

model = "gpt-4"

problem_definition = "Count the frequency of how many times each country is explicitly named in the input text."

actions = {
    "split": "Split the input text into smaller segments.",
    "count": "Count the frequency of each country that appears at least once in the input text.",
    "score": "Count the number of mistakes in the frequency of each country in the input text.",
    "refine": "Fix the incorrect list of countries.",
    "keepbest": "Keep the best output.",
    "aggregate": "Combine the frequencies of countries from two segments.",
    "groundtruth": "Compare the current output with the ground truth.",
}

# Implementation

split_prompt = """<Instruction> Split the following input text into individual sentences.
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
):
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[int(node)]
        out = llm(split_prompt.format(input=graph.nodes[int(node)]["thought"]), model=model)[0]

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

def count(
    graph, 
    nodes,
):
    for node in nodes:
        node_idx = int(node)
        out = llm(count_prompt.format(input=graph.nodes[int(node)]["thought"]), model=model)[0]

        as_dict = ast.literal_eval(out)

        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=as_dict,
            original=graph.nodes[int(node)]["thought"],
        )
        graph.add_edge(node_idx, idx)

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

def refine(
    graph, 
    nodes,
):
    for node in nodes:
        node_idx = int(node)
        graph_node = graph.nodes[node_idx]

        out = llm(
            refine_prompt.format(
                input=graph.nodes[int(node)]["original"], 
                incorrect_dict=graph.nodes[int(node)]["thought"], 
            ),
            model=model,
        )


        try:
            output = out[0].split("{")[1].split("}")[0]
            as_dict = ast.literal_eval("{" + output + "}")
        except:
            as_dict = graph_node["thought"]
        
        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=as_dict,
            original=graph.nodes[node_idx]["original"],
        )
        graph.add_edge(node_idx, idx)

    return graph, False

with open("data/keyword_counting.csv", "r") as f:
    data = pd.read_csv(f)

def get_country_list(text):
    for i in data.iterrows():
        if i[1]["Text"] == text:
            clist = i[1]["Countries"]
            clist = clist.split("[")[1].split("]")[0].split(", ")
            return clist
        
def get_ground_truth(text, country_list):
    count = {}
    for country in COUNTRIES:
        cnt = text.count(country)
        if cnt > 0:
            count[country] = cnt
    return count
        
def score(
    graph, 
    nodes,
):
    for node in nodes:
        graph_node = graph.nodes[int(node)]
        
        if "score" in graph_node.keys() and graph_node["score"] is not None:
            continue
        
        original = graph_node["original"]
        
        country_list = get_country_list(original)
        
        real_count = get_ground_truth(original, country_list)
        error = 0

        for country in COUNTRIES:
            err = abs(
                real_count.get(
                    country,
                    0,
                ) - graph.nodes[int(node)]["thought"].get(
                    country, 
                    0,
                )
            )
            error += err

        graph_node["score"] = error
            
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
    return common_keepbest(graph, nodes)

def aggregate(
    graph, 
    nodes,
):
    node1 = graph.nodes[int(nodes[0])]
    thought1 = node1["thought"]

    node2 = graph.nodes[int(nodes[1])]
    thought2 = node2["thought"]

    # merge the two dictionaries
    thought = {k: thought1.get(k, 0) + thought2.get(k, 0) for k in set(thought1) | set(thought2)}

    idx = max(list(graph.nodes)) + 1
    graph.add_node(
        idx,
        thought=thought,
        score=node1["score"] + node2["score"],
        original=f"{node1['original']} {node2['original']}",
    )
    graph.add_edge(int(nodes[0]), idx)
    graph.add_edge(int(nodes[1]), idx)

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
        country_list = get_country_list(original)
        
        matches = True
        real_count = get_ground_truth(original, country_list)
        for country in country_list:
            if country not in graph_node["thought"].keys():
                matches = False
            
            if real_count[country] != graph_node["thought"][country]:
                matches = False

        if matches:
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

