from llm import llm
import ast
import pandas as pd

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

# IO

io_prompt = """<Instruction> Count the frequency of how many times each country is explicitly named in the input text. Output only the frequency of each country that appears at least once in the following json format; make sure to keep the same spelling and output no additional text:
{{
    "country1": frequency1,
    "country2": frequency2,
    ...
}}
</Instruction>

<Examples>
Input:
Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.
Output: 
{{
    "Canada": 1,
    "Mexico": 1,
    "Brazil": 1,
    "Argentina": 1    
}}

Input:
The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.
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
Output:"""

_io_action_list = [
    "count",
    "score",
    "groundtruth",
]

_io_node_list = [
    "0",
    "1",
    "1",
]

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


_cot_action_list = [
    "cot",
    "score",
    "groundtruth",
]
_cot_node_list = [
    "0",
    "1",
    "1",
]

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
    actions += ["count"]
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
    branches:int,
    attempts:int,
) -> int:        
    # Create two split branches
    actions = ["split"]
    action_nodes = [["0"]]

    last_node = branches
    keepbest_nodes = []
    for split_branch in range(1, branches + 1):
        
        # Sorting
        count_nodes = []
        actions += ["count"]
        action_nodes += [[str(split_branch)] * attempts]
        count_nodes += list(range(last_node + 1, last_node + 1 + attempts))
        last_node += attempts

        # Scoring
        actions += ["score"]
        action_nodes += [count_nodes]

        # Keep best
        actions += ["keepbest"]
        action_nodes += [count_nodes]
        last_node += 1
        keepbest_nodes += [str(last_node)]

    # Aggregate
    # create pairs of nodes from keepbest_nodes
    # todo: check on small graph if this works
    while len(keepbest_nodes) > 1:
        nodepairs = [keepbest_nodes[i:i + 2] for i in range(0, len(keepbest_nodes), 2)]

        for pair in nodepairs:
            if len(pair) == 1:
                continue

            actions += ["aggregate"]
            action_nodes += [pair]
            last_node += 1
            
            # Update keepbest list
            keepbest_nodes.remove(pair[0])
            keepbest_nodes.remove(pair[1])
            keepbest_nodes.append(str(last_node))

    # Refine
    actions += ["refine"]
    action_nodes += [keepbest_nodes]
    last_node += 1

    # Groundtruth
    actions += ["groundtruth"]
    action_nodes += [[str(last_node)]]
    last_node += 1
    return actions, action_nodes

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
                input=graph.nodes[0]["thought"], 
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
        
        # original = graph.nodes[0]["thought"]
        try:
            original = graph_node["original"]
        except:
            breakpoint()
        
        country_list = get_country_list(original)
        
        real_count = get_ground_truth(original, country_list)
        error = 0

        for country in COUNTRIES:
            
            # Only consider countries included in the text
            if country not in real_count.keys():
                continue
            
            # Add a large error when the agent forgets about a country
            if country in real_count.keys() and country not in graph.nodes[int(node)]["thought"].keys():
                error += 100
                continue

            error += abs(real_count[country] - graph.nodes[int(node)]["thought"][country])

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
    min_score = 1000000
    best_node_idx = None
    
    # Node id for the new node
    # (decide before deleting nodes)
    new_idx = max(list(graph.nodes)) + 1

    nodes_to_remove = []

    # Find node with highest score
    for idx, node in enumerate(nodes):
        graph_node = graph.nodes[int(node)]
        
        if graph_node["score"] < min_score:
            min_score = graph_node["score"]
            best_node_idx = node

    # Duplicate the best node
    for _, node in enumerate(nodes):
        node_idx = int(node)
        
        if node == best_node_idx:
            graph.add_node(
                new_idx, 
                thought=graph.nodes[int(best_node_idx)]["thought"], 
                score=min_score,
                original=graph.nodes[int(best_node_idx)]["original"],
            )

            parent_node = get_parent_nodes(graph, node_idx)[0]
            graph.add_edge(parent_node, new_idx)
        
        # Flag node to remove
        nodes_to_remove.append(node_idx)

    # Remove the other nodes
    for node in nodes_to_remove:
        graph.remove_node(node)

    return graph, False

aggregate_prompt = """<Instruction> Combine the following 2 dictionaries, each containing the frequency of countries in a text, into a single dictionary.
Simply add the frequencies together for each country and if a country is not present in one of the dictionaries, add it to the final dictionary with the frequency from the other dictionary.
Only output the final merged dictionary without any additional text or thoughts! </Instruction>

<Approach>
To combine the 2 dictionaries into single one, follow these steps:
1. Create a new dictionary to store the combined frequencies.
2. Iterate through the keys of the first dictionary and add the frequency of each country to the new dictionary.
3. Iterate through the keys of the second dictionary and add the frequency of each country to the new dictionary and if it is already present, add the frequency to the existing value.
</Approach>

Combine the following 2 dictionaries into a single dictionary:
{input1}

{input2}

Combined Output:
"""

def aggregate(
    graph, 
    nodes,
):
    # Split nodes list into pairs of nodes
    nodepairs = [nodes[i:i + 2] for i in range(0, len(nodes), 2)]

    for pair in nodepairs:
        node1, node2 = pair

        node1_idx = int(node1)
        node2_idx = int(node2)

        out = llm(
            aggregate_prompt.format(
                input1=graph.nodes[node1_idx]["thought"], 
                input2=graph.nodes[node2_idx]["thought"], 
            ),
            model=model,
        )[0]

        as_dict = ast.literal_eval(out)

        idx = max(list(graph.nodes)) + 1
        graph.add_node(
            idx,
            thought=as_dict,
        )
        graph.add_edge(node1_idx, idx)
        graph.add_edge(node2_idx, idx)

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