import re

def analyze_log(log_text):
    refine_pattern = re.compile(r"Step (\d+)\n=+\nAction: refine\nNodes: \[(.*?)\]", re.MULTILINE)
    score_pattern = re.compile(r"Step (\d+)\n=+\nAction: score\nNodes: \[(.*?)\]", re.MULTILINE)
    score_value_pattern = re.compile(r'(\d+):\s*\{[^}]*"score":\s*(\d+)', re.MULTILINE)
    
    success, failure = [], []
    i = 0

    no_score_match = []
    no_correct_step = []
    
    while i < len(log_text):
        print(i / len(log_text))  # Optional, for progress tracking
        refine_match = refine_pattern.search(log_text, i)

        if refine_match:
            refine_step, node_ids = int(refine_match.group(1)), refine_match.group(2).split(', ')
            i = refine_match.end()

            # # Verify that the score of all nodes being refined is not 0
            # score_entries = score_value_pattern.findall(log_text[i:])
            # score_dict = {node: int(value) for node, value in score_entries}
            
            # if any(score_dict.get(node, 1) == 0 for node in node_ids):  # If any refined node has score 0
            #     print(f"Skipping since refining a node that's already scored 0: {node_ids}")
            #     continue  # Skip this refinement step
            
            score_match = score_pattern.search(log_text, i)
            if score_match:
                score_step, score_node_ids = int(score_match.group(1)), score_match.group(2).split(', ')
                if score_step == refine_step + 1:
                    
                    # Search for the correct score entry within the current step
                    # find the index for the next instance of "========================" from i
                    end_of_score = log_text.find("========================", score_match.end())
                    score_entries = score_value_pattern.findall(log_text[score_match.end():end_of_score])
                    score_dict = {}
                    for node, value in score_entries:
                        score_dict[node] = int(value)
                    
                    for node_id in score_node_ids:
                        
                        if node_id.replace("'", "") in score_dict:
                            if score_dict[node_id.replace("'", "")] == 0:
                                success.append((refine_step, node_ids))
                            else:
                                failure.append((refine_step, node_ids))

        else:
            break
    
    total = len(success) + len(failure)
    success_rate = (len(success) / total * 100) if total > 0 else 0

    print(f"Successes: {success}")
    print(f"Failures: {failure}")    

    print(f"No score match: {no_score_match}")
    print(f"No correct step: {no_correct_step}")

    print(f"Success: {len(success)}, Failure: {len(failure)}, Success Rate: {success_rate:.2f}%")

    return success, failure, success_rate

# Example usage
with open("experiments/logs/llama-3.1-405b/human-eval-got25.log", "r") as file:
    log_data = file.read()

analyze_log(log_data)
