import re

def analyze_log(log_text):
    aggregate_pattern = re.compile(r"Step (\d+)\n=+\nAction: aggregate\nNodes: \[(.*?)\]", re.MULTILINE)
    score_pattern = re.compile(r"Step (\d+)\n=+\nAction: score\nNodes: \[(.*?)\]", re.MULTILINE)
    score_value_pattern = re.compile(r"(\d+):\s*\{[^}]*'score':\s*(\d+)", re.MULTILINE)
    
    success, failure = [], []
    i = 0

    cnt = 0
    
    while i < len(log_text):
        print(i / len(log_text))  # Optional, for progress tracking
        aggregate_match = aggregate_pattern.search(log_text, i)

        if aggregate_match:
            aggregate_step, node_ids = int(aggregate_match.group(1)), aggregate_match.group(2).split(', ')
            i = aggregate_match.end()

            score_match = score_pattern.search(log_text, i)
            if score_match:
                score_step, score_node_ids = int(score_match.group(1)), score_match.group(2).split(', ')
                if score_step == aggregate_step + 1:

                    cnt += 1
                    
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
                                success.append((aggregate_step, node_ids))
                            else:
                                failure.append((aggregate_step, node_ids))

        else:
            break
    
    total = len(success) + len(failure)
    success_rate = (len(success) / total * 100) if total > 0 else 0

    # print(f"Successes: {success}")
    # print(f"Failures: {failure}")    

    print(f"Instances found: {cnt}")
    print(f"Success: {len(success)}, Failure: {len(failure)}, Success Rate: {success_rate:.2f}%")

    return success, failure, success_rate

# Example usage
with open("experiments/logs/llama-3.1-405b/sorting32-got100.log", "r") as file:
    log_data = file.read()

analyze_log(log_data)
