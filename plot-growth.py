import re
import matplotlib.pyplot as plt

def parse_log_and_plot(log_file_path):
    """
    Parses the log file to track the growth of accuracy over steps, ignoring specific actions.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        steps, accuracies: Lists of step numbers and corresponding accuracies.
    """
    steps = []
    accuracies = []
    current_accuracy = 0

    excluded_actions = {"keepbest", "score"}
    current_action = None

    # check if file exists
    try:
        with open(log_file_path, 'r') as file:
            pass
    except FileNotFoundError:
        print(f"File {log_file_path} not found.")
        return [0], [0]

    with open(log_file_path, 'r') as file:
        for line in file:
            # Match step lines
            step_match = re.match(r"^Step (\d+)", line)
            # Only increment step count if the last action was not excluded
            if step_match:
                
                # Always increment for getting the action
                # if "llm" in log_file_path:
                #     # steps.append(len(steps) + 1)
                #     # accuracies.append(current_accuracy)
                #     steps += [len(steps) + 1] * 5
                #     accuracies += [current_accuracy] * 5

                # Then increment for actions requiring LLM query
                if current_action not in excluded_actions:
                    steps.append(len(steps) + 1)
                    accuracies.append(current_accuracy)
                continue

            # Match action lines
            action_match = re.match(r"^Action: ([^\n]+)", line)
            if action_match:
                current_action = action_match.group(1)

            # Increment accuracy for success results
            if "Result: success" in line:
                current_accuracy += 1

    if not steps or not accuracies:
        print("No data found to plot.")
        return None, None

    print(f"log: {log_file_path}")
    print(f"num steps: {len(steps)}")
    print(f"num accuracies: {len(accuracies)}")

    return steps, accuracies

# Usage

# Generate data for sorting32
task = "sorting32"
sorting_llm_steps, sorting_llm_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-llm.log")
sorting_got25_steps, sorting_got25_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-got25.log")
sorting_got50_steps, sorting_got50_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-got50.log")
sorting_got100_steps, sorting_got100_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-got100.log")

# Generate data for sets
task = "set-intersection32"
set_llm_steps, set_llm_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-llm.log")
set_got25_steps, set_got25_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-got25.log")
set_got50_steps, set_got50_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-got50.log")
set_got100_steps, set_got100_accuracies = parse_log_and_plot(f"experiments/logs/llama-3.1-405b/{task}-got100.log")

# Create a figure with two horizontally arranged subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))  # Increased figure size for better visibility

# Define colors for better contrast
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red

# Shared font sizes for consistency
font_title = 16
font_axes = 16
font_legend =  16
font_ticks = 16

# Subfigure 1: set-intersection32
axes[0].plot(set_llm_steps, set_llm_accuracies, label="GoT-LLM", color=colors[0], linewidth=2)
axes[0].plot(set_got25_steps, set_got25_accuracies, label="GoT-25", color=colors[1], linewidth=2)
axes[0].plot(set_got50_steps, set_got50_accuracies, label="GoT-50", color=colors[2], linewidth=2)
axes[0].plot(set_got100_steps, set_got100_accuracies, label="GoT-100", color=colors[3], linewidth=2)
axes[0].set_title("set-intersection32", fontsize=font_title, weight='bold')
axes[0].set_xlabel("LLM Queries", fontsize=font_axes)
axes[0].set_ylabel("Number of successful iterations", fontsize=font_axes)
# axes[0].set_ylim(0, 105)
axes[0].tick_params(axis='both', which='major', labelsize=font_ticks)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(fontsize=font_legend, loc="lower right", frameon=True, shadow=True)

# Subfigure 2: sorting32
axes[1].plot(sorting_llm_steps, sorting_llm_accuracies, label="GoT-LLM", color=colors[0], linewidth=2)
axes[1].plot(sorting_got25_steps, sorting_got25_accuracies, label="GoT-25", color=colors[1], linewidth=2)
axes[1].plot(sorting_got50_steps, sorting_got50_accuracies, label="GoT-50", color=colors[2], linewidth=2)
axes[1].plot(sorting_got100_steps, sorting_got100_accuracies, label="GoT-100", color=colors[3], linewidth=2)
axes[1].set_title("sorting32", fontsize=font_title, weight='bold')
axes[1].set_xlabel("LLM Queries", fontsize=font_axes)
# axes[1].set_ylim(0, 105)
axes[1].tick_params(axis='both', which='major', labelsize=font_ticks)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(fontsize=font_legend, loc="lower right", frameon=True, shadow=True)

# Add spacing and save the improved figure
plt.tight_layout(pad=2.0)
plt.savefig("query-efficiency.pdf", dpi=300)  # High DPI for publication-quality output
plt.show()

