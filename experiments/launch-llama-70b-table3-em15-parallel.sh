# em15 sorting32 (0-49, 50-100)
tmux new-session -d -s sorting32-em15-llm-0-49 "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task sorting32 \
                                --agent llm \
                                --max_iterations 23 \
                                --cot_sc_branches 15 \
                                --start 0 \
                                --end 49 \
                                &> experiments/logs/ablation/llama-3.1-70b/sorting32-em15-llm-0-49.log"

tmux new-session -d -s sorting32-em15-llm-50-100 "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task sorting32 \
                                --agent llm \
                                --max_iterations 23 \
                                --cot_sc_branches 15 \
                                --start 50 \
                                --end 100 \
                                &> experiments/logs/ablation/llama-3.1-70b/sorting32-em15-llm-50-100.log"
# em15 sorting64 (0-49, 50-100)
tmux new-session -d -s sorting64-em15-llm-0-49 "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task sorting64 \
                                --agent llm \
                                --max_iterations 59 \
                                --cot_sc_branches 15 \
                                --start 0 \
                                --end 49 \
                                &> experiments/logs/ablation/llama-3.1-70b/sorting64-em15-llm-0-49.log"

tmux new-session -d -s sorting64-em15-llm-50-100 "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task sorting64 \
                                --agent llm \
                                --max_iterations 59 \
                                --cot_sc_branches 15 \
                                --start 50 \
                                --end 100 \
                                &> experiments/logs/ablation/llama-3.1-70b/sorting64-em15-llm-50-100.log"


# tmux new-session -d -s sorting64-em15-llm "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
#                                 --task sorting64 \
#                                 --agent llm \
#                                 --max_iterations 59 \
#                                 --cot_sc_branches 15 \
#                                 &> experiments/logs/ablation/llama-3.1-70b/sorting64-em15-llm.log"

# sorting128
tmux new-session -d -s sorting128-em15-llm "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task sorting128 \
                                --agent llm \
                                --max_iterations 131 \
                                --cot_sc_branches 15 \
                                &> experiments/logs/ablation/llama-3.1-70b/sorting128-em15-llm.log"

# set_intersection32
tmux new-session -d -s set-intersection32-em15-llm "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task set_intersection32 \
                                --agent llm \
                                --max_iterations 20 \
                                --cot_sc_branches 15 \
                                &> experiments/logs/ablation/llama-3.1-70b/set-intersection32-em15-llm.log"

# set_intersection64
tmux new-session -d -s set-intersection64-em15-llm "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task set_intersection64 \
                                --agent llm \
                                --max_iterations 50 \
                                --cot_sc_branches 15 \
                                &> experiments/logs/ablation/llama-3.1-70b/set-intersection64-em15-llm.log"

# set_intersection128
tmux new-session -d -s set-intersection128-em15-llm "conda deactivate; conda activate sglang; cd aries; python -u src/main.py \
                                --task set_intersection128 \
                                --agent llm \
                                --max_iterations 75 \
                                --cot_sc_branches 15 \
                                &> experiments/logs/ablation/llama-3.1-70b/set-intersection128-em15-llm.log"