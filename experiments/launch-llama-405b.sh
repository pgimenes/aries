# sorting32
tmux new-session -d -s sorting32-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting32 --agent io &> experiments/logs/llama-3.1-405b/sorting32-io.log"
tmux new-session -d -s sorting32-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting32 --agent cot &> experiments/logs/llama-3.1-405b/sorting32-cot.log"
tmux new-session -d -s sorting32-tot "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task sorting32 \
                                --agent tot \
                                --tot_width 10 \
                                --tot_depth 3 \
                                &> experiments/logs/llama-3.1-405b/sorting32-tot.log"
tmux new-session -d -s sorting32-got "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task sorting32 \
                                --agent got \
                                --got_branches 2 \
                                --got_generate_attempts 5 \
                                --got_aggregate_attempts 10 \
                                --got_refine_attempts 10 \
                                &> experiments/logs/llama-3.1-405b/sorting32-got.log"
tmux new-session -d -s sorting32-llm "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting32 --agent llm &> experiments/logs/llama-3.1-405b/sorting32-llm.log"

# sorting64
tmux new-session -d -s sorting64-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting64 --agent io &> experiments/logs/llama-3.1-405b/sorting64-io.log"
tmux new-session -d -s sorting64-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting64 --agent cot &> experiments/logs/llama-3.1-405b/sorting64-cot.log"
tmux new-session -d -s sorting64-tot "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task sorting64 \
                                --agent tot \
                                --tot_width 10 \
                                --tot_depth 7 \
                                &> experiments/logs/llama-3.1-405b/sorting64-tot.log"
tmux new-session -d -s sorting64-got "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task sorting64 \
                                --agent got \
                                --got_branches 4 \
                                --got_generate_attempts 5 \
                                --got_aggregate_attempts 10 \
                                --got_refine_attempts 5 \
                                &> experiments/logs/llama-3.1-405b/sorting64-got.log"
tmux new-session -d -s sorting64-llm "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting64 --agent llm &> experiments/logs/llama-3.1-405b/sorting64-llm.log"

# sorting128
tmux new-session -d -s sorting128-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting128 --agent io &> experiments/logs/llama-3.1-405b/sorting128-io.log"
tmux new-session -d -s sorting128-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting128 --agent cot &> experiments/logs/llama-3.1-405b/sorting128-cot.log"
tmux new-session -d -s sorting128-tot "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task sorting128 \
                                --agent tot \
                                --tot_width 10 \
                                --tot_depth 10 \
                                &> experiments/logs/llama-3.1-405b/sorting128-tot.log"
tmux new-session -d -s sorting128-got "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task sorting128 \
                                --agent got \
                                --got_branches 8 \
                                --got_generate_attempts 5 \
                                --got_aggregate_attempts 10 \
                                --got_refine_attempts 5 \
                                &> experiments/logs/llama-3.1-405b/sorting128-got.log"
tmux new-session -d -s sorting128-llm "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting128 --agent llm &> experiments/logs/llama-3.1-405b/sorting128-llm.log"

# keyword_counting
tmux new-session -d -s keyword-counting-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task keyword_counting --agent io &> experiments/logs/llama-3.1-405b/keyword-counting-io.log"
tmux new-session -d -s keyword-counting-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task keyword_counting --agent cot &> experiments/logs/llama-3.1-405b/keyword-counting-cot.log"
tmux new-session -d -s keyword-counting-tot "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task keyword_counting \
                                --agent tot \
                                --tot_width 10 \
                                --tot_depth 6 \
                                &> experiments/logs/llama-3.1-405b/keyword-counting-tot.log"
tmux new-session -d -s keyword-counting-got "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task keyword_counting \
                                --agent got \
                                --got_generate_attempts 10 \
                                --got_aggregate_attempts 3 \
                                --got_refine_attempts 3 \
                                &> experiments/logs/llama-3.1-405b/keyword-counting-got.log"
tmux new-session -d -s keyword-counting-llm "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task keyword_counting --agent llm &> experiments/logs/llama-3.1-405b/keyword-counting-llm.log"

# set_intersection32
tmux new-session -d -s set-intersection32-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection32 --agent io &> experiments/logs/llama-3.1-405b/set-intersection32-io.log"
tmux new-session -d -s set-intersection32-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection32 --agent cot &> experiments/logs/llama-3.1-405b/set-intersection32-cot.log"
tmux new-session -d -s set-intersection32-tot "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task set_intersection32 \
                                --agent tot \
                                --tot_width 10 \
                                --tot_depth 3 \
                                &> experiments/logs/llama-3.1-405b/set-intersection32-tot.log"
tmux new-session -d -s set-intersection32-got "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task set_intersection32 \
                                --agent got \
                                --got_branches 2 \
                                --got_generate_attempts 5 \
                                --got_aggregate_attempts 10 \
                                &> experiments/logs/llama-3.1-405b/set-intersection32-got.log"
tmux new-session -d -s set-intersection32-llm "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection32 --agent llm &> experiments/logs/llama-3.1-405b/set-intersection32-llm.log"

# set_intersection64
tmux new-session -d -s set-intersection64-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection64 --agent io &> experiments/logs/llama-3.1-405b/set-intersection64-io.log"
tmux new-session -d -s set-intersection64-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection64 --agent cot &> experiments/logs/llama-3.1-405b/set-intersection64-cot.log"
tmux new-session -d -s set-intersection64-tot "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task set_intersection64 \
                                --agent tot \
                                --tot_width 10 \
                                --tot_depth 7 \
                                &> experiments/logs/llama-3.1-405b/set-intersection64-tot.log"
tmux new-session -d -s set-intersection64-got "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task set_intersection64 \
                                --agent got \
                                --got_branches 4 \
                                --got_generate_attempts 5 \
                                --got_aggregate_attempts 10 \
                                &> experiments/logs/llama-3.1-405b/set-intersection64-got.log"
tmux new-session -d -s set-intersection64-llm "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection64 --agent llm &> experiments/logs/llama-3.1-405b/set-intersection64-llm.log"

# set_intersection128
tmux new-session -d -s set-intersection128-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection128 --agent io &> experiments/logs/llama-3.1-405b/set-intersection128-io.log"
tmux new-session -d -s set-intersection128-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection128 --agent cot &> experiments/logs/llama-3.1-405b/set-intersection128-cot.log"
tmux new-session -d -s set-intersection128-tot "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task set_intersection128 \
                                --agent tot \
                                --tot_width 10 \
                                --tot_depth 9 \
                                &> experiments/logs/llama-3.1-405b/set-intersection128-tot.log"
tmux new-session -d -s set-intersection128-got "conda deactivate; conda activate sglang; cd reasoning-agent; \
                                python -u src/main.py \
                                --task set_intersection128 \
                                --agent got \
                                --got_branches 8 \
                                --got_generate_attempts 5 \
                                --got_aggregate_attempts 5 \
                                &> experiments/logs/llama-3.1-405b/set-intersection128-got.log"
tmux new-session -d -s set-intersection128-llm "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection128 --agent llm &> experiments/logs/llama-3.1-405b/set-intersection128-llm.log"