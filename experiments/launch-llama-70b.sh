tmux new-session -d -s sorting32-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting32 --method io &> experiments/logs/llama-3.1-70b/sorting32-io.log"
tmux new-session -d -s sorting32-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting32 --method cot &> experiments/logs/llama-3.1-70b/sorting32-cot.log"
tmux new-session -d -s sorting32-tot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting32 --method tot --tot_width 10 --tot_depth 3 &> experiments/logs/llama-3.1-70b/sorting32-tot.log"

tmux new-session -d -s sorting64-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting64 --method io &> experiments/logs/llama-3.1-70b/sorting64-io.log"
tmux new-session -d -s sorting64-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting64 --method cot &> experiments/logs/llama-3.1-70b/sorting64-cot.log"
tmux new-session -d -s sorting64-tot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting64 --method tot --tot_width 10 --tot_depth 7 &> experiments/logs/llama-3.1-70b/sorting64-tot.log"

tmux new-session -d -s sorting128-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting128 --method io &> experiments/logs/llama-3.1-70b/sorting128-io.log"
tmux new-session -d -s sorting128-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting128 --method cot &> experiments/logs/llama-3.1-70b/sorting128-cot.log"
tmux new-session -d -s sorting128-tot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task sorting128 --method tot --tot_width 10 --tot_depth 10 &> experiments/logs/llama-3.1-70b/sorting128-tot.log"

tmux new-session -d -s keyword-counting-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task keyword_counting --method io &> experiments/logs/llama-3.1-70b/keyword-counting-io.log"
tmux new-session -d -s keyword-counting-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task keyword_counting --method cot &> experiments/logs/llama-3.1-70b/keyword-counting-cot.log"
tmux new-session -d -s keyword-counting-tot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task keyword_counting --method tot --tot_width 10 --tot_depth 6 &> experiments/logs/llama-3.1-70b/keyword-counting-tot.log"

tmux new-session -d -s set-intersection32-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection32 --method io &> experiments/logs/llama-3.1-70b/set-intersection32-io.log"
tmux new-session -d -s set-intersection32-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection32 --method cot &> experiments/logs/llama-3.1-70b/set-intersection32-cot.log"
tmux new-session -d -s set-intersection32-tot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection32 --method tot --tot_width 10 --tot_depth 3 &> experiments/logs/llama-3.1-70b/set-intersection32-tot.log"

tmux new-session -d -s set-intersection64-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection64 --method io &> experiments/logs/llama-3.1-70b/set-intersection64-io.log"
tmux new-session -d -s set-intersection64-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection64 --method cot &> experiments/logs/llama-3.1-70b/set-intersection64-cot.log"
tmux new-session -d -s set-intersection64-tot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection64 --method tot --tot_width 10 --tot_depth 7 &> experiments/logs/llama-3.1-70b/set-intersection64-tot.log"

tmux new-session -d -s set-intersection128-io "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection128 --method io &> experiments/logs/llama-3.1-70b/set-intersection128-io.log"
tmux new-session -d -s set-intersection128-cot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection128 --method cot &> experiments/logs/llama-3.1-70b/set-intersection128-cot.log"
tmux new-session -d -s set-intersection128-tot "conda deactivate; conda activate sglang; cd reasoning-agent; python -u src/main.py --task set_intersection128 --method tot --tot_width 10 --tot_depth 9 &> experiments/logs/llama-3.1-70b/set-intersection128-tot.log"