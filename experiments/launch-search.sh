tmux new-session -d -s search-sorting32 "conda deactivate; conda activate sglang; python src/search.py --task sorting32"
tmux new-session -d -s search-sorting64 "conda deactivate; conda activate sglang; python src/search.py --task sorting64"
tmux new-session -d -s search-sorting128 "conda deactivate; conda activate sglang; python src/search.py --task sorting128"
tmux new-session -d -s search-set-intersection32 "conda deactivate; conda activate sglang; python src/search.py --task set_intersection32"
tmux new-session -d -s search-set-intersection64 "conda deactivate; conda activate sglang; python src/search.py --task set_intersection64"
tmux new-session -d -s search-set-intersection128 "conda deactivate; conda activate sglang; python src/search.py --task set_intersection128"
tmux new-session -d -s search-keyword-counting "conda deactivate; conda activate sglang; python src/search.py --task keyword_counting"

# tmux kill-session -t search-sorting32
# tmux kill-session -t search-sorting64
# tmux kill-session -t search-sorting128
# tmux kill-session -t search-set-intersection32
# tmux kill-session -t search-set-intersection64
# tmux kill-session -t search-set-intersection128
# tmux kill-session -t search-keyword-counting