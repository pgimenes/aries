
import argparse

def get_args():
    # Read args from command line
    parser = argparse.ArgumentParser()
    
    # Operating mode
    parser.add_argument(
        "--replay",
        action="store_true",
    )

    # Main arguments
    parser.add_argument(
        "--agent", 
        type=str, 
        default="tot"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="sorting32"
    )
    parser.add_argument(
        "--start", 
        type=int, 
        default=0,
    )
    parser.add_argument(
        "--end", 
        type=int, 
        default=1000,
    )

    # LLM Agent
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--cot_sc_branches",
        type=int,
        default=1,
    )

    # GoT Agent
    parser.add_argument(
        "--got_branches", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--got_generate_attempts", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--got_aggregate_attempts", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--got_post_aggregate_keepbest", 
        action="store_true",
    )
    parser.add_argument(
        "--got_post_aggregate_refine", 
        action="store_true",
    )
    parser.add_argument(
        "--got_refine_attempts", 
        type=int, 
        default=2,
    )

    # ToT Agent
    parser.add_argument(
        "--tot_width", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--tot_depth", 
        type=int, 
        default=2,
    )

    return parser.parse_args()