from .base import PolicyAgent
from .cot import CoTAgent
from .got import GoTAgent
from .io import IOAgent
from .llm import LLMAgent
from .tot import ToTAgent

def get_agent(agent, env, task, args, **kwargs):
    if agent == "io":
        return IOAgent(
            env=env,
            task = task,
        )
    
    if agent == "cot":
        return CoTAgent(
            env=env,
            task=task,
        )
    
    if agent == "tot":
        return ToTAgent(
            env = env,
            task = task,
            model = args.model,
            problem_definition = task.problem_definition,
            actions = task.actions,
            width = kwargs.get("tot_width"),
            depth = kwargs.get("tot_depth"),
        )
    
    if agent == "got":
        return GoTAgent(
            env=env,
            task = task,

            # GoT parameters
            branches=kwargs.get("got_branches"),
            generate_attempts=kwargs.get("got_generate_attempts"),
            aggregate_attempts=kwargs.get("got_aggregate_attempts"),
            post_aggregate_keepbest=kwargs.get("got_post_aggregate_keepbest"),
            post_aggregate_refine=kwargs.get("got_post_aggregate_refine"),
            refine_attempts=kwargs.get("got_refine_attempts"), 
        )
    
    if agent == "llm":
        return LLMAgent(
            env=env,
            task = task,
            model = args.model,
            problem_definition = task.problem_definition,
            actions = task.actions,
            max_iterations = kwargs.get("max_iterations", 25),
            cot_sc_branches=args.cot_sc_branches,
        )