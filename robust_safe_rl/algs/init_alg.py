"""Interface to algorithms."""
from robust_safe_rl.algs.base_alg import BaseAlg

def init_alg(idx,env,env_eval,actor,critic,cost_critic,rob_net,
    alg_kwargs,safety_kwargs,rl_update_kwargs):
    """Initializes algorithm."""

    alg = BaseAlg(idx,env,env_eval,actor,critic,cost_critic,rob_net,
        alg_kwargs,safety_kwargs,rl_update_kwargs)
        
    return alg