"""Interface to update algorithms."""
from robust_safe_rl.algs.update_algs.mpo import MPO

def init_update_alg(rl_alg_name,actor,critic,cost_critic,rob_net,
    rl_update_kwargs,safety_kwargs):
    """Initializes update algorithm."""

    inputs = (actor,critic,cost_critic,rob_net,rl_update_kwargs,safety_kwargs)
    
    if rl_alg_name == 'mpo':
        rl_alg = MPO(*inputs)
    else:
        raise ValueError('invalid rl_alg')
        
    return rl_alg