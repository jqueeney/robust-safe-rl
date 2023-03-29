"""Interface to environments."""
import copy

def init_env(env_type,env_name,task_name,env_setup_kwargs):
    """Creates environments.
    
    Args:
        env_type (str): environment type (rwrl, dmc)
        env_name (str): environment / domain name
        task_name (str): task name
        env_setup_kwargs (dict): setup parameters
    
    Returns:
        Training environment and evaluation environment.
    """

    env_eval_setup_kwargs = copy.deepcopy(env_setup_kwargs)
    env_eval_setup_kwargs['perturb_param_min'] = None
    env_eval_setup_kwargs['perturb_param_max'] = None

    if env_type == 'rwrl':
        from robust_safe_rl.envs.wrappers.rwrl_wrapper import make_rwrl_env
        env = make_rwrl_env(env_name,task_name,env_setup_kwargs)
        env_eval_nominal = make_rwrl_env(env_name,task_name,env_eval_setup_kwargs)
    elif env_type == 'dmc':
        from robust_safe_rl.envs.wrappers.dmc_wrapper import make_dmc_env
        env = make_dmc_env(env_name,task_name)
        env_eval_nominal = make_dmc_env(env_name,task_name)
    else:
        raise ValueError('Supported env_type: rwrl, dmc')
    
    return env, env_eval_nominal