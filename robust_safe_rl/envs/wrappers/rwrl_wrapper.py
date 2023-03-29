from collections import OrderedDict
import numpy as np
import realworldrl_suite.environments as rwrl

from robust_safe_rl.envs.wrappers.dmc_wrapper import DMCWrapper

def make_rwrl_env(domain_name,task_name,env_setup_kwargs):
    """Creates Real World RL Suite task."""
    env = RWRLWrapper(domain_name,task_name,env_setup_kwargs)
    return env

class RWRLWrapper(DMCWrapper):
    """Wrapper to convert Real World RL Suite tasks to OpenAI Gym format."""

    def __init__(self,domain_name,task_name,env_setup_kwargs):
        """Initializes Real World RL Suite tasks.
        
        Args:
            domain_name (str): name of RWRL Suite domain
            task_name (str): name of RWRL Suite task
            env_setup_kwargs (dict): setup parameters
        """
        safety_spec = create_safety_spec(domain_name,env_setup_kwargs)
        perturb_spec = create_perturb_spec(env_setup_kwargs)
        noise_spec = create_noise_spec(env_setup_kwargs)
        env = rwrl.load(domain_name=domain_name,task_name=task_name,
            safety_spec=safety_spec,perturb_spec=perturb_spec,
            noise_spec=noise_spec)
        
        exclude_keys = ['constraints']
        self._setup(env,exclude_keys)
    
    def step(self,a):
        """Takes step in environment.

        Args:
            a (np.ndarray): action
        
        Returns:
            s (np.ndarray): flattened next state
            r (float): reward
            d (bool): done flag
            info (dict): dictionary with additional environment info
        """
        s, r, d, info = super(RWRLWrapper,self).step(a)
        
        constraints = info.get('constraints',np.array([True]))
        cost = 1.0 - np.all(constraints)
        info['cost'] = cost

        return s, r, d, info

# Config helper functions
#########################################

def create_safety_spec(domain_name,env_setup_kwargs):
    """Creates safety_spec dictionary."""
    safety_spec = {
        'enable':       True,
        'observations': True,
        'safety_coeff': env_setup_kwargs['safety_coeff'],
    }

    rwrl_constraints_list = env_setup_kwargs['rwrl_constraints']
    rwrl_constraints_domain = rwrl_constraints_combined[domain_name]
    if env_setup_kwargs['rwrl_constraints_all']:
        rwrl_constraints_list = list(rwrl_constraints_domain.keys())
    
    if rwrl_constraints_list:
        rwrl_constraints = OrderedDict()
        for constraint in rwrl_constraints_list:
            rwrl_constraints[constraint] = rwrl_constraints_domain[constraint]
        
        safety_spec['constraints'] = rwrl_constraints

    return safety_spec

def create_perturb_spec(env_setup_kwargs):
    """Creates perturb_spec dictionary."""
    perturb_spec = {
        'enable':       False,
        'period':       1,
        'scheduler':    'constant',
    }

    if env_setup_kwargs['perturb_param_name']:
        perturb_spec['param'] = env_setup_kwargs['perturb_param_name']
    
    perturb_min = env_setup_kwargs['perturb_param_min']
    perturb_max = env_setup_kwargs['perturb_param_max']

    if env_setup_kwargs['perturb_param_value'] is not None:
        perturb_spec['enable'] = True
        perturb_spec['start'] = env_setup_kwargs['perturb_param_value']
        perturb_spec['min'] = env_setup_kwargs['perturb_param_value']
        perturb_spec['max'] = env_setup_kwargs['perturb_param_value']
    elif (perturb_min is not None) and (perturb_max is not None):
        perturb_spec['enable'] = True
        perturb_spec['start'] = (perturb_min + perturb_max) / 2
        perturb_spec['min'] = perturb_min
        perturb_spec['max'] = perturb_max
        perturb_spec['scheduler'] = 'uniform'

    return perturb_spec

def create_noise_spec(env_setup_kwargs):
    """Creates noise_spec dictionary."""
    noise_spec = dict()

    action_noise_std = env_setup_kwargs['action_noise_std']
    observation_noise_std = env_setup_kwargs['observation_noise_std']
    if (action_noise_std > 0.0) or (observation_noise_std > 0.0):
        noise_spec['gaussian'] = {
            'enable':   True,
            'actions':  action_noise_std,
            'observations': observation_noise_std
        }
    
    return noise_spec

# RWRL Constraints
#########################################

rwrl_constraints_cartpole = {
    'slider_pos_constraint':        rwrl.cartpole.slider_pos_constraint,
    'balance_velocity_constraint':  rwrl.cartpole.balance_velocity_constraint,
    'slider_accel_constraint':      rwrl.cartpole.slider_accel_constraint,
}

rwrl_constraints_walker = {
    'joint_angle_constraint':       rwrl.walker.joint_angle_constraint,
    'joint_velocity_constraint':    rwrl.walker.joint_velocity_constraint,
    'dangerous_fall_constraint':    rwrl.walker.dangerous_fall_constraint,
    'torso_upright_constraint':     rwrl.walker.torso_upright_constraint,
}

rwrl_constraints_quadruped = {
    'joint_angle_constraint':       rwrl.quadruped.joint_angle_constraint,
    'joint_velocity_constraint':    rwrl.quadruped.joint_velocity_constraint,
    'upright_constraint':           rwrl.quadruped.upright_constraint,
    'foot_force_constraint':        rwrl.quadruped.foot_force_constraint,
}

rwrl_constraints_humanoid = {
    'joint_angle_constraint':       rwrl.humanoid.joint_angle_constraint,
    'joint_velocity_constraint':    rwrl.humanoid.joint_velocity_constraint,
    'upright_constraint':           rwrl.humanoid.upright_constraint,
    'dangerous_fall_constraint':    rwrl.humanoid.dangerous_fall_constraint,
    'foot_force_constraint':        rwrl.humanoid.foot_force_constraint,
}

rwrl_constraints_manipulator = {
    'joint_angle_constraint':       rwrl.manipulator.joint_angle_constraint,
    'joint_velocity_constraint':    rwrl.manipulator.joint_velocity_constraint,
    'joint_accel_constraint':       rwrl.manipulator.joint_accel_constraint,
    'grasp_force_constraint':       rwrl.manipulator.grasp_force_constraint,
}

rwrl_constraints_combined = {
    'cartpole':     rwrl_constraints_cartpole,
    'walker':       rwrl_constraints_walker,
    'quadruped':    rwrl_constraints_quadruped,
    'humanoid':     rwrl_constraints_humanoid,
    'manipulator':  rwrl_constraints_manipulator
}