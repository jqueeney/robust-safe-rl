import numpy as np
import gym

from robust_safe_rl.envs import init_env
from robust_safe_rl.actors import init_actor
from robust_safe_rl.common.normalizer import RunningNormalizers
from robust_safe_rl.common.samplers import trajectory_sampler
from robust_safe_rl.common.seeding import init_seeds
from robust_safe_rl.common.train_parser import env_kwargs as env_kwargs_keys
from robust_safe_rl.common.train_parser import actor_kwargs as actor_kwargs_keys

def show_perturbations(import_logs):
    """Returns dictionary of possible perturbations and nominal values."""
    try:
        env_kwargs = import_logs[0]['param']['env_kwargs']
        domain = env_kwargs['env_name']
        rwrl_nominal = rwrl_defaults_nominal[domain]
    except:
        raise ValueError('evaluation not supported on this domain')
    
    return rwrl_nominal

def set_perturb_defaults(import_logs,inputs_dict):
    """Imports default perturbation values if not set on command line."""

    setup_kwargs = inputs_dict['setup_kwargs']

    perturb_param_min = setup_kwargs['perturb_param_min']
    perturb_param_max = setup_kwargs['perturb_param_max']

    if (perturb_param_min is None) and (perturb_param_max is None):
        try:
            env_kwargs = import_logs[0]['param']['env_kwargs']
            domain = env_kwargs['env_name']

            rwrl_defaults = rwrl_defaults_eval[domain]
            setup_kwargs.update(rwrl_defaults)
        except:
            raise ValueError('Must set perturb_param_min, perturb_param_max')
    
    return inputs_dict

def create_rwrl_kwargs(perturb_param_value,setup_kwargs):
    """Creates dictionary of RWRL kwargs for environment."""

    rwrl_kwargs = {
        'action_noise_std':         0.0,
        'observation_noise_std':    0.0,
        'perturb_param_min':        None,
        'perturb_param_max':        None,
    }

    if setup_kwargs['perturb_param_name'] == 'safety_coeff':
        rwrl_kwargs['safety_coeff'] = perturb_param_value
    elif setup_kwargs['perturb_param_name'] == 'action_noise':
        rwrl_kwargs['action_noise_std'] = perturb_param_value
    elif setup_kwargs['perturb_param_name'] == 'observation_noise':
        rwrl_kwargs['observation_noise_std'] = perturb_param_value
    else:
        if setup_kwargs['perturb_param_name']:
            rwrl_kwargs['perturb_param_name'] = setup_kwargs['perturb_param_name']
        if perturb_param_value is not None:
            rwrl_kwargs['perturb_param_value'] = perturb_param_value

        if setup_kwargs['safety_coeff']:
            rwrl_kwargs['safety_coeff'] = setup_kwargs['safety_coeff']
        if setup_kwargs['action_noise_std']:
            rwrl_kwargs['action_noise_std'] = setup_kwargs['action_noise_std']
        if setup_kwargs['observation_noise_std']:
            rwrl_kwargs['observation_noise_std'] = setup_kwargs['observation_noise_std']

    if setup_kwargs['rwrl_constraints_all']:
        rwrl_kwargs['rwrl_constraints_all'] = setup_kwargs['rwrl_constraints_all']
    
    if setup_kwargs['rwrl_constraints']:
        rwrl_kwargs['rwrl_constraints'] = setup_kwargs['rwrl_constraints']

    return rwrl_kwargs

def eval_setup(perturb_param_value,setup_kwargs,import_logs):
    """Sets up environment and actor."""

    rwrl_kwargs = create_rwrl_kwargs(perturb_param_value,setup_kwargs)
    
    import_objects_all = []
    for import_log in import_logs:

        import_log_param = import_log['param']
        import_log_final = import_log['final']

        env_kwargs = import_log_param['env_kwargs']
        for key in list(env_kwargs.keys()):
            if key not in env_kwargs_keys:
                env_kwargs.pop(key)

        try:
            env_setup_kwargs = import_log_param['env_setup_kwargs']
            env_setup_kwargs.update(rwrl_kwargs)
        except:
            env_setup_kwargs = rwrl_kwargs

        gamma = import_log_param['alg_kwargs']['gamma']

        actor_kwargs = import_log_param['actor_kwargs']
        for key in list(actor_kwargs.keys()):
            if key not in actor_kwargs_keys:
                actor_kwargs.pop(key)
        
        actor_weights = import_log_final['actor_weights']
        actor_kwargs['actor_weights'] = actor_weights
        if setup_kwargs['import_adversary']:
            adversary_weights = import_log_final['adversary_weights']
            actor_kwargs['adversary_weights'] = adversary_weights
        else:
            actor_kwargs['actor_adversary_prob'] = 0.0
        
        import_rms_stats = import_log_final['rms_stats']
        
        env, _ = init_env(**env_kwargs,env_setup_kwargs=env_setup_kwargs)
        actor = init_actor(env,**actor_kwargs)

        s_dim = gym.spaces.utils.flatdim(env.observation_space)
        a_dim = gym.spaces.utils.flatdim(env.action_space)

        normalizer = RunningNormalizers(s_dim,a_dim,gamma,import_rms_stats)

        actor.set_rms(normalizer)

        import_objects = {
            'env':          env,
            'actor':        actor,
            'gamma':        gamma
        }

        import_objects_all.append(import_objects)

    return import_objects_all

def evaluate(env,actor,gamma=1.00,env_horizon=1000,num_traj=5,
    deterministic=True,seed=0):
    """Evaluates performance of actor on environment."""
    init_seeds(seed,[env])
    J_tot_list = []
    Jc_tot_list = []
    Jc_vec_tot_list = []
    J_disc_list = []
    Jc_disc_list = []
    Jc_vec_disc_list = []
    for _ in range(num_traj):
        _, J_all = trajectory_sampler(env,actor,env_horizon,
            deterministic=deterministic,gamma=gamma)
        
        J_tot, Jc_tot, Jc_vec_tot, J_disc, Jc_disc, Jc_vec_disc = J_all
        J_tot_list.append(J_tot)
        Jc_tot_list.append(Jc_tot)
        Jc_vec_tot_list.append(Jc_vec_tot)
        J_disc_list.append(J_disc)
        Jc_disc_list.append(Jc_disc)
        Jc_vec_disc_list.append(Jc_vec_disc)
    
    J_tot_ave = np.mean(J_tot_list)
    Jc_tot_ave = np.mean(Jc_tot_list)
    Jc_vec_tot_ave = np.mean(Jc_vec_tot_list,axis=0)
    J_disc_ave = np.mean(J_disc_list)
    Jc_disc_ave = np.mean(Jc_disc_list)
    Jc_vec_disc_ave = np.mean(Jc_vec_disc_list,axis=0)

    return J_tot_ave, Jc_tot_ave, Jc_vec_tot_ave, J_disc_ave, Jc_disc_ave, Jc_vec_disc_ave

def evaluate_list(inputs_dict):
    """Evaluates performance of list of actors across training runs."""

    import_logs = inputs_dict['import_logs']
    perturb_param_value = inputs_dict['perturb_param_value']
    setup_kwargs = inputs_dict['setup_kwargs']
    eval_kwargs = inputs_dict['eval_kwargs']

    init_seeds(eval_kwargs['seed'])
    import_objects_all = eval_setup(perturb_param_value,setup_kwargs,import_logs)

    J_tot_all = []
    Jc_tot_all = []
    Jc_vec_tot_all = []
    J_disc_all = []
    Jc_disc_all = []
    Jc_vec_disc_all = []
    for import_objects in import_objects_all:
        env = import_objects['env']
        actor = import_objects['actor']
        gamma = import_objects['gamma']

        J_all = evaluate(env,actor,gamma,**eval_kwargs)
        (J_tot_ave, Jc_tot_ave, Jc_vec_tot_ave, J_disc_ave, Jc_disc_ave, 
            Jc_vec_disc_ave) = J_all
        J_tot_all.append(J_tot_ave)
        Jc_tot_all.append(Jc_tot_ave)
        Jc_vec_tot_all.append(Jc_vec_tot_ave)
        J_disc_all.append(J_disc_ave)
        Jc_disc_all.append(Jc_disc_ave)
        Jc_vec_disc_all.append(Jc_vec_disc_ave)
    
    J_tot_all = np.array(J_tot_all)
    Jc_tot_all = np.array(Jc_tot_all)
    Jc_vec_tot_all = np.array(Jc_vec_tot_all).T
    J_disc_all = np.array(J_disc_all)
    Jc_disc_all = np.array(Jc_disc_all)
    Jc_vec_disc_all = np.array(Jc_vec_disc_all).T

    return J_tot_all, Jc_tot_all, Jc_vec_tot_all, J_disc_all, Jc_disc_all, Jc_vec_disc_all

# RWRL Perturbations
#########################################

# Default Evaluation
######################

rwrl_defaults_eval_cartpole = {
    'perturb_param_name':       'pole_length',
    'perturb_param_min':        0.75,
    'perturb_param_max':        1.25
}

rwrl_defaults_eval_walker = {
    'perturb_param_name':       'torso_length',
    'perturb_param_min':        0.10,
    'perturb_param_max':        0.50
}

rwrl_defaults_eval_quadruped = {
    'perturb_param_name':       'torso_density',
    'perturb_param_min':        500,
    'perturb_param_max':        1500
}

rwrl_defaults_eval = {
    'cartpole':     rwrl_defaults_eval_cartpole,
    'walker':       rwrl_defaults_eval_walker,
    'quadruped':    rwrl_defaults_eval_quadruped,
}


# Nominal Values
######################

rwrl_defaults_nominal_cartpole = {
    'pole_length':          1.0,
    'pole_mass':            0.1,
    'joint_damping':        2e-6,
    'slider_damping':       5e-4,
}

rwrl_defaults_nominal_quadruped = {
    'shin_length':          0.25,
    'torso_density':        1000.,
    'joint_damping':        30.,
    'contact_friction':     1.5,
}

rwrl_defaults_nominal_walker = {
    'thigh_length':         0.225,
    'torso_length':         0.3,
    'joint_damping':        0.1,
    'contact_friction':     0.7,
}

rwrl_defaults_nominal_humanoid = {
    'contact_friction':     0.7,
    'joint_damping':        0.2,
    'head_size':            0.09,
}

rwrl_defaults_nominal_manipulator = {
    'lower_arm_length':     0.12,
    'root_damping':         2.0,
    'shoulder_damping':     1.5,
}

rwrl_defaults_nominal = {
    'cartpole':     rwrl_defaults_nominal_cartpole,
    'quadruped':    rwrl_defaults_nominal_quadruped,
    'walker':       rwrl_defaults_nominal_walker,
    'humanoid':     rwrl_defaults_nominal_humanoid,
    'manipulator':  rwrl_defaults_nominal_manipulator
}