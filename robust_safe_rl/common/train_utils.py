"""Helper functions for training inputs."""
import os
import pickle


def gather_inputs(args,all_kwargs):
    """Organizes inputs."""

    args_dict = vars(args)
    inputs_dict = dict()

    for key,param_list in all_kwargs.items():
        active_dict = dict()
        for param in param_list:
            active_dict[param] = args_dict[param]
        inputs_dict[key] = active_dict

    return inputs_dict

def import_inputs(inputs_dict):
    """Imports inputs from provided log file."""

    setup_dict = inputs_dict['setup_kwargs']
    
    import_path = setup_dict['import_path']
    import_file = setup_dict['import_file']
    import_idx = setup_dict['import_idx']
    import_all = setup_dict['import_all']

    train_idx = setup_dict['idx'] - setup_dict['runs_start']

    if import_path and import_file:
        import_filefull = os.path.join(import_path,import_file)
        with open(import_filefull,'rb') as f:
            import_log = pickle.load(f)
        
        if import_idx is None:
            if len(import_log) > train_idx:
                import_idx = train_idx
            else:
                import_idx = 0
        else:
            assert import_idx < len(import_log), 'import_idx too large'
        
        import_log_param = import_log[import_idx]['param']

        env_kwargs = import_log_param['env_kwargs']
        actor_kwargs = import_log_param['actor_kwargs']
        critic_kwargs = import_log_param['critic_kwargs']
        rob_kwargs = import_log_param['rob_kwargs']

        if import_all:
            inputs_dict = import_log_param
            inputs_dict['setup_kwargs'] = setup_dict
        else:
            inputs_dict['env_kwargs'] = env_kwargs
            inputs_dict['actor_kwargs'] = actor_kwargs
            inputs_dict['critic_kwargs'] = critic_kwargs
            inputs_dict['rob_kwargs'] = rob_kwargs

        import_log_final = import_log[import_idx]['final']
        
        actor_weights = import_log_final['actor_weights']
        adversary_weights = import_log_final['adversary_weights']
        critic_weights = import_log_final['critic_weights']
        cost_critic_weights = import_log_final['cost_critic_weights']
        rob_weights = import_log_final['rob_weights']
        rms_stats = import_log_final['rms_stats']
    else:
        actor_weights = None
        adversary_weights = None
        critic_weights = None
        cost_critic_weights = None
        rob_weights = None
        rms_stats = None

    inputs_dict['actor_kwargs']['actor_weights'] = actor_weights
    inputs_dict['actor_kwargs']['adversary_weights'] = adversary_weights
    inputs_dict['critic_kwargs']['critic_weights'] = critic_weights
    inputs_dict['critic_kwargs']['cost_critic_weights'] = cost_critic_weights
    inputs_dict['rob_kwargs']['rob_weights'] = rob_weights
    inputs_dict['alg_kwargs']['init_rms_stats'] = rms_stats
    
    return inputs_dict

def set_default_inputs(inputs_dict):
    """Imports default values for inputs not set on command line."""
    
    env_kwargs = inputs_dict['env_kwargs']
    env_setup_kwargs = inputs_dict['env_setup_kwargs']   
    
    # Safety
    if env_kwargs['env_type'] == 'rwrl':
        domain = env_kwargs['env_name']
        task = env_kwargs['task_name']

        if (env_setup_kwargs['safety_coeff'] is None):
            try:
                safety_coeff = rwrl_defaults_safe[domain][task]['safety_coeff']
                env_setup_kwargs['safety_coeff'] = safety_coeff
            except:
                env_setup_kwargs['safety_coeff'] = 0.30

        if (env_setup_kwargs['rwrl_constraints'] is None):
            try:
                constraints = rwrl_defaults_safe[domain][task]['rwrl_constraints']
                env_setup_kwargs['rwrl_constraints'] = constraints
            except:
                pass

    # Robustness
    rob_kwargs = inputs_dict['rob_kwargs']
    rob_setup_kwargs = inputs_dict['rob_setup_kwargs']

    if (rob_setup_kwargs['rob_magnitude'] is None):
        if rob_kwargs['robust_type'] == 'otp':
            rob_setup_kwargs['rob_magnitude'] = 0.02
        elif rob_kwargs['robust_type'] == 'ramu':
            rob_setup_kwargs['rob_magnitude'] = 0.10
    
    # Adversarial RL
    adversarial_rl = inputs_dict['setup_kwargs']['adversarial_rl']
    actor_kwargs = inputs_dict['actor_kwargs']
    rl_update_kwargs = inputs_dict['rl_update_kwargs']

    if adversarial_rl:
        if (actor_kwargs['actor_adversary_prob'] == 0.0):
            actor_kwargs['actor_adversary_prob'] = 0.10
        
        if (rl_update_kwargs['actor_adversary_freq'] == 0):
            rl_update_kwargs['actor_adversary_freq'] = 10

    # Domain Randomization
    domain_rand = inputs_dict['setup_kwargs']['domain_rand']
    domain_rand_ood = inputs_dict['setup_kwargs']['domain_rand_ood']

    if env_kwargs['env_type'] == 'rwrl':
        domain = env_kwargs['env_name']

        if domain_rand:
            try:
                perturb_name = rwrl_defaults_DR[domain]['perturb_param_name']
                perturb_min = rwrl_defaults_DR[domain]['perturb_param_min']
                perturb_max = rwrl_defaults_DR[domain]['perturb_param_max']
            except:
                perturb_name = None
                perturb_min = None
                perturb_max = None
        elif domain_rand_ood:
            try:
                perturb_name = rwrl_defaults_DR_OOD[domain]['perturb_param_name']
                perturb_min = rwrl_defaults_DR_OOD[domain]['perturb_param_min']
                perturb_max = rwrl_defaults_DR_OOD[domain]['perturb_param_max']
            except:
                perturb_name = None
                perturb_min = None
                perturb_max = None
        else:
            perturb_name = None
            perturb_min = None
            perturb_max = None

        if (env_setup_kwargs['perturb_param_name'] is None):
            env_setup_kwargs['perturb_param_name'] = perturb_name

        if (env_setup_kwargs['perturb_param_name'] == perturb_name):
            if (env_setup_kwargs['perturb_param_min'] is None):
                env_setup_kwargs['perturb_param_min'] = perturb_min

            if (env_setup_kwargs['perturb_param_max'] is None):
                env_setup_kwargs['perturb_param_max'] = perturb_max
    
    return inputs_dict

# RWRL defaults
#########################################

# Safety
######################

rwrl_defaults_safe_cartpole = {
    'realworld_swingup': {
        'safety_coeff':         0.30,
        'rwrl_constraints':     ['slider_pos_constraint'],
    }
}

rwrl_defaults_safe_walker = {
    'realworld_walk': {
        'safety_coeff':         0.25,
        'rwrl_constraints':     ['joint_velocity_constraint'],
    },
    'realworld_run': {
        'safety_coeff':         0.30,
        'rwrl_constraints':     ['joint_velocity_constraint'],
    }
}

rwrl_defaults_safe_quadruped = {
    'realworld_walk': {
        'safety_coeff':         0.15,
        'rwrl_constraints':     ['joint_angle_constraint'],
    },
    'realworld_run': {
        'safety_coeff':         0.30,
        'rwrl_constraints':     ['joint_angle_constraint'],
    }
}

rwrl_defaults_safe = {
    'cartpole':     rwrl_defaults_safe_cartpole,
    'walker':       rwrl_defaults_safe_walker,
    'quadruped':    rwrl_defaults_safe_quadruped,
}

# Domain Randomization
######################

rwrl_defaults_DR_cartpole = {
    'domain_rand': {
        'perturb_param_name':   'pole_length',
        'perturb_param_min':    0.875,
        'perturb_param_max':    1.125
    },
    'domain_rand_ood': {
        'perturb_param_name':   'pole_mass',
        'perturb_param_min':    0.05,
        'perturb_param_max':    0.15
    },
}

rwrl_defaults_DR_walker = {
    'domain_rand': {
        'perturb_param_name':   'torso_length',
        'perturb_param_min':    0.20,
        'perturb_param_max':    0.40
    },
    'domain_rand_ood': {
        'perturb_param_name':   'contact_friction',
        'perturb_param_min':    0.40,
        'perturb_param_max':    1.00
    }
}

rwrl_defaults_DR_quadruped = {
    'domain_rand': {
        'perturb_param_name':   'torso_density',
        'perturb_param_min':    750,
        'perturb_param_max':    1250
    },
    'domain_rand_ood': {
        'perturb_param_name':   'contact_friction',
        'perturb_param_min':    1.00,
        'perturb_param_max':    2.00
    }
}

rwrl_defaults_DR = {
    'cartpole':     rwrl_defaults_DR_cartpole['domain_rand'],
    'walker':       rwrl_defaults_DR_walker['domain_rand'],
    'quadruped':    rwrl_defaults_DR_quadruped['domain_rand'],
}

rwrl_defaults_DR_OOD = {
    'cartpole':     rwrl_defaults_DR_cartpole['domain_rand_ood'],
    'walker':       rwrl_defaults_DR_walker['domain_rand_ood'],
    'quadruped':    rwrl_defaults_DR_quadruped['domain_rand_ood'],
}