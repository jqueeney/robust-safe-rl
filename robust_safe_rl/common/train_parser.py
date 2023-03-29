"""Creates command line parser for train.py."""
import argparse

parser = argparse.ArgumentParser()

# Setup
#########################################
setup_kwargs = [
    'runs','runs_start','cores','seed','setup_seed','sim_seed','eval_seed',
    'save_path','save_file',
    'import_path','import_file','import_idx','import_all',
    'adversarial_rl','domain_rand','domain_rand_ood'
]

parser.add_argument('--runs',help='number of trials',type=int,default=5)
parser.add_argument('--runs_start',help='starting trial index',
    type=int,default=0)
parser.add_argument('--cores',help='number of processes',type=int)

parser.add_argument('--seed',help='master seed',type=int,default=0)
parser.add_argument('--setup_seed',help='setup seed',type=int)
parser.add_argument('--sim_seed',help='simulation seed',type=int)
parser.add_argument('--eval_seed',help='simulation seed',type=int)

parser.add_argument('--save_path',help='save path',type=str,default='./logs')
parser.add_argument('--save_file',help='save file name',type=str)

parser.add_argument('--import_path',help='import path',type=str,
    default='./logs')
parser.add_argument('--import_file',help='import file name',type=str)
parser.add_argument('--import_idx',help='import index',type=int)
parser.add_argument('--import_all',help='import all params',action='store_true')

# Setup helper flags
parser.add_argument('--adversarial_rl',help='use adversarial RL (PR-MDP)',
    action='store_true')
parser.add_argument('--domain_rand',help='use domain randomization',
    action='store_true')
parser.add_argument('--domain_rand_ood',help='use OOD domain randomization',
    action='store_true')

# Environment initialization
#########################################
env_kwargs = [
    'env_type','env_name','task_name'
]

parser.add_argument('--env_type',help='environment type',type=str,
    default='rwrl')
parser.add_argument('--env_name',help='environment / domain name',type=str,
    default='walker')
parser.add_argument('--task_name',help='task name',type=str,
    default='realworld_walk')

# Environment setup
#########################################

# RWRL parameters
# ---------------------------
rwrl_kwargs = [
    'safety_coeff','rwrl_constraints','rwrl_constraints_all',
    'perturb_param_name','perturb_param_value',
    'perturb_param_min','perturb_param_max',
    'action_noise_std','observation_noise_std',
]

parser.add_argument('--safety_coeff',help='RWRL safety coefficient',type=float)
parser.add_argument('--rwrl_constraints',nargs='+',
    help='RWRL active constraint names',type=str)
parser.add_argument('--rwrl_constraints_all',help='use all RWRL constraints',
    action='store_true')

parser.add_argument('--perturb_param_name',help='perturbation parameter name',
    type=str)
parser.add_argument('--perturb_param_value',help='perturbation parameter value',
    type=float)
parser.add_argument('--perturb_param_min',help='min perturbation parameter',
    type=float)
parser.add_argument('--perturb_param_max',help='max perturbation parameter',
    type=float)

parser.add_argument('--action_noise_std',help='Gaussian action noise std',
    type=float,default=0.0)
parser.add_argument('--observation_noise_std',
    help='Gaussian observation noise std',type=float,default=0.0)

# Environment setup parameters
# ---------------------------
env_setup_kwargs = rwrl_kwargs

# Safety setup
#########################################
safety_kwargs = [
    'safe','safe_type',
    'safety_budget','env_horizon','gamma',
    'safety_lagrange_init','safety_lagrange_lr'
]

parser.add_argument('--safe',help='enable safe RL formulation',
    action='store_true')
parser.add_argument('--safe_type',help='safety update type',
    type=str,default='crpo',choices=['crpo','lagrange'])

parser.add_argument('--safety_budget',help='cost constraint budget',type=float,
    default=100.0)

parser.add_argument('--safety_lagrange_init',
    help='safety Lagrange initial value',type=float,default=0.1)
parser.add_argument('--safety_lagrange_lr',help='safety lagrange learning rate',
    type=float,default=1e-6)

# Actor initialization
#########################################
actor_kwargs = [
    'actor_layers','actor_activations','actor_init_type','actor_gain',
    'actor_layer_norm','actor_std_mult','actor_per_state_std','actor_output_norm',
    'actor_adversary_prob',
]

parser.add_argument('--actor_layers',nargs='+',
    help='list of hidden layer sizes for actor',type=int,default=[256,256,256])
parser.add_argument('--actor_activations',nargs='+',
    help='list of activations for actor',type=str,default=['elu'])
parser.add_argument('--actor_init_type',help='actor initialization type',
    type=str,default='var')
parser.add_argument('--actor_gain',
    help='mult factor for final layer of actor',type=float,default=1e-4)
parser.add_argument('--no_actor_layer_norm',
    help='do not use layer norm on first layer',
    dest='actor_layer_norm',default=True,action='store_false')
parser.add_argument('--actor_std_mult',
    help='initial policy std deviation multiplier',type=float,default=0.3)
parser.add_argument('--no_actor_per_state_std',
    help='do not use state dependent std deviation',
    dest='actor_per_state_std',default=True,action='store_false')
parser.add_argument('--actor_output_norm',
    help='normalize mean NN output magnitude',action='store_true')

parser.add_argument('--actor_adversary_prob',
    help='probability of using adversary action',type=float,default=0.0)

# Critic initialization
#########################################
critic_kwargs = [
    'critic_layers','critic_activations','critic_init_type','critic_gain',
    'critic_layer_norm',
]

parser.add_argument('--critic_layers',nargs='+',
    help='list of hidden layer sizes for critic',
    type=int,default=[256,256,256])
parser.add_argument('--critic_activations',nargs='+',
    help='list of activations for critic',type=str,default=['elu'])
parser.add_argument('--critic_init_type',help='critic initialization type',
    type=str,default='var')
parser.add_argument('--critic_gain',
    help='mult factor for final layer of critic',type=float,default=1e-4)
parser.add_argument('--no_critic_layer_norm',
    help='do not use layer norm on first layer',
    dest='critic_layer_norm',default=True,action='store_false')

# Robustness perturbation initialization
#########################################
rob_kwargs = [
    'robust_type',
    'otp_layers','otp_activations','otp_init_type','otp_gain','otp_layer_norm',
]

parser.add_argument('--robust_type',help='robustness type',type=str,
    default='ramu',choices=['otp','ramu'])

parser.add_argument('--otp_layers',nargs='+',
    help='list of hidden layer sizes for OTP NN',type=int,default=[64,64])
parser.add_argument('--otp_activations',nargs='+',
    help='list of activations for OTP NN',type=str,default=['elu'])
parser.add_argument('--otp_init_type',help='OTP NN initialization type',
    type=str,default='var')
parser.add_argument('--otp_gain',
    help='mult factor for final layer of OTP NN',type=float,default=1e-2)
parser.add_argument('--otp_layer_norm',
    help='use layer norm on first layer',action='store_true')

# Robustness perturbation setup parameters
#########################################
rob_setup_kwargs = [
    'rob_magnitude','rob_out_max','rob_reward_attitude',
    'distortion_type','distortion_param','ramu_critic_samples',
    'otp_nn_lr','otp_dual_lr','otp_dual_init'
]

# Shared
parser.add_argument('--rob_magnitude',
    help='target magnitude for robustness perturbations',type=float)
parser.add_argument('--rob_out_max',
    help='max magnitude for robustness perturbations',type=float)
parser.add_argument('--rob_reward_attitude',help='robustness reward attitude',
    type=str,default='robust',choices=['robust','neutral','optimistic'])

# RAMU
parser.add_argument('--distortion_type',help='distortion measure',
    type=str,default='wang')
parser.add_argument('--distortion_param',help='distortion measure parameter',
    type=float,default=0.75)
parser.add_argument('--ramu_critic_samples',help='RAMU samples per real sample',
    type=int,default=5)

# OTP
parser.add_argument('--otp_nn_lr',help='learning rate for OTP neural networks',
    type=float,default=1e-4)
parser.add_argument('--otp_dual_lr',help='learning rate for OTP dual variables',
    type=float,default=1e-2)
parser.add_argument('--otp_dual_init',
    help='OTP dual variables initial value',type=float,default=10.0)

# Algorithm parameters
#########################################
alg_kwargs = [
    'gamma','env_buffer_size',
    'save_path','checkpoint_file','save_freq',
    'eval_freq','eval_num_traj','eval_deterministic',
    'on_policy','rl_alg','total_timesteps',
    'env_horizon','env_batch_type','env_batch_size_init','env_batch_size',
]

parser.add_argument('--gamma',help='discount rate',type=float,default=0.99)
parser.add_argument('--env_buffer_size',help='real data buffer size',
    type=float,default=1e6)

parser.add_argument('--checkpoint_file',help='checkpoint file name',type=str,
    default='TEMPLOG')
parser.add_argument('--save_freq',help='how often to store temp files',
    type=float)

parser.add_argument('--eval_freq',help='how often to evaluate policy',
    type=float)
parser.add_argument('--eval_num_traj',
    help='number of trajectories for evaluation',type=int,default=3)
parser.add_argument('--eval_deterministic',
    help='use deterministic policies for evaluation',action='store_true')

parser.add_argument('--on_policy',help='on-policy data only',action='store_true')
parser.add_argument('--rl_alg',help='update algorithm',type=str,default='mpo')
parser.add_argument('--total_timesteps',
    help='total number of real timesteps for training',type=float,default=1e6)

parser.add_argument('--env_horizon',
    help='rollout horizon for real trajectories',type=int,default=1000)
parser.add_argument('--env_batch_type',help='real data batch type',
    type=str,default='steps',choices=['steps','traj'])
parser.add_argument('--env_batch_size_init',
    help='real data batch size (initial no-op batch)',type=int,default=10000)
parser.add_argument('--env_batch_size',
    help='real data batch size',type=int,default=3000)

# Update parameters
#########################################

# Shared
# ---------------------------
rl_shared_kwargs = [
    'robust',
    'tau','use_targ','critic_lr','actor_lr','max_grad_norm',
    'update_batch_size','updates_per_step','eval_batch_size',
    'actor_adversary_freq',
]

parser.add_argument('--robust',help='enable robustness',action='store_true')

parser.add_argument('--tau',help='actor / critic target moving average weight',
    type=float,default=5e-3)
parser.add_argument('--no_use_targ',
    help='do not use target actor / critic for updates',
    dest='use_targ',default=True,action='store_false')
parser.add_argument('--critic_lr',help='critic learning rate',
    type=float,default=1e-4)
parser.add_argument('--actor_lr',help='actor learning rate',
    type=float,default=1e-4)
parser.add_argument('--max_grad_norm',help='max policy gradient norm',
    type=float,default=0.5)

parser.add_argument('--update_batch_size',help='batch size for updates',
    type=int,default=256)
parser.add_argument('--updates_per_step',
    help='number of minibatch updates per env step',type=float,default=1)
parser.add_argument('--eval_batch_size',help='eval batch size',
    type=int,default=2560)

parser.add_argument('--actor_adversary_freq',
    help='actor adversary update frequency',type=int,default=0)

# MPO
# ---------------------------
mpo_kwargs = [
    'temp_init','act_penalty','act_temp_init',
    'temp_closed','temp_opt_type','temp_tol','temp_lr',
    'kl_separate','kl_per_dim',
    'dual_init','dual_mean_init','dual_std_init','dual_lr',
    'mpo_num_actions','mpo_delta_E','mpo_delta_E_penalty','mpo_delta_M',
    'mpo_delta_M_mean','mpo_delta_M_std',
]

parser.add_argument('--temp_init',
    help='temperature parameter initial value',type=float,default=10.0)

parser.add_argument('--no_act_penalty',help='do not use action penalty',
    dest='act_penalty',default=True,action='store_false')
parser.add_argument('--act_temp_init',
    help='action penalty temp parameter initial value',type=float,default=10.0)

parser.add_argument('--no_temp_closed',help='do not use closed form temp update',
    dest='temp_closed',default=True,action='store_false')
parser.add_argument('--temp_opt_type',help='closed form temp update type',
    type=str,default='SLSQP',choices=['SLSQP','Powell'])
parser.add_argument('--temp_tol',help='closed form temp update tolerance',
    type=float,default=1e-3)
parser.add_argument('--temp_lr',help='temperature parameter learning rate',
    type=float,default=1e-2)

parser.add_argument('--no_kl_separate',help='do not use separate mean and std KL',
    dest='kl_separate',default=True,action='store_false')
parser.add_argument('--no_kl_per_dim',help='do not use per dimension KL',
    dest='kl_per_dim',default=True,action='store_false')
parser.add_argument('--dual_init',
    help='dual KL parameter initial value',type=float,default=10.0)
parser.add_argument('--dual_mean_init',
    help='dual mean KL parameter initial value',type=float,default=10.0)
parser.add_argument('--dual_std_init',
    help='dual std KL parameter initial value',type=float,default=100.0)
parser.add_argument('--dual_lr',help='dual parameter learning rate',
    type=float,default=1e-2)

parser.add_argument('--mpo_num_actions',
    help='number of action samples per state',type=int,default=20)

parser.add_argument('--mpo_delta_E',help='KL expectation parameter',
    type=float,default=1e-1)
parser.add_argument('--mpo_delta_E_penalty',
    help='KL action penalty expectation parameter',type=float,default=1e-3)
parser.add_argument('--mpo_delta_M',help='KL maximization parameter',
    type=float,default=1e-2)
parser.add_argument('--mpo_delta_M_mean',help='KL mean maximization parameter',
    type=float,default=1e-2)
parser.add_argument('--mpo_delta_M_std',help='KL std maximization parameter',
    type=float,default=1e-5)

# Combined
# ---------------------------
rl_update_kwargs = rl_shared_kwargs + mpo_kwargs

# For export to train.py
#########################################
def create_train_parser():
    return parser

all_kwargs = {
    'setup_kwargs':         setup_kwargs,
    'env_kwargs':           env_kwargs,
    'env_setup_kwargs':     env_setup_kwargs,
    'safety_kwargs':        safety_kwargs,
    'actor_kwargs':         actor_kwargs,
    'critic_kwargs':        critic_kwargs,
    'rob_kwargs':           rob_kwargs,
    'rob_setup_kwargs':     rob_setup_kwargs,
    'alg_kwargs':           alg_kwargs,
    'rl_update_kwargs':     rl_update_kwargs,
}