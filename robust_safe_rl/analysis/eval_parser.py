"""Creates command line parser for eval.py."""
import argparse

parser = argparse.ArgumentParser()

# Help
#########################################
help_kwargs = [
    'show_perturbations'
]

parser.add_argument('--show_perturbations',
    help='print perturbations and nominal values',action='store_true')

# Setup
#########################################

# Shared
# ---------------------------
setup_shared_kwargs = [
    'cores','import_path','import_file','import_adversary',
    'save_path','save_file',
]

parser.add_argument('--cores',help='number of processes',type=int)

parser.add_argument('--import_path',help='import path',type=str,
    default='./logs')
parser.add_argument('--import_file',help='import file name',type=str)
parser.add_argument('--import_adversary',help='evaluate w/ adversary',
    action='store_true')

parser.add_argument('--save_path',help='save path',type=str,default='./logs')
parser.add_argument('--save_file',help='save file name',type=str)

# RWRL perturbations
# ---------------------------
setup_rwrl_kwargs = [
    'safety_coeff','rwrl_constraints','rwrl_constraints_all',
    'perturb_param_name','perturb_param_count',
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
parser.add_argument('--perturb_param_count',help='number of perturbation values',
    type=int,default=21)
parser.add_argument('--perturb_param_min',help='min perturbation parameter',
    type=float)
parser.add_argument('--perturb_param_max',help='max perturbation parameter',
    type=float)

parser.add_argument('--action_noise_std',help='Gaussian action noise std',
    type=float)
parser.add_argument('--observation_noise_std',
    help='Gaussian observation noise std',type=float)

# Combined
# ---------------------------
setup_kwargs = setup_shared_kwargs + setup_rwrl_kwargs

# Evaluation parameters
#########################################
eval_kwargs = [
    'env_horizon','num_traj','deterministic','seed'
]

parser.add_argument('--env_horizon',
    help='rollout horizon for real trajectories',type=int,default=1000)
parser.add_argument('--num_traj',
    help='number of trajectories for eval',type=int,default=10)
parser.add_argument('--deterministic',help='use deterministic policy',
    action='store_true')
parser.add_argument('--seed',help='master seed',type=int,default=0)

# For export
#########################################
def create_eval_parser():
    return parser

all_eval_kwargs = {
    'help_kwargs':          help_kwargs,
    'setup_kwargs':         setup_kwargs,
    'eval_kwargs':          eval_kwargs,
}