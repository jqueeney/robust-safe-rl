"""Creates command line parser for video.py."""
import argparse

parser = argparse.ArgumentParser()

# Setup
#########################################

# Shared
# ---------------------------
setup_import_kwargs = [
    'import_path','import_files','import_indices','import_adversary'
]

parser.add_argument('--import_path',help='import path',
    type=str,default='./logs')
parser.add_argument('--import_files',nargs='+',
    help='list of simulation files',type=str)
parser.add_argument('--import_indices',nargs='+',
    help='list of simulation indices',type=int)
parser.add_argument('--import_adversary',help='evaluate w/ adversary',
    action='store_true')

# RWRL parameters
# ---------------------------
setup_rwrl_kwargs = [
    'safety_coeff','rwrl_constraints','rwrl_constraints_all',
    'perturb_param_name','perturb_param_value',
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

parser.add_argument('--action_noise_std',help='Gaussian action noise std',
    type=float)
parser.add_argument('--observation_noise_std',
    help='Gaussian observation noise std',type=float)

# Combined
# ---------------------------
setup_kwargs = setup_import_kwargs + setup_rwrl_kwargs

# Simulate parameters
#########################################
sim_kwargs = [
    'seed','T','deterministic','camera_id','terminate','safety_budget','alpha'
]

parser.add_argument('--seed',help='seed for simulations',
    type=int,default=0)
parser.add_argument('--T',help='simulation horizon length',
    type=int,default=1000)
parser.add_argument('--deterministic',help='use deterministic actor',
    action='store_true')

parser.add_argument('--camera_id',help='camera ID for video',type=int)
parser.add_argument('--terminate',help='freeze simulation on termination',
    action='store_true')

parser.add_argument('--safety_budget',help='cost constraint budget',type=float,
    default=100.0)
parser.add_argument('--alpha',help='strength of safety notifications',type=float,
    default=0.3)

# Video parameters
#########################################
video_kwargs = [
    'save_path','save_file','video_type','fps','save_placeholder','image_type',
]

parser.add_argument('--save_path',help='save path',type=str,default='./videos')
parser.add_argument('--save_file',
    help='file name to use when saving video',type=str,default='uservideo')
parser.add_argument('--video_type',
    help='file type to use when saving video',type=str,default='mp4')

parser.add_argument('--fps',help='frames per second',type=int,default=60)

parser.add_argument('--save_placeholder',help='save placeholder image',
    action='store_true')
parser.add_argument('--image_type',
    help='file type to use when saving placeholder image',
    type=str,default='png',choices=['png','pdf'])

# For export
#########################################
def create_video_parser():
    return parser

all_video_kwargs = {
    'setup_kwargs': setup_kwargs,
    'sim_kwargs':   sim_kwargs,
    'video_kwargs': video_kwargs,
}