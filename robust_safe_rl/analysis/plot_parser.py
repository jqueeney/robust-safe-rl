"""Creates command line parser for plot.py."""
import argparse

parser = argparse.ArgumentParser()

# Help
#########################################
help_kwargs = [
    'show_metrics'
]

parser.add_argument('--show_metrics',help='print list of metrics',
    action='store_true')

# Setup
#########################################
setup_kwargs = [
    'import_path','import_files','plot_type',
    'timesteps','interval','metrics','window',
]

parser.add_argument('--import_path',help='import path',
    type=str,default='./logs')
parser.add_argument('--import_files',nargs='+',
    help='list of simulation files',type=str)

parser.add_argument('--plot_type',help='training or evaluation plot',
    type=str,default='train',choices=['train','eval'])

parser.add_argument('--timesteps',help='number of steps to plot',type=float)
parser.add_argument('--interval',help='how often to plot data',
    type=float,default=1e4)
parser.add_argument('--metrics',nargs='+',
    help='list of metrics to plot',type=str,default=['J_tot'])
parser.add_argument('--window',
    help='number of steps for plot smoothing',type=int,default=1e5)

# Plot parameters
#########################################
plot_kwargs = [
    'save_path','save_file','save_type',
    'titles','labels',
    'figsize','rows','show_all','se_val',
]

parser.add_argument('--save_path',help='save path',type=str,default='./figs')
parser.add_argument('--save_file',
    help='file name to use when saving plot',type=str,default='userplot')
parser.add_argument('--save_type',help='file type when saving plot',
    type=str,default='png',choices=['png','pdf'])

parser.add_argument('--titles',nargs='+',help='list of plot titles',type=str)
parser.add_argument('--labels',nargs='+',help='list of labels',type=str)

parser.add_argument('--figsize',nargs='+',help='figure size',type=int)
parser.add_argument('--rows',help='number of rows',type=int,default=1)
parser.add_argument('--show_all',help='show plots of all seeds',
    action='store_true')
parser.add_argument('--se_val',
    help='standard error multiplier for plot shading',type=float,default=0.5)


# For export
#########################################
def create_plot_parser():
    return parser

all_plot_kwargs = {
    'help_kwargs':  help_kwargs,
    'setup_kwargs': setup_kwargs,
    'plot_kwargs':  plot_kwargs,
}