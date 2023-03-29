"""Entry point for plotting results."""
from robust_safe_rl.common.train_utils import gather_inputs
from robust_safe_rl.analysis.plot_parser import create_plot_parser, all_plot_kwargs
from robust_safe_rl.analysis.plot_utils import show_metrics
from robust_safe_rl.analysis.plot_utils import plot_setup, create_plot

def main():
    """Parses inputs, creates and saves plot."""
    parser = create_plot_parser()
    args = parser.parse_args()
    inputs_dict = gather_inputs(args,all_plot_kwargs)

    setup_kwargs = inputs_dict['setup_kwargs']
    plot_kwargs = inputs_dict['plot_kwargs']

    if args.show_metrics:
        metrics = show_metrics(**setup_kwargs)
        
        print('\n')
        print('Available Metrics:')
        print('------------------')
        print(metrics)
    else:
        x_list, metrics_list = plot_setup(**setup_kwargs)

        if plot_kwargs['titles'] is None:
            plot_kwargs['titles'] = setup_kwargs['metrics']

        if setup_kwargs['plot_type'] == 'train':
            plot_kwargs['x_label'] = 'Steps (M)'
        elif setup_kwargs['plot_type'] == 'eval':
            plot_kwargs['x_label'] = 'Perturbation Value'

        create_plot(x_list,metrics_list,**plot_kwargs)    


if __name__=='__main__':
    main()