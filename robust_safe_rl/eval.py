"""Entry point for evaluation."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from datetime import datetime
import pickle
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import copy

from robust_safe_rl.common.train_utils import gather_inputs
from robust_safe_rl.analysis.eval_parser import create_eval_parser, all_eval_kwargs
from robust_safe_rl.analysis.eval_utils import show_perturbations
from robust_safe_rl.analysis.eval_utils import set_perturb_defaults, evaluate_list

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def main():
    """Parses inputs, runs evaluations, saves data."""
    start_time = datetime.now()
    
    parser = create_eval_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args,all_eval_kwargs)

    import_filefull = os.path.join(args.import_path,args.import_file)
    with open(import_filefull,'rb') as f:
        import_logs = pickle.load(f)
    
    if args.show_perturbations:
        perturb_nominal = show_perturbations(import_logs)
        
        print('\n')
        print('Perturbation Names and Nominal Values:')
        print('--------------------------------------')
        print(perturb_nominal)
    else:
        inputs_dict = set_perturb_defaults(import_logs,inputs_dict)
        inputs_dict['import_logs'] = import_logs

        # Create range of perturbation values
        setup_kwargs = inputs_dict['setup_kwargs']
        perturb_param_min = setup_kwargs['perturb_param_min']
        perturb_param_max = setup_kwargs['perturb_param_max']
        perturb_param_count = setup_kwargs['perturb_param_count']
        
        perturb_param_values = np.linspace(
            perturb_param_min,perturb_param_max,perturb_param_count)

        # Create input list and run evaluations
        inputs_list = []
        for value in perturb_param_values:
            inputs_dict['perturb_param_value'] = value
            inputs_list.append(copy.deepcopy(inputs_dict))

        if args.cores is None:
            cpu_count = mp.cpu_count()
            if args.perturb_param_count > cpu_count:
                raise ValueError((
                    "WARNING. Number of test environments (%d) "
                    "exceeds number of CPUs (%d). "
                    "Specify number of parallel processes using --cores. "
                    "CPU and GPU memory should also be considered when "
                    "setting --cores."
                    )%(args.perturb_param_count,cpu_count)
                )
            else:
                args.cores = args.perturb_param_count

        with mp.get_context('spawn').Pool(args.cores) as pool:
            out_list = pool.map(evaluate_list,inputs_list)
        (J_tot_list, Jc_tot_list, Jc_vec_tot_list, J_disc_list, Jc_disc_list, 
            Jc_vec_disc_list) = zip(*out_list)

        J_tot_all = np.moveaxis(np.array(J_tot_list),-1,0)
        Jc_tot_all = np.moveaxis(np.array(Jc_tot_list),-1,0)
        Jc_vec_tot_all = np.moveaxis(np.array(Jc_vec_tot_list),-1,0)
        J_disc_all = np.moveaxis(np.array(J_disc_list),-1,0)
        Jc_disc_all = np.moveaxis(np.array(Jc_disc_list),-1,0)
        Jc_vec_disc_all = np.moveaxis(np.array(Jc_vec_disc_list),-1,0)

        output = {
            'eval': {
                'J_tot':                J_tot_all,
                'Jc_tot':               Jc_tot_all,
                'Jc_vec_tot':           Jc_vec_tot_all,
                'J_disc':               J_disc_all,
                'Jc_disc':              Jc_disc_all,
                'Jc_vec_disc':          Jc_vec_disc_all,
                'perturb_param_values': perturb_param_values,            
            },
            'param': vars(args)
        }

        # Save data
        if args.save_file is None:
            save_file = 'EVAL__%s'%(args.import_file)
        else:
            save_file = 'EVAL_%s__%s'%(args.save_file,args.import_file)

        os.makedirs(args.save_path,exist_ok=True)
        save_filefull = os.path.join(args.save_path,save_file)

        with open(save_filefull,'wb') as f:
            pickle.dump(output,f)
        
        end_time = datetime.now()
        print('Time Elapsed: %s'%(end_time-start_time))

if __name__=='__main__':
    main()