import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

sns.set()
sns.set_context('paper')

# Setup
#########################################

def aggregate_train(x,steps_window,results,metric='J_tot',steps_metric='steps'):
    """Aggregates training data."""

    metric_all = []
    for idx in range(len(results)):
        log = results[idx]['train']
        
        try:
            steps = log[steps_metric]
            log_metric = np.squeeze(log[metric])
        except:
            steps = log['steps']
            log_metric = np.ones_like(steps) * np.inf

        samples = np.cumsum(steps)
        if x is None:
            x = samples
        batch_size = steps[1]
        window = int(np.maximum(steps_window / batch_size,1))
        
        x_filter = np.argmax(np.expand_dims(samples,1) 
            >= np.expand_dims(x,0),0)           
        
        if len(log_metric.shape) == 1:
            metric_total = log_metric*steps
        else:
            metric_total = log_metric*np.expand_dims(steps,1)
        
        if window > 1:
            steps_totsmooth = np.convolve(steps,
                np.ones(window),'full')[:-(window-1)]

            if len(metric_total.shape) == 1:
                metric_totsmooth = np.convolve(metric_total,
                    np.ones(window),'full')[:-(window-1)]

                metric_ave = metric_totsmooth / steps_totsmooth
                metric_ave_filtered = metric_ave[x_filter]
            else:
                metric_ave_filtered = []
                for metric_col in metric_total.T:
                    metric_totsmooth = np.convolve(metric_col,
                        np.ones(window),'full')[:-(window-1)]

                    metric_ave_col = metric_totsmooth / steps_totsmooth
                    metric_ave_col_filtered = metric_ave_col[x_filter]

                    metric_ave_filtered.append(metric_ave_col_filtered)
                metric_ave_filtered = np.array(metric_ave_filtered).T
        else:
            metric_ave = log_metric
            metric_ave_filtered = metric_ave[x_filter]
        
        metric_all.append(metric_ave_filtered)

        x_M = x / 1e6
    
    return x_M, np.array(metric_all)

def aggregate_eval(results,metric='J_tot'):
    """Aggregates evaluation data."""

    x = results['eval']['perturb_param_values']
    metric_all = results['eval'][metric]

    return x, metric_all

def open_and_aggregate(import_path,import_file,plot_type,x,window,metrics):
    """Returns aggregated data from raw filename."""

    import_filefull = os.path.join(import_path,import_file)
    with open(import_filefull,'rb') as f:
        import_logs = pickle.load(f)
    
    x_out_all = []
    metrics_out_all = []
    
    for metric in metrics:
        if plot_type == 'train':
            if metric in metrics_steps_eval:
                steps_metric = 'steps_eval'
            else:
                steps_metric = 'steps'
        
            x_out, metric_out = aggregate_train(x,window,import_logs,
                metric,steps_metric)
        elif plot_type == 'eval':
            x_out, metric_out = aggregate_eval(import_logs,metric)
        
        x_out_all.append(x_out)
        metrics_out_all.append(metric_out)
    
    return x_out_all, metrics_out_all

metrics_steps_eval = [
    'J_tot_eval_nominal','Jc_tot_eval_nominal','Jc_vec_tot_eval_nominal',
    'J_disc_eval_nominal','Jc_disc_eval_nominal','Jc_vec_disc_eval_nominal'
]

def plot_setup(import_path,import_files,plot_type,
    timesteps,interval,window,metrics):
    """Returns aggregated data for plotting."""

    if timesteps:
        x = np.arange(0,timesteps+1,interval)
    else:
        x = None
    
    x_list = []
    metrics_list = []
    for import_file in import_files:
        x_out, metrics_out = open_and_aggregate(import_path,import_file,
            plot_type,x,window,metrics)
        x_list.append(x_out)
        metrics_list.append(metrics_out)
    
    return x_list, metrics_list

def show_metrics(import_path,import_files,plot_type,**kwargs):
    """Returns list of metrics."""

    import_filefull = os.path.join(import_path,import_files[0])
    with open(import_filefull,'rb') as f:
        import_logs = pickle.load(f)
    
    if plot_type == 'train':
        train_dict = import_logs[0]['train']
        metrics = train_dict.keys()
    elif plot_type == 'eval':
        eval_dict = import_logs['eval']
        metrics = eval_dict.keys()
    
    return list(metrics)

# Plots
#########################################

def se_shading(data_active,se_val):
    """Returns low and high values for shading."""
    data_mean = np.mean(data_active,axis=0)
    if data_active.shape[0] > 1:
        data_std = np.std(data_active,axis=0,ddof=1)
        data_se = data_std / np.sqrt(data_active.shape[0])
    else:
        data_se = np.zeros_like(data_mean)
    
    low = data_mean - se_val * data_se
    high = data_mean + se_val * data_se
    
    return low, high

def create_plot(x_list,metrics_list,
    save_path,save_file,save_type,
    x_label,titles,labels,figsize,rows,show_all,se_val):
    """Creates and saves plot."""

    tot = len(metrics_list[0])
    assert rows <= tot, 'number of rows larger than number of metrics'
    cols = int(np.ceil(tot / rows))

    if figsize is None:
        pass
    elif len(figsize) >= 2:
        figsize = tuple(figsize[:2])
    elif len(figsize) == 1:
        figsize = (figsize[0],figsize[0])

    fig, ax = plt.subplots(rows,cols,figsize=figsize)

    for plot_idx in range(tot):
        row_active = int(plot_idx / cols)
        col_active = plot_idx % cols
        tot_active = plot_idx
        
        if tot == 1:
            ax_active = ax
        elif rows == 1 or cols == 1:
            ax_active = ax[tot_active]
        else:
            ax_active = ax[(row_active,col_active)]

        if show_all:
            for file_idx in range(len(metrics_list)):
                x_active = x_list[file_idx][tot_active]
                data_active = metrics_list[file_idx][tot_active]

                for datum in data_active:
                    ax_active.plot(x_active,datum,
                        color='C%d'%file_idx,alpha=0.5)

        for file_idx in range(len(metrics_list)):
            x_active = x_list[file_idx][tot_active]
            data_active = metrics_list[file_idx][tot_active]
            try:
                label_active = labels[file_idx]
            except:
                label_active = 'File %d'%file_idx

            data_mean = np.mean(data_active,axis=0)
            ax_active.plot(x_active,data_mean,
                color='C%d'%file_idx,label=label_active)

            if se_val > 0.0:
                if len(data_active.shape) == 3:
                    for datum_idx in range(data_active.shape[-1]):
                        data_sub = data_active[:,:,datum_idx]
                        data_low, data_high = se_shading(data_sub,se_val)
                        ax_active.fill_between(x_active,data_low,data_high,
                            color='C%d'%file_idx,alpha=0.2)
                elif len(data_active.shape)==2:
                    data_low, data_high = se_shading(data_active,se_val)
                    ax_active.fill_between(x_active,data_low,data_high,
                        color='C%d'%file_idx,alpha=0.2)
                else:
                    raise ValueError('Data is wrong shape')

        if row_active == rows-1:
            ax_active.set_xlabel(x_label)
        if col_active == 0:
            ax_active.set_ylabel('Metric')
        if row_active == 0:
            handles, labels = ax_active.get_legend_handles_labels()
            temp = {k:v for k,v in zip(labels, handles)}
            if col_active == 0:
                ax_active.legend(list(temp.values()), list(temp.keys()))

        ax_active.set_title(titles[tot_active])

    # Save plot
    os.makedirs(save_path,exist_ok=True)
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    save_file = '%s_%s.%s'%(save_file,save_date,save_type)
    save_filefull = os.path.join(save_path,save_file)

    plt.subplots_adjust(hspace=0.4)
    fig.savefig(save_filefull,bbox_inches='tight',dpi=300)
    print('\nPlot Saved at {}\n'.format(save_filefull))