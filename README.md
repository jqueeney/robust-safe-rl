# Robust and Safe Deep Reinforcement Learning Algorithms

This repository contains the official implementations for the following papers on robust and safe deep reinforcement learning (RL):

- [Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/05b63fa06784b71aab3939004e0f0a0d-Paper-Conference.pdf) (NeurIPS 2023)
- [Optimal Transport Perturbations for Safe Reinforcement Learning with Robustness Guarantees](https://openreview.net/pdf?id=cgSXpAR4Gl) (TMLR 2024)

Risk-Averse Model Uncertainty (RAMU) and Optimal Transport Perturbations (OTP) both provide robustness to environment uncertainty in safe RL, while only requiring data collected under a nominal training environment. RAMU considers a distribution over potential transition models, while OTP considers an uncertainty set of potential transition models defined using optimal transport cost.

Please consider citing our papers as follows:

```bibtex
@inproceedings{queeney_2023_ramu,
 author = {James Queeney and Mouhacine Benosman},
 title = {Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 volume = {36},
 year = {2023}
}

@article{queeney_2024_otp,
 author = {James Queeney and Erhan Can Ozcan and Ioannis Ch. Paschalidis and Christos G. Cassandras},
 title = {Optimal Transport Perturbations for Safe Reinforcement Learning with Robustness Guarantees},
 journal = {Transactions on Machine Learning Research},
 issn = {2835-8856},
 year = {2024}
}
```

## Requirements

The source code requires the following packages to be installed:

- python
- dm-control
- gym
- matplotlib
- numpy
- pillow
- realworldrl_suite
- scipy
- seaborn
- tensorflow

See the file `environment.yml` for the latest conda environment used to run our code, which can be built with conda using the command `conda env create`. Note that `environment.yml` downloads and installs the `realworldrl_suite` package from the [RWRL Suite](https://github.com/google-research/realworldrl_suite) GitHub repository.

## Training

Policies can be trained by calling `train` on the command line. This repository primarily supports training on tasks from the RWRL Suite, but training on tasks from the DeepMind Control Suite is also possible. Hyperparameters can be changed to non-default values by using the relevant options on the command line. For more information on the inputs accepted by `train`, use the `--help` option or reference `common/train_parser.py`. The results of training are saved in the `logs/` folder upon completion.

### Safety

The safe RL setting can be considered by including `--safe` on the command line. See below for an example of training a baseline safe RL policy on the Walker Walk task from the RWRL Suite:

```
python -m robust_safe_rl.train --env_type rwrl --env_name walker --task_name realworld_walk --safe
```

By default, safety is enforced using CRPO ([Xu et al., 2021](https://proceedings.mlr.press/v139/xu21a.html)). The input `--safe_type` accepts `crpo` or `lagrange`.

For the RWRL Suite tasks considered in our papers, the safety constraints that we used are loaded by default. Safety constraints can instead be set manually using the inputs `--rwrl_constraints` and `--safety_coeff`. See `envs/wrappers/rwrl_wrapper.py` and the RWRL Suite GitHub repository for more details on possible safety constraints.

### Robustness

Our robust methods can be applied by including `--robust` on the command line, and setting `--robust_type` to `ramu` or `otp`.  See below for robust versions of the previous safe RL training example using RAMU and OTP, respectively:

```
python -m robust_safe_rl.train --env_type rwrl --env_name walker --task_name realworld_walk --safe --robust --robust_type ramu
python -m robust_safe_rl.train --env_type rwrl --env_name walker --task_name realworld_walk --safe --robust --robust_type otp
```

### Baseline Comparisons

In addition to our robust methods, the repository also provides the ability to train policies using adversarial RL or domain randomization for comparison.

Adversarial RL using the action-robust PR-MDP approach ([Tessler et al., 2019](https://proceedings.mlr.press/v97/tessler19a.html)) can be run with default values from our papers with the flag `--adversarial_rl`. Alternatively, adversarial RL can be used with non-default values by manually setting `--actor_adversary_prob` and `--actor_adversary_freq`.

Domain randomization can be run using the default perturbations from the in-distribution and out-of-distribution versions in our papers using the flags `--domain_rand` and `--domain_rand_ood`, respectively. Alternatively, domain randomization can be used with non-default values by manually setting `--perturb_param_name`, `--perturb_param_min`, and `--perturb_param_max`. See `analysis/eval_utils.py` and the RWRL Suite GitHub repository for more details on possible perturbations.

## Analysis

### Evaluation

Trained policies can be evaluated across a range of perturbed test environments by calling `eval` on the command line. Evaluation requires that the name of a training file be passed to the input `--import_file`. See below for an example:

```
python -m robust_safe_rl.eval --import_file <train_filename>
```

For the RWRL Suite tasks considered in our papers, the range of perturbed test environments that we used for evaluation are loaded by default. Perturbations can instead be set manually using the inputs `--perturb_param_name`, `--perturb_param_count`, `--perturb_param_min`, and `--perturb_param_max`. See the RWRL Suite GitHub repository for more details on possible perturbations, or use the `--show_perturbations` flag to display possible perturbation parameters and their nominal values.

For more information on the inputs accepted by `eval`, use the `--help` option or reference `analysis/eval_parser.py`. The results of evaluation are saved in the `logs/` folder upon completion.

### Plots

Results from training or evaluation can be visualized by calling `plot` on the command line. Training or evaluation files can be passed to the input `--import_files`, and the metrics to be plotted can be specified using the input `--metrics`. Use the `--show_metrics` flag to display names of all metrics available for plotting. See below for examples of plotting total rewards (`J_tot`) and total costs (`Jc_tot`) during training and evaluation, respectively:

```
python -m robust_safe_rl.plot --plot_type train --metrics J_tot Jc_tot --import_files <train_filename_1> <train_filename_2> ...
python -m robust_safe_rl.plot --plot_type eval  --metrics J_tot Jc_tot --import_files <eval_filename_1> <eval_filename_2> ...
```

For more information on the inputs accepted by `plot`, use the `--help` option or reference `analysis/plot_parser.py`. Plots are saved in the `figs/` folder.

### Videos

Videos of trained policies interacting in the environment can be recorded by calling `video` on the command line. Training files can be passed to the input `--import_files`. See below for an example:

```
python -m robust_safe_rl.video --import_files <train_filename_1> <train_filename_2> ...
```

For more information on the inputs accepted by `video`, use the `--help` option or reference `analysis/video_parser.py`. Videos are saved in the `videos/` folder.