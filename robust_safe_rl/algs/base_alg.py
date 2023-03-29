import numpy as np
import gym

from robust_safe_rl.common.normalizer import RunningNormalizers
from robust_safe_rl.common.buffers import init_buffer
from robust_safe_rl.common.samplers import trajectory_sampler
from robust_safe_rl.common.logger import Logger
from robust_safe_rl.algs.update_algs import init_update_alg

class BaseAlg:
    """Base class for training algorithms."""

    def __init__(self,idx,env,env_eval,actor,critic,cost_critic,rob_net,
        alg_kwargs,safety_kwargs,rl_update_kwargs):
        """Initializes BaseAlg class.

        Args:
            idx (int): index for checkpoint file
            env (object): training environment
            env_eval (object): evaluation environment
            actor (object): policy
            critic (object): value function
            cost_critic (object): cost value function
            rob_net (object): robustness perturbation network
            alg_kwargs (dict): algorithm parameters
            safety_kwargs (dict): safety parameters
            rl_update_kwargs (dict): update algorithm parameters
        """
        self._setup(alg_kwargs,safety_kwargs)

        self.logger = Logger()
        self.checkpoint_name = '%s_idx%d'%(self.checkpoint_file,idx)

        self.env = env
        self.env_eval = env_eval
        self.actor = actor
        self.critic = critic
        self.cost_critic = cost_critic
        self.rob_net = rob_net

        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)

        self.env_data = init_buffer(self.s_dim,self.a_dim,self.gamma,
            self.env_buffer_size)

        self.normalizer = RunningNormalizers(self.s_dim,self.a_dim,self.gamma,
            self.init_rms_stats)
        
        self.rl_alg = init_update_alg(self.rl_alg_name,
            self.actor,self.critic,self.cost_critic,self.rob_net,
            rl_update_kwargs,safety_kwargs)

    def _setup(self,alg_kwargs,safety_kwargs):
        """Sets up hyperparameters as class attributes."""
        
        self.gamma = alg_kwargs['gamma']

        self.env_buffer_size = alg_kwargs['env_buffer_size']
        if self.env_buffer_size:
            self.env_buffer_size = int(self.env_buffer_size)
        
        self.init_rms_stats = alg_kwargs['init_rms_stats']

        self.save_path = alg_kwargs['save_path']
        self.checkpoint_file = alg_kwargs['checkpoint_file']
        self.save_freq = alg_kwargs['save_freq']

        self.last_eval = 0
        self.eval_freq = alg_kwargs['eval_freq']
        self.eval_num_traj = alg_kwargs['eval_num_traj']
        self.eval_deterministic = alg_kwargs['eval_deterministic']
        
        self.on_policy = alg_kwargs['on_policy']
        self.rl_alg_name = alg_kwargs['rl_alg']

        self.env_horizon = alg_kwargs['env_horizon']
        self.env_batch_type = alg_kwargs['env_batch_type']
        self.env_batch_size_init = alg_kwargs['env_batch_size_init']
        self.env_batch_size = alg_kwargs['env_batch_size']

        self.safe = safety_kwargs['safe']

    def _collect_env_data(self,num_timesteps):
        """Collects data by interacting with the training environment."""
        if num_timesteps == 0:
            batch_size = self.env_batch_size_init + self.env_batch_size
            batch_size_noop = self.env_batch_size_init
        else:
            batch_size = self.env_batch_size
            batch_size_noop = 0
        
        J_tot_all = []
        Jc_tot_all = []
        Jc_vec_tot_all = []
        J_disc_all = []
        Jc_disc_all = []
        Jc_vec_disc_all = []

        steps_start = self.env_data.steps_total
        traj_start = self.env_data.traj_total

        batch_size_cur = 0
        steps_noop = 0
        while batch_size_cur < batch_size:
            if self.env_batch_type == 'steps':
                horizon = np.minimum(batch_size-batch_size_cur,
                    self.env_horizon)
            else:
                horizon = self.env_horizon

            rollout_traj, J_all = trajectory_sampler(
                self.env,self.actor,horizon,gamma=self.gamma)
            (s_traj, a_traj, r_traj, sp_traj, d_traj, c_traj, 
                r_raw_traj) = rollout_traj
            J_tot, Jc_tot, Jc_vec_tot, J_disc, Jc_disc, Jc_vec_disc = J_all
            self.normalizer.update_rms(s_traj,a_traj,r_traj,sp_traj,c_traj,
                r_raw_traj)
            self.env_data.add(s_traj,a_traj,r_traj,sp_traj,d_traj,c_traj,
                r_raw_traj)

            if horizon == self.env_horizon:
                J_tot_all.append(J_tot)
                Jc_tot_all.append(Jc_tot)
                Jc_vec_tot_all.append(Jc_vec_tot)
                J_disc_all.append(J_disc)
                Jc_disc_all.append(Jc_disc)
                Jc_vec_disc_all.append(Jc_vec_disc)
            
            if batch_size_cur < batch_size_noop:
                steps_cur = self.env_data.steps_total
                steps_noop = steps_cur - steps_start

            if self.env_batch_type == 'steps':
                steps_cur = self.env_data.steps_total
                batch_size_cur = steps_cur - steps_start
            else:
                traj_cur = self.env_data.traj_total
                batch_size_cur = traj_cur - traj_start

        steps_end = self.env_data.steps_total
        traj_end = self.env_data.traj_total

        steps_new = steps_end - steps_start
        traj_new = traj_end - traj_start

        J_tot_ave = np.mean(J_tot_all)
        Jc_tot_ave = np.mean(Jc_tot_all)
        Jc_vec_tot_ave = np.mean(Jc_vec_tot_all,axis=0)
        J_disc_ave = np.mean(J_disc_all)
        Jc_disc_ave = np.mean(Jc_disc_all)
        Jc_vec_disc_ave = np.mean(Jc_vec_disc_all,axis=0)
        
        log_env_data = {
            'J_tot':            J_tot_ave,
            'Jc_tot':           Jc_tot_ave,
            'Jc_vec_tot':       Jc_vec_tot_ave,
            'J_disc':           J_disc_ave,
            'Jc_disc':          Jc_disc_ave,
            'Jc_vec_disc':      Jc_vec_disc_ave,
            'steps':            steps_new,
            'traj':             traj_new,
        }
        self.logger.log_train(log_env_data)

        return steps_new, steps_noop
    
    def _evaluate(self,num_timesteps):
        """Evaluates current policy."""
        log_env_eval_data = dict()

        if self.eval_num_traj > 0:
            J_tot_all_nominal = []
            Jc_tot_all_nominal = []
            Jc_vec_tot_all_nominal = []
            J_disc_all_nominal = []
            Jc_disc_all_nominal = []
            Jc_vec_disc_all_nominal = []
            for _ in range(self.eval_num_traj):
                _, J_all = trajectory_sampler(self.env_eval,self.actor,
                    self.env_horizon,deterministic=self.eval_deterministic,
                    gamma=self.gamma)
                J_tot, Jc_tot, Jc_vec_tot, J_disc, Jc_disc, Jc_vec_disc = J_all
                J_tot_all_nominal.append(J_tot)
                Jc_tot_all_nominal.append(Jc_tot)
                Jc_vec_tot_all_nominal.append(Jc_vec_tot)
                J_disc_all_nominal.append(J_disc)
                Jc_disc_all_nominal.append(Jc_disc)
                Jc_vec_disc_all_nominal.append(Jc_vec_disc)
            
            J_tot_ave_nominal = np.mean(J_tot_all_nominal)
            Jc_tot_ave_nominal = np.mean(Jc_tot_all_nominal)
            Jc_vec_tot_ave_nominal = np.mean(Jc_vec_tot_all_nominal,axis=0)
            J_disc_ave_nominal = np.mean(J_disc_all_nominal)
            Jc_disc_ave_nominal = np.mean(Jc_disc_all_nominal)
            Jc_vec_disc_ave_nominal = np.mean(Jc_vec_disc_all_nominal,axis=0)
            
            log_env_eval_data['J_tot_eval_nominal'] = J_tot_ave_nominal
            log_env_eval_data['Jc_tot_eval_nominal'] = Jc_tot_ave_nominal
            log_env_eval_data['Jc_vec_tot_eval_nominal'] = Jc_vec_tot_ave_nominal
            log_env_eval_data['J_disc_eval_nominal'] = J_disc_ave_nominal
            log_env_eval_data['Jc_disc_eval_nominal'] = Jc_disc_ave_nominal
            log_env_eval_data['Jc_vec_disc_eval_nominal'] = Jc_vec_disc_ave_nominal
        
        log_env_eval_data['steps_eval'] = num_timesteps - self.last_eval
        
        self.logger.log_train(log_env_eval_data)

        self.last_eval = num_timesteps

    def _set_rms(self):
        """Shares normalizers with all relevant classes."""
        self.actor.set_rms(self.normalizer)
        self.critic.set_rms(self.normalizer)
        self.cost_critic.set_rms(self.normalizer)
        self.rob_net.set_rms(self.normalizer)

    def _update(self,steps_new):
        """Updates actor and critic."""       
        logs_ac = self.rl_alg.update_actor_critic(self.env_data,steps_new)
        log_actor, log_critic, log_cost_critic = logs_ac

        self.logger.log_train(log_actor)
        self.logger.log_train(log_critic)

        if self.safe:
            self.logger.log_train(log_cost_critic,prefix='cost')
        
        if self.on_policy:
            self.env_data.reset()

    def train(self,total_timesteps,params):
        """Training loop.

        Args:
            total_timesteps (int): number of environment steps for training
            params (dict): dictionary of input parameters for logging

        Returns:
            Name of checkpoint file.
        """

        self._set_rms()
        
        checkpt_idx = 0
        if self.save_freq is None:
            checkpoints = np.array([total_timesteps])
        else:
            checkpoints = np.concatenate(
                (np.arange(0,total_timesteps,self.save_freq)[1:],
                [total_timesteps]))

        eval_idx = 0
        if self.eval_freq is None:
            evaluate = False 
        else:
            evaluate = True
            eval_points = np.concatenate(
                (np.arange(0,total_timesteps,self.eval_freq)[1:],
                [total_timesteps]))

        # Training loop
        num_timesteps = 0
        if evaluate:
            self._evaluate(num_timesteps)
        while num_timesteps < total_timesteps:
            # Collect and store data from training environment
            steps_new, steps_noop = self._collect_env_data(num_timesteps)
            num_timesteps += steps_new

            # Update
            self._update(steps_new-steps_noop)

            # Evaluate
            if evaluate:
                if num_timesteps >= eval_points[eval_idx]:
                    self._evaluate(num_timesteps)
                    eval_idx += 1

            # Save training data to checkpoint file
            if num_timesteps >= checkpoints[checkpt_idx]:
                self._dump_and_save(params)
                checkpt_idx += 1
        
        return self.checkpoint_name

    def _dump_stats(self):
        """Returns dictionary of NN weights and normalization stats."""
        
        final = dict()
        
        # Actor weights
        final['actor_weights'] = self.actor.get_weights()
        try:
            final['adversary_weights'] = self.actor.get_adversary_weights()
        except:
            final['adversary_weights'] = None
        
        # Critic weights
        final['critic_weights'] = self.critic.get_weights()
        if self.safe:
            final['cost_critic_weights'] = self.cost_critic.get_weights()
        else:
            final['cost_critic_weights'] = None
        
        # Robustness perturbation weights    
        final['rob_weights'] = self.rob_net.get_weights()
        
        # Normalization stats
        final['rms_stats'] = self.normalizer.get_rms_stats()

        return final

    def _dump_and_save(self,params):
        """Saves training data to checkpoint file and resets logger."""
        self.logger.log_params(params)

        final = self._dump_stats()
        self.logger.log_final(final)

        self.logger.dump_and_save(self.save_path,self.checkpoint_name)
        self.logger.reset()