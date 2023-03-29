import numpy as np
import tensorflow as tf
import scipy as sp

from robust_safe_rl.algs.update_algs.base_update_alg import BaseUpdate
from robust_safe_rl.common.nn_utils import soft_value

class MPO(BaseUpdate):
    """Algorithm class for MPO actor updates."""

    def __init__(self,actor,critic,cost_critic,rob_net,
        update_kwargs,safety_kwargs):
        """Initializes MPO class."""
        super(MPO,self).__init__(actor,critic,cost_critic,rob_net,
            update_kwargs,safety_kwargs)
    
    def _setup(self,update_kwargs,safety_kwargs):
        """Sets up hyperparameters as class attributes."""
        super(MPO,self)._setup(update_kwargs,safety_kwargs)

        self.actor_lr = update_kwargs['actor_lr']
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.actor_lr)
        
        self.max_grad_norm = update_kwargs['max_grad_norm']

        self.temp_init = update_kwargs['temp_init']
        self.soft_temp_init = soft_value(self.temp_init)
        self.soft_temp = tf.Variable(self.soft_temp_init,dtype=tf.float32)

        self.act_penalty = update_kwargs['act_penalty']
        self.act_temp_init = update_kwargs['act_temp_init']
        self.soft_act_temp_init = soft_value(self.act_temp_init)
        self.soft_act_temp = tf.Variable(self.soft_act_temp_init,dtype=tf.float32)

        self.temp_closed = update_kwargs['temp_closed']
        self.temp_tol = update_kwargs['temp_tol']
        self.temp_bounds = [(1e-6, 1e3)]
        self.temp_opt_type = update_kwargs['temp_opt_type']
        if self.temp_opt_type == 'SLSQP':
            self.temp_options = {"maxiter": 10}
        elif self.temp_opt_type == 'Powell':
            self.temp_options = None
        else:
            raise ValueError('temp_opt_type not valid')
        
        self.temp_lr = update_kwargs['temp_lr']
        self.temp_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.temp_lr)

        self.temp_vars = [self.soft_temp]
        if self.act_penalty:
            self.temp_vars += [self.soft_act_temp]

        self.kl_separate = update_kwargs['kl_separate']
        self.kl_per_dim = update_kwargs['kl_per_dim']
        self.a_dim = self.actor.a_dim
        
        if self.kl_separate:
            self.dual_mean_init = update_kwargs['dual_mean_init']
            self.soft_dual_mean_init = soft_value(self.dual_mean_init)
            if self.kl_per_dim:
                self.soft_dual_mean_init = np.ones(self.a_dim) * (
                    self.soft_dual_mean_init)

            self.soft_dual_mean = tf.Variable(self.soft_dual_mean_init,
                dtype=tf.float32)
            
            self.dual_std_init = update_kwargs['dual_std_init']
            self.soft_dual_std_init = soft_value(self.dual_std_init)
            if self.kl_per_dim:
                self.soft_dual_std_init = np.ones(self.a_dim) * (
                    self.soft_dual_std_init)

            self.soft_dual_std = tf.Variable(self.soft_dual_std_init,
                dtype=tf.float32)
            self.dual_vars = [self.soft_dual_mean,self.soft_dual_std]
        else:
            self.dual_init = update_kwargs['dual_init']
            self.soft_dual_init = soft_value(self.dual_init)
            if self.kl_per_dim:
                self.soft_dual_init = np.ones(self.a_dim) * self.soft_dual_init
            
            self.soft_dual = tf.Variable(self.soft_dual_init,dtype=tf.float32)
            self.dual_vars = [self.soft_dual]
        
        self.dual_lr = update_kwargs['dual_lr']
        self.dual_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.dual_lr)

        self.mpo_num_actions = update_kwargs['mpo_num_actions']

        self.mpo_delta_E = update_kwargs['mpo_delta_E']
        self.mpo_delta_E_penalty = update_kwargs['mpo_delta_E_penalty']
        self.mpo_delta_M = update_kwargs['mpo_delta_M']
        self.mpo_delta_M_mean = update_kwargs['mpo_delta_M_mean']
        self.mpo_delta_M_std = update_kwargs['mpo_delta_M_std']

        if self.kl_per_dim:
            self.mpo_delta_M = self.mpo_delta_M / self.a_dim
            self.mpo_delta_M_mean = self.mpo_delta_M_mean / self.a_dim
            self.mpo_delta_M_std = self.mpo_delta_M_std / self.a_dim

        if self.use_adversary:
            self.adversary_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.actor_lr)
            self.soft_temp_adversary = tf.Variable(self.soft_temp_init,
                dtype=tf.float32)
            self.soft_act_temp_adversary = tf.Variable(self.soft_act_temp_init,
                dtype=tf.float32)

            self.temp_adversary_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.temp_lr)

            self.temp_adversary_vars = [self.soft_temp_adversary]
            if self.act_penalty:
                self.temp_adversary_vars += [self.soft_act_temp_adversary]

            if self.kl_separate:
                self.soft_dual_mean_adversary = tf.Variable(
                    self.soft_dual_mean_init,dtype=tf.float32)
                self.soft_dual_std_adversary = tf.Variable(
                    self.soft_dual_std_init,dtype=tf.float32)
                self.dual_adversary_vars = [
                    self.soft_dual_mean_adversary,self.soft_dual_std_adversary]
            else:               
                self.soft_dual_adversary = tf.Variable(
                    self.soft_dual_init,dtype=tf.float32)
                self.dual_adversary_vars = [self.soft_dual_adversary]
            
            self.dual_adversary_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.dual_lr)

    def _get_actor_log(self):
        """Returns stats for logging."""
        log_actor = {
            'mpo_temp':     tf.math.softplus(self.soft_temp).numpy(),
            'mpo_act_temp': tf.math.softplus(self.soft_act_temp).numpy(),
        }

        if self.kl_separate:
            log_actor['mpo_dual_mean'] = tf.math.softplus(
                self.soft_dual_mean).numpy()
            log_actor['mpo_dual_std'] = tf.math.softplus(
                self.soft_dual_std).numpy()
        else:
            log_actor['mpo_dual'] = tf.math.softplus(self.soft_dual).numpy()
        
        if self.use_adversary:
            log_actor['mpo_temp_adversary'] = tf.math.softplus(
                self.soft_temp_adversary).numpy()
            log_actor['mpo_act_temp_adversary'] = tf.math.softplus(
                self.soft_act_temp_adversary).numpy()

            if self.kl_separate:
                log_actor['mpo_dual_mean_adversary'] = tf.math.softplus(
                    self.soft_dual_mean_adversary).numpy()
                log_actor['mpo_dual_std_adversary'] = tf.math.softplus(
                    self.soft_dual_std_adversary).numpy()
            else:
                log_actor['mpo_dual_adversary'] = tf.math.softplus(
                    self.soft_dual_adversary).numpy()
        
        return log_actor

    def _update_temp_closed(self,Q_batch_active,soft_temp,target_kl):
        """Full solve of temperature variables."""

        def temp_dual(temp):
            Q_temp_batch = Q_batch_active.numpy() / temp
            Q_logsumexp_batch = sp.special.logsumexp(
                Q_temp_batch,axis=0) - np.log(self.mpo_num_actions)
            Q_logsumexp = np.mean(Q_logsumexp_batch)
            temp_loss = temp * (Q_logsumexp + target_kl)
            return temp_loss

        temp_cur = tf.math.softplus(soft_temp).numpy()
        start = np.array([temp_cur])
        try:
            res = sp.optimize.minimize(temp_dual,
                            start,
                            method=self.temp_opt_type,
                            bounds=self.temp_bounds*len(start),
                            tol=self.temp_tol,
                            options=self.temp_options)
            
            temp_star = res.x[0]
        except:
            print('Error in temperature optimization')
            temp_star = start[0]

        soft_temp_star = soft_value(temp_star)
        soft_temp.assign(soft_temp_star)

    def _calc_temp_loss(self,Q_batch_active,soft_temp,target_kl):
        """Calculates temperature variable loss."""
        temp = tf.math.softplus(soft_temp)

        Q_temp_batch = Q_batch_active / temp
        Q_logsumexp_batch = (tf.reduce_logsumexp(Q_temp_batch,axis=0) 
            - tf.math.log(tf.cast(self.mpo_num_actions,dtype=tf.float32)))
        Q_logsumexp = tf.reduce_mean(Q_logsumexp_batch)
        
        temp_loss = temp * (Q_logsumexp + target_kl)

        return temp_loss

    def _calc_policy_grad(self,s_batch_flat,a_batch_flat,weights_batch_flat,
        policy_vars,dual_vars,adversary=False):
        """Calculates policy gradient."""
        with tf.GradientTape() as tape:
            neglogp_batch_flat = self.actor.neglogp(
                s_batch_flat,a_batch_flat,adversary=adversary)

            pg_loss_batch = weights_batch_flat * neglogp_batch_flat
            pg_loss = tf.reduce_mean(pg_loss_batch)
            
            kl_targ_batch = self.actor.kl_targ(s_batch_flat,
                self.kl_separate,self.kl_per_dim,adversary=adversary)
            if self.kl_separate:
                soft_dual_mean, soft_dual_std = dual_vars
                dual_mean = tf.math.softplus(soft_dual_mean)
                dual_std = tf.math.softplus(soft_dual_std)
                
                kl_targ_mean_batch, kl_targ_std_batch = kl_targ_batch
                kl_targ_mean = tf.reduce_mean(kl_targ_mean_batch,axis=0)
                kl_targ_std = tf.reduce_mean(kl_targ_std_batch,axis=0)
                kl_targ = [kl_targ_mean, kl_targ_std]

                pg_loss = (pg_loss
                    + tf.reduce_sum(dual_mean * kl_targ_mean) 
                    + tf.reduce_sum(dual_std * kl_targ_std)
                )
            else:
                soft_dual = dual_vars[0]
                dual = tf.math.softplus(soft_dual)
                kl_targ = tf.reduce_mean(kl_targ_batch,axis=0)

                pg_loss = pg_loss + tf.reduce_sum(dual * kl_targ)

        pol_grad = tape.gradient(pg_loss,policy_vars)

        return pol_grad, kl_targ

    def _calc_dual_grad(self,kl_targ,dual_vars):
        """Calculates dual variable gradients."""
        with tf.GradientTape() as tape:
            if self.kl_separate:
                kl_targ_mean, kl_targ_std = kl_targ
                soft_dual_mean, soft_dual_std = dual_vars
                dual_mean = tf.math.softplus(soft_dual_mean)
                dual_std = tf.math.softplus(soft_dual_std)

                dual_loss_mean = tf.reduce_sum(dual_mean 
                    * (self.mpo_delta_M_mean - kl_targ_mean))
                dual_loss_std = tf.reduce_sum(dual_std 
                    * (self.mpo_delta_M_std - kl_targ_std))
                dual_loss = dual_loss_mean + dual_loss_std
            else:
                soft_dual = dual_vars[0]
                dual = tf.math.softplus(soft_dual)
                dual_loss = tf.reduce_sum(dual * (self.mpo_delta_M - kl_targ))
        
        soft_dual_grad = tape.gradient(dual_loss,dual_vars)

        return soft_dual_grad

    def _apply_actor_grads(self,s_batch,adversary=False):
        """Performs single actor update."""

        # Calculate non-parametric weights
        if len(np.shape(s_batch)) == 1:
            s_batch = np.expand_dims(s_batch,axis=0)

        if adversary:
            self._apply_adversary_grads(s_batch)

        batch_size = np.shape(s_batch)[0]
        s_batch_flat = tf.tile(s_batch,[self.mpo_num_actions,1])
        
        if self.use_adversary:
            a_batch_flat = self.actor.sample_separate(s_batch_flat,
                targ=self.use_targ,adversary=False)
        else:
            a_batch_flat = self.actor.sample(s_batch_flat,targ=self.use_targ)

        if self.act_penalty:
            a_diff_batch_flat = a_batch_flat - tf.clip_by_value(
                a_batch_flat,-1.,1.)
            a_penalty_batch_flat = tf.norm(a_diff_batch_flat, axis=-1) * -1
            a_penalty_batch = tf.reshape(a_penalty_batch_flat,
                [self.mpo_num_actions,batch_size])

        if self.use_targ:
            Q_batch_flat = self.critic.value_targ((s_batch_flat,a_batch_flat))
        else:
            Q_batch_flat = self.critic.value((s_batch_flat,a_batch_flat))
        Q_batch = tf.reshape(Q_batch_flat,[self.mpo_num_actions,batch_size])
        
        if self.safe:
            if self.use_targ:
                Q_cost_batch_flat = self.cost_critic.value_targ(
                    (s_batch_flat,a_batch_flat))
            else:
                Q_cost_batch_flat = self.cost_critic.value(
                    (s_batch_flat,a_batch_flat))
            Q_cost_batch = tf.reshape(Q_cost_batch_flat,
                [self.mpo_num_actions,batch_size])
                        
            if (self.safe_type == 'crpo'):
                cost_ave = tf.reduce_mean(Q_cost_batch) * -1
                if (cost_ave > self.safety_budget):
                    Q_batch_active = Q_cost_batch
                else:
                    Q_batch_active = Q_batch
            else:
                Q_batch_active = Q_batch + self.safety_lagrange * Q_cost_batch
        else:
            Q_batch_active = Q_batch

        if self.temp_closed:
            self._update_temp_closed(Q_batch_active,
                self.soft_temp,self.mpo_delta_E)
            if self.act_penalty:
                self._update_temp_closed(a_penalty_batch,
                    self.soft_act_temp,self.mpo_delta_E_penalty)

        temp = tf.math.softplus(self.soft_temp)
        Q_temp_batch = Q_batch_active / temp
        
        weights_batch = tf.nn.softmax(
            Q_temp_batch,axis=0) * self.mpo_num_actions
        weights_batch_flat = tf.reshape(weights_batch,[-1])

        if self.act_penalty:
            act_temp = tf.math.softplus(self.soft_act_temp)
            a_penalty_temp_batch = a_penalty_batch / act_temp
            weights_act_batch = tf.nn.softmax(
                a_penalty_temp_batch,axis=0) * self.mpo_num_actions
            weights_act_batch_flat = tf.reshape(weights_act_batch,[-1])

            weights_batch_flat = tf.reduce_mean(
                [weights_batch_flat,weights_act_batch_flat],axis=0)

        # Parametric policy update gradient
        pol_grad, kl_targ = self._calc_policy_grad(s_batch_flat,a_batch_flat,
            weights_batch_flat,self.actor.trainable,self.dual_vars,
            adversary=False)
        
        # Temperature update
        if not self.temp_closed:
            with tf.GradientTape() as tape:
                temp_loss = self._calc_temp_loss(Q_batch_active,
                    self.soft_temp,self.mpo_delta_E)
                if self.act_penalty:
                    act_temp_loss = self._calc_temp_loss(a_penalty_batch,
                        self.soft_act_temp,self.mpo_delta_E_penalty)
                    temp_loss += act_temp_loss

            temp_grad = tape.gradient(temp_loss,self.temp_vars)
            self.temp_optimizer.apply_gradients(zip(temp_grad,self.temp_vars))

        # Dual update
        soft_dual_grad = self._calc_dual_grad(kl_targ,self.dual_vars)
        self.dual_optimizer.apply_gradients(
            zip(soft_dual_grad,self.dual_vars))
        
        # Apply policy update
        if self.max_grad_norm is not None:
            pol_grad, grad_norm_pre = tf.clip_by_global_norm(
                pol_grad,self.max_grad_norm)
        else:
            grad_norm_pre = tf.linalg.global_norm(pol_grad)
        grad_norm_post = tf.linalg.global_norm(pol_grad)

        self.actor_optimizer.apply_gradients(zip(pol_grad,self.actor.trainable))

        # Safety Lagrange update
        if self.safe and (self.safe_type == 'lagrange'):
            cost_all = Q_cost_batch_flat * -1
            cost_ave = tf.reduce_mean(cost_all)
            self._update_safety_lagrange(cost_ave)

        return grad_norm_pre, grad_norm_post

    def _apply_adversary_grads(self,s_batch):
        """Performs single actor adversary update."""

        # Calculate non-parametric weights
        batch_size = np.shape(s_batch)[0]
        s_batch_flat = tf.tile(s_batch,[self.mpo_num_actions,1])
        # adversary actions
        a_batch_flat = self.actor.sample_separate(s_batch_flat,
            targ=self.use_targ,adversary=True)

        if self.act_penalty:
            a_diff_batch_flat = a_batch_flat - tf.clip_by_value(
                a_batch_flat,-1.,1.)
            a_penalty_batch_flat = tf.norm(a_diff_batch_flat, axis=-1) * -1
            a_penalty_batch = tf.reshape(a_penalty_batch_flat,
                [self.mpo_num_actions,batch_size])
        
        if self.safe:
            if self.use_targ:
                Q_batch_active_flat = self.cost_critic.value_targ(
                    (s_batch_flat,a_batch_flat))
            else:
                Q_batch_active_flat = self.cost_critic.value(
                    (s_batch_flat,a_batch_flat))
        else:
            if self.use_targ:
                Q_batch_active_flat = self.critic.value_targ(
                    (s_batch_flat,a_batch_flat))
            else:
                Q_batch_active_flat = self.critic.value(
                    (s_batch_flat,a_batch_flat))
        
        # flip sign for adversarial
        Q_batch_active_flat = Q_batch_active_flat * -1
        Q_batch_active = tf.reshape(Q_batch_active_flat,
            [self.mpo_num_actions,batch_size])

        if self.temp_closed:
            self._update_temp_closed(Q_batch_active,
                self.soft_temp_adversary,self.mpo_delta_E)
            if self.act_penalty:
                self._update_temp_closed(a_penalty_batch,
                    self.soft_act_temp_adversary,self.mpo_delta_E_penalty)
        
        temp = tf.math.softplus(self.soft_temp_adversary)
        Q_temp_batch = Q_batch_active / temp
        
        weights_batch = tf.nn.softmax(
            Q_temp_batch,axis=0) * self.mpo_num_actions
        weights_batch_flat = tf.reshape(weights_batch,[-1])

        if self.act_penalty:
            act_temp = tf.math.softplus(self.soft_act_temp_adversary)
            a_penalty_temp_batch = a_penalty_batch / act_temp
            weights_act_batch = tf.nn.softmax(
                a_penalty_temp_batch,axis=0) * self.mpo_num_actions
            weights_act_batch_flat = tf.reshape(weights_act_batch,[-1])

            weights_batch_flat = tf.reduce_mean(
                [weights_batch_flat,weights_act_batch_flat],axis=0)

        # Parametric policy update gradient
        pol_grad, kl_targ = self._calc_policy_grad(s_batch_flat,a_batch_flat,
            weights_batch_flat,self.actor.adversary_trainable,
            self.dual_adversary_vars,adversary=True)
        
        # Temperature update
        if not self.temp_closed:
            with tf.GradientTape() as tape:
                temp_loss = self._calc_temp_loss(Q_batch_active,
                    self.soft_temp_adversary,self.mpo_delta_E)
                if self.act_penalty:
                    act_temp_loss = self._calc_temp_loss(a_penalty_batch,
                        self.soft_act_temp_adversary,self.mpo_delta_E_penalty)
                    temp_loss += act_temp_loss

            temp_grad = tape.gradient(temp_loss,self.temp_adversary_vars)
            self.temp_adversary_optimizer.apply_gradients(
                zip(temp_grad,self.temp_adversary_vars))

        # Dual update
        soft_dual_grad = self._calc_dual_grad(kl_targ,self.dual_adversary_vars)
        self.dual_adversary_optimizer.apply_gradients(
            zip(soft_dual_grad,self.dual_adversary_vars))

        # Apply policy update
        if self.max_grad_norm is not None:
            pol_grad, _ = tf.clip_by_global_norm(
                pol_grad,self.max_grad_norm)

        self.adversary_optimizer.apply_gradients(
            zip(pol_grad,self.actor.adversary_trainable))