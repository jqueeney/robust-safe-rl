import numpy as np
import tensorflow as tf

from robust_safe_rl.common.nn_utils import soft_value

class BaseUpdate:
    """Base class for RL updates."""

    def __init__(self,actor,critic,cost_critic,rob_net,
        update_kwargs,safety_kwargs):
        """Initializes RL update class.
        
        Args:
            actor (object): policy
            critic (object): value function
            cost_critic (object): cost value function
            rob_net (object): robustness perturbation network
            update_kwargs (dict): update algorithm parameters
            safety_kwargs (dict): safety parameters
        """

        self.actor = actor
        self.critic = critic
        self.cost_critic = cost_critic
        self.rob_net = rob_net
        
        self._setup(update_kwargs,safety_kwargs)

    def _setup(self,update_kwargs,safety_kwargs):
        """Sets up hyperparameters as class attributes."""

        self.critic_lr = update_kwargs['critic_lr']
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr)
        self.cost_critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr)
        
        self.use_targ = update_kwargs['use_targ']
        self.tau = update_kwargs['tau']

        self.safe = safety_kwargs['safe']
        self.safe_type = safety_kwargs['safe_type']
        self.gamma = safety_kwargs['gamma']
        self.safety_budget_tot = safety_kwargs['safety_budget'] 
        self.env_horizon = safety_kwargs['env_horizon']
        if self.gamma < 1.0:
            self.safety_budget = (self.safety_budget_tot / self.env_horizon) / (
                1-self.gamma)
        else:
            self.safety_budget = self.safety_budget_tot
        
        self.safety_lagrange_init = safety_kwargs['safety_lagrange_init']
        self.soft_safety_lagrange_init = soft_value(self.safety_lagrange_init)
        self.soft_safety_lagrange = tf.Variable(self.soft_safety_lagrange_init,
            dtype=tf.float32)
        self.safety_lagrange = tf.math.softplus(self.soft_safety_lagrange)
        
        self.safety_lagrange_lr = safety_kwargs['safety_lagrange_lr']
        self.safety_lagrange_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.safety_lagrange_lr)

        self.adversary_freq = update_kwargs['actor_adversary_freq']
        self.use_adversary = (self.adversary_freq > 0)
        self.adversary_counter = 0
        
        self.update_batch_size = update_kwargs['update_batch_size']
        self.updates_per_step = update_kwargs['updates_per_step']

        self.eval_batch_size = update_kwargs['eval_batch_size']

        self.robust = update_kwargs['robust']

    def _update_safety_lagrange(self,cost_ave):
        """Updates safety Lagrange variable."""
        with tf.GradientTape() as tape:
            safety_lagrange = tf.math.softplus(self.soft_safety_lagrange)
            safety_lagrange_loss = - safety_lagrange * (
                cost_ave - self.safety_budget)
        
        safety_lagrange_grad = tape.gradient(safety_lagrange_loss,
            self.soft_safety_lagrange)

        self.safety_lagrange_optimizer.apply_gradients(
            zip([safety_lagrange_grad],[self.soft_safety_lagrange]))
        
        self.safety_lagrange = tf.math.softplus(self.soft_safety_lagrange)

    def _apply_actor_grads(self,s_batch,adversary=False):
        """Performs single actor update.

        Args:
            s_batch (np.ndarray): states
            adversary (bool): if True, update actor and adversary
        """
        raise NotImplementedError

    def _get_actor_log(self):
        """Returns stats for logging."""
        raise NotImplementedError

    def _apply_rob_grads(self,s_batch,a_batch,sp_batch,disc_batch):
        """Performs single robustness update and returns critic next values."""

        with tf.GradientTape() as tape:
            (sp_final, sp_final_cost, delta_out, 
                delta_out_cost) = self.rob_net.sample(s_batch,a_batch,sp_batch)

            rtg_next_all, ctg_next_all = self._get_critic_next_values(
                sp_final,sp_final_cost)

            (rob_loss, rtg_next_values, 
                ctg_next_values) =  self.rob_net.loss_and_targets(disc_batch,
                    rtg_next_all,ctg_next_all,delta_out,delta_out_cost)
        
        self.rob_net.apply_updates(tape,rob_loss)

        return rtg_next_values, ctg_next_values

    def _get_critic_next_values(self,sp_active,sp_cost_active=None):
        """Calculates critic next values for flattened next state batch."""
        ap_active = self.actor.sample(sp_active,targ=self.use_targ)
        
        rtg_next_values = self.critic.value_targ((sp_active,ap_active))

        if self.safe:
            if sp_cost_active is not None:
                ap_cost_active = self.actor.sample(
                    sp_cost_active,targ=self.use_targ)
                ctg_next_values = self.cost_critic.value_targ(
                    (sp_cost_active,ap_cost_active))
            else:
                ctg_next_values = self.cost_critic.value_targ(
                    (sp_active,ap_active))
        else:
            ctg_next_values = tf.zeros_like(rtg_next_values)
        
        return rtg_next_values, ctg_next_values

    def _get_critic_targets(self,r_active,c_active,disc_active,
        rtg_next_values,ctg_next_values):
        """Calculates target values for critic loss."""
        
        rtg_active = r_active + disc_active * rtg_next_values

        if self.safe:
            ctg_active = c_active + disc_active * ctg_next_values
        else:
            ctg_active = tf.zeros_like(rtg_active)
        
        return rtg_active, ctg_active

    def _apply_critic_grads(self,s_active,a_active,rtg_active,critic,
        critic_optimizer):
        """Applies critic gradients."""

        inputs_active = (s_active,a_active)
        with tf.GradientTape() as tape:
            critic_loss = critic.get_loss(inputs_active,rtg_active)
        
        grads = tape.gradient(critic_loss,critic.trainable)
        critic_optimizer.apply_gradients(zip(grads,critic.trainable))

        critic.update_targs(self.tau)

    def update_actor_critic(self,buffer,steps_new):
        """Updates actor and critic."""

        rollout_data_eval = buffer.get_offpolicy_info(
            batch_size=self.eval_batch_size)
        s_eval, a_eval, sp_eval, disc_eval, r_eval, c_eval = rollout_data_eval
        kl_info_ref = self.actor.get_kl_info(s_eval)
        if self.use_adversary:
            kl_info_ref_adversary = self.actor.get_kl_info(s_eval,adversary=True)
        
        num_updates = int(steps_new * self.updates_per_step)

        grad_norm_pre_all = 0.0
        grad_norm_post_all = 0.0

        for _ in range(num_updates):
            rollout_data = buffer.get_offpolicy_info(
                batch_size=self.update_batch_size)
            
            (s_batch, a_batch, sp_batch, disc_batch,
                r_batch, c_batch) = rollout_data

            if self.robust:
                rtg_next_values, ctg_next_values = self._apply_rob_grads(
                    s_batch,a_batch,sp_batch,disc_batch)
            else:
                rtg_next_values, ctg_next_values = self._get_critic_next_values(
                    sp_batch)
            
            rtg_batch, ctg_batch = self._get_critic_targets(r_batch,c_batch,
                disc_batch,rtg_next_values,ctg_next_values)
        
            # Critics update
            self._apply_critic_grads(s_batch,a_batch,rtg_batch,self.critic,
                self.critic_optimizer)
            
            if self.safe:
                self._apply_critic_grads(s_batch,a_batch,ctg_batch,
                    self.cost_critic,self.cost_critic_optimizer)
            
            # Actor update
            if self.use_adversary and (self.adversary_counter == 0):
                grad_norm_pre, grad_norm_post = self._apply_actor_grads(
                    s_batch,adversary=True)
                self.actor.update_adversary_targ(self.tau)
            else:
                grad_norm_pre, grad_norm_post = self._apply_actor_grads(s_batch)
            
            self.actor.update_targ(self.tau)
            
            if self.use_adversary:
                self.adversary_counter += 1
                self.adversary_counter = self.adversary_counter % self.adversary_freq

            grad_norm_pre_all += grad_norm_pre
            grad_norm_post_all += grad_norm_post

        # Logging
        grad_norm_pre_ave = grad_norm_pre_all.numpy() / num_updates
        grad_norm_post_ave = grad_norm_post_all.numpy() / num_updates
        ent = tf.reduce_mean(self.actor.entropy(s_eval))
        
        kl = tf.reduce_mean(self.actor.kl(s_eval,kl_info_ref))
        kl_targ_all = self.actor.kl_targ(s_eval,separate=False,per_dim=False)
        kl_targ = tf.reduce_mean(kl_targ_all)
        
        kl_targ_mean_all, kl_targ_std_all = self.actor.kl_targ(s_eval,
            separate=True,per_dim=False)
        kl_targ_mean = tf.reduce_mean(kl_targ_mean_all)
        kl_targ_std = tf.reduce_mean(kl_targ_std_all)

        log_actor = {
            'ent':                  ent.numpy(),
            'kl':                   kl.numpy(),
            'kl_targ':              kl_targ.numpy(),
            'kl_targ_mean':         kl_targ_mean.numpy(),
            'kl_targ_std':          kl_targ_std.numpy(),
            'actor_grad_norm_pre':  grad_norm_pre_ave,
            'actor_grad_norm':      grad_norm_post_ave,
            'safety_lagrange':      self.safety_lagrange.numpy(),
            'safety_budget':        self.safety_budget,
        }
        log_actor_alg = self._get_actor_log()
        log_actor.update(log_actor_alg)

        if self.use_adversary:
            ent_adversary = tf.reduce_mean(
                self.actor.entropy(s_eval,adversary=True))
            
            kl_adversary = tf.reduce_mean(self.actor.kl(
                s_eval,kl_info_ref_adversary,adversary=True))
            kl_targ_all_adversary = self.actor.kl_targ(
                s_eval,separate=False,per_dim=False,adversary=True)
            kl_targ_adversary = tf.reduce_mean(kl_targ_all_adversary)
            
            (kl_targ_mean_all_adversary, 
                kl_targ_std_all_adversary) = self.actor.kl_targ(
                s_eval,separate=True,per_dim=False,adversary=True)
            kl_targ_mean_adversary = tf.reduce_mean(kl_targ_mean_all_adversary)
            kl_targ_std_adversary = tf.reduce_mean(kl_targ_std_all_adversary)

            log_adversary = {
                'ent_adversary':            ent_adversary.numpy(),
                'kl_adversary':             kl_adversary.numpy(),
                'kl_targ_adversary':        kl_targ_adversary.numpy(),
                'kl_targ_mean_adversary':   kl_targ_mean_adversary.numpy(),
                'kl_targ_std_adversary':    kl_targ_std_adversary.numpy(),
            }
            log_actor.update(log_adversary)

        idx = np.arange(self.eval_batch_size)
        sections = np.arange(0,self.eval_batch_size,self.update_batch_size)[1:]
        batches = np.array_split(idx,sections)
        
        critic_loss_all = 0.0
        cost_critic_loss_all = 0.0
        
        rob_mag_all = 0.0
        rob_cost_mag_all = 0.0
        
        for batch_idx in batches:
            s_eval_active = s_eval[batch_idx]
            a_eval_active = a_eval[batch_idx]
            sp_eval_active = sp_eval[batch_idx]
            disc_eval_active = disc_eval[batch_idx]
            r_eval_active = r_eval[batch_idx]
            c_eval_active = c_eval[batch_idx]

            if self.robust:
                (sp_final_eval, sp_final_cost_eval, delta_out_eval, 
                    delta_out_cost_eval) = self.rob_net.sample(
                        s_eval_active,a_eval_active,sp_eval_active)
                
                next_values_all = self._get_critic_next_values(
                    sp_final_eval,sp_final_cost_eval)
                rtg_next_eval_all, ctg_next_eval_all = next_values_all

                (_, rtg_next_eval_active, 
                    ctg_next_eval_active) =  self.rob_net.loss_and_targets(
                        disc_eval_active,rtg_next_eval_all,ctg_next_eval_all,
                        delta_out_eval,delta_out_cost_eval)

                rob_mag_eval = self.rob_net.get_rob_magnitude(delta_out_eval)
                rob_cost_mag_eval = self.rob_net.get_rob_magnitude(delta_out_cost_eval)
                
                rob_mag_all += tf.reduce_mean(rob_mag_eval)
                rob_cost_mag_all += tf.reduce_mean(rob_cost_mag_eval)
            else:
                next_values_active = self._get_critic_next_values(
                    sp_eval_active)
                rtg_next_eval_active, ctg_next_eval_active = next_values_active
            
            rtg_eval_active, ctg_eval_active = self._get_critic_targets(
                r_eval_active,c_eval_active,disc_eval_active,
                rtg_next_eval_active,ctg_next_eval_active)

            inputs_eval_active = (s_eval_active,a_eval_active)
            critic_loss_active = self.critic.get_loss(
                inputs_eval_active,rtg_eval_active)
            critic_loss_all += critic_loss_active
            
            if self.safe:
                cost_critic_loss_active = self.cost_critic.get_loss(
                    inputs_eval_active,ctg_eval_active)
                cost_critic_loss_all += cost_critic_loss_active
        
        critic_loss = critic_loss_all / len(batches)
        cost_critic_loss = cost_critic_loss_all / len(batches)

        log_critic = {
            'critic_loss':  critic_loss.numpy(),
        }

        if self.safe:
            log_cost_critic = {
                'critic_loss':  cost_critic_loss.numpy(),
            }
        else:
            log_cost_critic = None

        if self.robust:
            rob_magnitude = rob_mag_all / len(batches)
            rob_cost_magnitude = rob_cost_mag_all / len(batches)

            try:
                otp_dual = tf.math.softplus(self.rob_net.soft_otp_dual)
            except:
                otp_dual = tf.cast(0.0,dtype=tf.float32)

            try:
                otp_cost_dual = tf.math.softplus(self.rob_net.soft_otp_cost_dual)
            except:
                otp_cost_dual = tf.cast(0.0,dtype=tf.float32)

            log_rob = {
                'rob_magnitude':        rob_magnitude.numpy(),
                'rob_cost_magnitude':   rob_cost_magnitude.numpy(),
                'otp_dual':             otp_dual.numpy(),
                'otp_cost_dual':        otp_cost_dual.numpy(),
            }
            log_critic.update(log_rob)


        return log_actor, log_critic, log_cost_critic