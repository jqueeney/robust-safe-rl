import numpy as np
import tensorflow as tf

from robust_safe_rl.robust_methods.base_rob_net import RobustNet
from robust_safe_rl.common.nn_utils import transform_features, create_nn
from robust_safe_rl.common.nn_utils import soft_value

class OTP(RobustNet):
    """Class for Optimal Transport Perturbations."""

    def __init__(self,env,
        otp_layers,otp_activations,otp_init_type,otp_gain,otp_layer_norm,
        rob_setup_kwargs,safety_kwargs):
        """Initializes OTP class.

        Args:
            env (object): environment

            otp_layers (list): list of hidden layer sizes for OTP NN
            otp_activations (list): list of activations for OTP NN
            otp_init_type (str): initialization type for OTP NN
            otp_gain (float): mult factor for final layer OTP NN init
            otp_layer_norm (bool): if True, first layer of OTP NN is layer norm

            rob_setup_kwargs (dict): robustness perturbation setup parameters
            safety_kwargs (dict): safety parameters
        """
        super(OTP,self).__init__(env,rob_setup_kwargs,safety_kwargs)

        in_dim = self.s_dim + self.a_dim + self.s_dim  # (s,a,s')
        out_dim = self.s_dim

        self._nn_otp = create_nn(in_dim,out_dim,otp_layers,
            otp_activations,otp_init_type,otp_gain,otp_layer_norm)
        self._nn_otp_cost = create_nn(in_dim,out_dim,otp_layers,
            otp_activations,otp_init_type,otp_gain,otp_layer_norm)
        
        self.otp_trainable = []
        self.otp_dual_trainable = []
        if self.rob_reward_attitude != 'neutral':
            self.otp_trainable += self._nn_otp.trainable_variables
            self.otp_dual_trainable += [self.soft_otp_dual]
        if self.safe:
            self.otp_trainable += self._nn_otp_cost.trainable_variables
            self.otp_dual_trainable += [self.soft_otp_cost_dual]

    def _setup(self,rob_setup_kwargs,safety_kwargs):
        """Sets up hyperparameters as class attributes."""
        super(OTP,self)._setup(rob_setup_kwargs,safety_kwargs)

        self.otp_nn_lr = rob_setup_kwargs['otp_nn_lr']
        self.otp_dual_lr = rob_setup_kwargs['otp_dual_lr']
        
        self.otp_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.otp_nn_lr)
        self.otp_dual_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.otp_dual_lr)

        self.otp_targ = np.square(self.rob_magnitude)

        self.otp_dual_init = rob_setup_kwargs['otp_dual_init']
        self.soft_otp_dual_init = soft_value(self.otp_dual_init)
        self.soft_otp_dual = tf.Variable(self.soft_otp_dual_init,
            dtype=tf.float32)
        self.soft_otp_cost_dual = tf.Variable(self.soft_otp_dual_init,
            dtype=tf.float32)

    def _transform_sa(self,s,a):
        """Preprocesses states and actions before passing data to NN."""
        s_norm = self.s_rms.normalize(s)
        a_norm = tf.clip_by_value(a,self.act_low,self.act_high)

        s_feat = transform_features(s_norm)
        a_feat = transform_features(a_norm)
        sa_feat = tf.concat([s_feat,a_feat],axis=-1)

        return sa_feat

    def _forward(self,in_feat,cost=False):
        """Returns output of robustness perturbation network."""
        
        if cost:
            delta_otp = self._nn_otp_cost(in_feat)
        else:
            delta_otp = self._nn_otp(in_feat)

        return self.rob_magnitude * delta_otp

    def _sample_perturb(self,s,delta_feat,in_feat,cost=False):
        """Returns samples of transition perturbations."""
        
        delta_out = self._forward(in_feat,cost=cost)
        sp_final = self._calc_sp(s,delta_feat,delta_out)

        return sp_final, delta_out

    def sample(self,s,a,sp):
        """Returns perturbed next state samples and perturbation outputs."""
        sa_feat = self._transform_sa(s,a)
        delta_feat = self._transform_delta(s,sp)
        in_feat = tf.concat([sa_feat,delta_feat],axis=-1)

        if self.rob_reward_attitude != 'neutral':
            sp_final, delta_out = self._sample_perturb(
                s,delta_feat,in_feat,cost=False)
        else:
            sp_final = sp
            delta_out = tf.zeros_like(delta_feat)

        if self.safe:
            sp_final_cost, delta_out_cost = self._sample_perturb(
                s,delta_feat,in_feat,cost=True)
        else:
            sp_final_cost = None
            delta_out_cost = tf.zeros_like(delta_feat)
        
        return sp_final, sp_final_cost, delta_out, delta_out_cost

    def loss_and_targets(self,disc,rtg_next_all,ctg_next_all,
        delta_out,delta_out_cost):
        """Returns loss function and critic targets."""
        otp_loss = 0.0

        if self.rob_reward_attitude != 'neutral':
            V_next = disc * rtg_next_all
            if self.rob_reward_attitude == 'optimistic':
                V_next = V_next * -1
            otp_reward_loss = tf.reduce_mean(V_next)

            otp_trust_all = self._get_rob_loss(delta_out)
            otp_trust = tf.reduce_mean(otp_trust_all)

            otp_dual = tf.math.softplus(self.soft_otp_dual)
            otp_reward_reg = otp_dual * (
                otp_trust - self.otp_targ) / self.otp_targ
            
            otp_reward_loss = otp_reward_loss + otp_reward_reg
            
            otp_loss = otp_loss + otp_reward_loss
        
        if self.safe:
            V_cost_next = disc * ctg_next_all
            otp_cost_loss = tf.reduce_mean(V_cost_next)

            otp_cost_trust_all = self._get_rob_loss(delta_out_cost)
            otp_cost_trust = tf.reduce_mean(otp_cost_trust_all)

            otp_cost_dual = tf.math.softplus(self.soft_otp_cost_dual)
            otp_cost_reg = otp_cost_dual * (
                otp_cost_trust - self.otp_targ) / self.otp_targ
            
            otp_cost_loss = otp_cost_loss + otp_cost_reg
            
            otp_loss = otp_loss + otp_cost_loss
        
        return otp_loss, rtg_next_all, ctg_next_all

    def apply_updates(self,tape,rob_loss):
        """Applies gradient updates."""
        otp_grads, otp_dual_grads = tape.gradient(rob_loss,
            [self.otp_trainable,self.otp_dual_trainable])
        otp_dual_grads = [grad*-1 for grad in otp_dual_grads]

        self.otp_optimizer.apply_gradients(
            zip(otp_grads,self.otp_trainable))
        self.otp_dual_optimizer.apply_gradients(
            zip(otp_dual_grads,self.otp_dual_trainable))

    def get_weights(self):
        """Returns parameter weights."""
        otp_weights = self._nn_otp.get_weights()
        otp_cost_weights = self._nn_otp_cost.get_weights()

        rob_weights = [
            otp_weights,
            otp_cost_weights
        ]
        return rob_weights

    def set_weights(self,weights):
        """Sets parameter weights."""
        otp_weights, otp_cost_weights = weights

        self._nn_otp.set_weights(otp_weights)
        self._nn_otp_cost.set_weights(otp_cost_weights)