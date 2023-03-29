import tensorflow as tf
import gym

from robust_safe_rl.common.nn_utils import transform_features, create_nn

class BaseCritic:
    """Base critic."""

    def __init__(self,env):
        """Initializes critic.

        Args:
            env (object): environment
        """        

        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)

        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        self.s_rms, _, _, _, self.ret_rms, self.c_ret_rms = all_rms


class QCritic(BaseCritic):
    """State-action value function."""

    def __init__(self,env,layers,activations,init_type,gain,layer_norm,
        safety=False):
        """Initializes Q function.
        
        Args:
            env (object): environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            init_type (str): initialization type
            gain (float): multiplicative factor for final layer 
                initialization
            layer_norm (bool): if True, first layer is layer norm
            safety (bool): if True, safety critic for costs
        """
        super(QCritic,self).__init__(env)
        
        self.safety = safety
        
        in_dim = self.s_dim + self.a_dim
        
        self._nn = create_nn(in_dim,1,
            layers,activations,init_type,gain,layer_norm)
        self._nn_targ = create_nn(in_dim,1,
            layers,activations,init_type,gain,layer_norm)
        self._nn_targ.set_weights(self._nn.get_weights())
        
        self._nn2 = create_nn(in_dim,1,
            layers,activations,init_type,gain,layer_norm)
        self._nn2_targ = create_nn(in_dim,1,
            layers,activations,init_type,gain,layer_norm)
        self._nn2_targ.set_weights(self._nn2.get_weights())

        self.trainable = self._nn.trainable_variables + self._nn2.trainable_variables

    def _forward(self,data,nn='base1'):
        """Returns output of neural network."""
        s, a = data
        
        s_norm = self.s_rms.normalize(s)
        a_norm = tf.clip_by_value(a,self.act_low,self.act_high)
        
        s_feat = transform_features(s_norm)
        a_feat = transform_features(a_norm)

        sa_feat = tf.concat([s_feat,a_feat],axis=-1)

        if nn == 'base1':
            return self._nn(sa_feat)
        elif nn == 'base2':
            return self._nn2(sa_feat)
        elif nn == 'targ1':
            return self._nn_targ(sa_feat)
        elif nn == 'targ2':
            return self._nn2_targ(sa_feat)

    def _nn_value(self,in_data,nn='base1'):
        """Calculates value for given neural network."""
        value = tf.squeeze(self._forward(in_data,nn),axis=-1)
        if self.safety:
            value = self.c_ret_rms.denormalize(value,center=False)
        else:
            value = self.ret_rms.denormalize(value,center=False)
        
        return value

    def value(self,in_data):
        """Calculates value given input data."""
        value_base1 = self._nn_value(in_data,nn='base1')
        value_base2 = self._nn_value(in_data,nn='base2')
        return tf.reduce_mean([value_base1,value_base2],axis=0)

    def value_targ(self,in_data):
        """Calculates target value given input data."""
        value_targ1 = self._nn_value(in_data,nn='targ1')
        value_targ2 = self._nn_value(in_data,nn='targ2')
        return tf.minimum(value_targ1,value_targ2)

    def _get_nn_loss(self,in_data,target,nn='base1'):
        """Returns critic loss for given neural network."""
        value = self._nn_value(in_data,nn)
        
        if self.safety:
            value_norm = self.c_ret_rms.normalize(value,center=False)
            target_norm = self.c_ret_rms.normalize(target,center=False)
        else:
            value_norm = self.ret_rms.normalize(value,center=False)
            target_norm = self.ret_rms.normalize(target,center=False)

        return 0.5 * tf.reduce_mean(tf.square(target_norm - value_norm))

    def get_loss(self,in_data,target):
        """Returns critic loss."""
        loss_base1 = self._get_nn_loss(in_data,target,nn='base1')
        loss_base2 = self._get_nn_loss(in_data,target,nn='base2')
        return loss_base1 + loss_base2

    def update_targs(self,tau):
        """Updates target neural network weights."""
        for base_weight, targ_weight in zip(
            self._nn.variables,self._nn_targ.variables):
            targ_weight.assign((1-tau) * targ_weight + tau * base_weight)
        for base_weight, targ_weight in zip(
            self._nn2.variables,self._nn2_targ.variables):
            targ_weight.assign((1-tau) * targ_weight + tau * base_weight)

    def get_weights(self):
        """Returns parameter weights."""
        weights_base1 = self._nn.get_weights()
        weights_base2 = self._nn2.get_weights()
        weights_targ1 = self._nn_targ.get_weights()
        weights_targ2 = self._nn2_targ.get_weights()
        return weights_base1, weights_base2, weights_targ1, weights_targ2

    def set_weights(self,weights):
        """Sets parameter weights."""
        weights_base1, weights_base2, weights_targ1, weights_targ2 = weights
        self._nn.set_weights(weights_base1)
        self._nn2.set_weights(weights_base2)
        self._nn_targ.set_weights(weights_targ1)
        self._nn2_targ.set_weights(weights_targ2)