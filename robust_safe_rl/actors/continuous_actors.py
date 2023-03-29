import gym
import numpy as np
import tensorflow as tf

from robust_safe_rl.common.nn_utils import transform_features, create_nn
from robust_safe_rl.common.nn_utils import flat_to_list, list_to_flat

class BaseActor:
    """Base policy class."""

    def __init__(self,env):
        """Initializes policy.

        Args:
            env (object): environment
        """
        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)

    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        self.s_rms, _, _, _, _, _ = all_rms

    def _transform_state(self,s):
        """Preprocesses state before passing data to neural network."""
        s_norm = self.s_rms.normalize(s)
        s_feat = transform_features(s_norm)
        return s_feat

class GaussianActor(BaseActor):
    """Multivariate Gaussian policy with diagonal covariance.

    Mean action for a given state is parameterized by a neural network.
    Diagonal covariance parameterized by same neural network if state-dependent,
    otherwise is parameterized separately.
    """

    def __init__(self,env,layers,activations,init_type,gain,layer_norm,
        std_mult=1.0,per_state_std=True,output_norm=False):
        """Initializes multivariate Gaussian policy with diagonal covariance.

        Args:
            env (object): environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            init_type (str): initialization type
            gain (float): multiplicative factor for final layer initialization
            layer_norm (bool): if True, first layer is layer norm
            std_mult (float): multiplicative factor for diagonal covariance 
                initialization
            per_state_std (bool): if True, state-dependent diagonal covariance
            output_norm (bool): if True, normalize output magnitude
        """

        assert isinstance(env.action_space,gym.spaces.Box), (
            'Only Box action space supported')
        
        super(GaussianActor,self).__init__(env)

        self.per_state_std = per_state_std
        self.output_norm = output_norm
        
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        if self.per_state_std:
            self.logstd_init = np.ones((1,)+env.action_space.shape,
                dtype='float32') * (np.log(std_mult) - np.log(np.log(2)))
        else:
            self.logstd_init = np.ones((1,)+env.action_space.shape,
                dtype='float32') * np.log(std_mult)
        
        if self.per_state_std:
            self._nn = create_nn(self.s_dim,2*self.a_dim,
                layers,activations,init_type,gain,layer_norm)
            self.trainable = self._nn.trainable_variables

            self._nn_targ = create_nn(self.s_dim,2*self.a_dim,
                layers,activations,init_type,gain,layer_norm)
        else:
            self._nn = create_nn(self.s_dim,self.a_dim,
                layers,activations,init_type,gain,layer_norm)
            self.logstd = tf.Variable(np.zeros_like(self.logstd_init),
                dtype=tf.float32)
            self.trainable = self._nn.trainable_variables + [self.logstd]

            self._nn_targ = create_nn(self.s_dim,self.a_dim,
                layers,activations,init_type,gain,layer_norm)
            self.logstd_targ = tf.Variable(np.zeros_like(self.logstd_init),
                dtype=tf.float32)

        self._nn_targ.set_weights(self._nn.get_weights())

    def _output_normalization(self,out):
        """Normalizes output of neural network."""
        out_max = tf.reduce_mean(tf.abs(out),axis=-1,keepdims=True)
        out_max = tf.maximum(out_max,1.0)
        return out / out_max

    def _forward(self,s,targ=False,adversary=False):
        """Returns output of neural network."""
        s_feat = self._transform_state(s)
        
        if targ:
            a_out = self._nn_targ(s_feat)
        else:
            a_out = self._nn(s_feat)
        
        if self.per_state_std:
            a_mean, a_std_out = tf.split(a_out,num_or_size_splits=2,axis=-1)
            a_std = tf.math.softplus(a_std_out)
            a_logstd = tf.math.log(a_std)
        else:
            a_mean = a_out
            if targ:
                a_logstd = self.logstd_targ * tf.ones_like(a_mean)
            else:
                a_logstd = self.logstd * tf.ones_like(a_mean)
        
        a_logstd = a_logstd + self.logstd_init
        a_logstd = tf.maximum(a_logstd,tf.math.log(1e-3))

        if self.output_norm:
            a_mean = self._output_normalization(a_mean)

        return a_mean, a_logstd

    def sample(self,s,deterministic=False,targ=False):
        """Samples an action from the current or target policy given the state.
        
        Args:
            s (np.ndarray): state
            deterministic (bool): if True, returns mean action
            targ (bool): if True, use target policy
        
        Returns:
            Action sampled from current or target policy.
        """
        a, a_logstd = self._forward(s,targ=targ)

        if not deterministic:
            u = np.random.normal(size=np.shape(a))
            a = a + tf.exp(a_logstd) * u

        if np.shape(a)[0] == 1:
            a = tf.squeeze(a,axis=0)

        return a

    def clip(self,a):
        """Clips action to feasible range."""
        return np.clip(a,self.act_low,self.act_high)
    
    def neglogp(self,s,a,targ=False,adversary=False):
        """Calculates negative log probability for given state and action."""
        a_mean, a_logstd = self._forward(s,targ=targ,adversary=adversary)

        neglogp_vec = (tf.square((a - a_mean) / tf.exp(a_logstd)) 
            + 2*a_logstd + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(neglogp_vec,axis=-1))

    def entropy(self,s,adversary=False):
        """Calculates entropy of current policy."""
        _, a_logstd = self._forward(s,adversary=adversary)
        ent_vec = 2*a_logstd + tf.math.log(2*np.pi) + 1
        return 0.5 * tf.reduce_sum(ent_vec,axis=-1)

    def kl(self,s,kl_info_ref,adversary=False):
        """Calculates forward KL divergence between current and reference policy.
        
        Args:
            s (np.ndarray): states
            kl_info_ref (np.ndarray): mean actions and log std. deviation for 
                reference policy
            adversary (bool): if True, use adversary policy
        
        Returns:
            np.ndarray of KL divergences between current policy and reference 
            policy at every input state.
        """
        ref_mean, ref_logstd = np.moveaxis(kl_info_ref,-1,0)
        a_mean, a_logstd = self._forward(s,adversary=adversary)

        num = tf.square(a_mean-ref_mean) + tf.exp(2*ref_logstd)
        kl_vec = num / tf.exp(2*a_logstd) + 2*a_logstd - 2*ref_logstd - 1

        return 0.5 * tf.reduce_sum(kl_vec,axis=-1)

    def get_kl_info(self,s,adversary=False):
        """Returns info needed to calculate KL divergence."""
        ref_mean, ref_logstd = self._forward(s,adversary=adversary)
        return np.stack((ref_mean,ref_logstd),axis=-1)

    def kl_targ(self,s,separate=False,per_dim=False,adversary=False):
        """Calculates KL divergence between current and target policy."""
        a_mean, a_logstd = self._forward(s,adversary=adversary)
        a_mean_targ, a_logstd_targ = self._forward(s,targ=True,adversary=adversary)

        if separate:
            kl_mean_vec = tf.square(a_mean-a_mean_targ) / tf.exp(2*a_logstd_targ)
            kl_std_vec = (tf.exp(2*a_logstd_targ) / tf.exp(2*a_logstd) 
                + 2*a_logstd - 2*a_logstd_targ - 1)
            
            if per_dim:
                kl_mean = 0.5 * kl_mean_vec
                kl_std = 0.5 * kl_std_vec
            else:
                kl_mean = 0.5 * tf.reduce_sum(kl_mean_vec,axis=-1)
                kl_std = 0.5 * tf.reduce_sum(kl_std_vec,axis=-1)
            
            return kl_mean, kl_std
        else:
            num = tf.square(a_mean-a_mean_targ) + tf.exp(2*a_logstd_targ)
            kl_vec = num / tf.exp(2*a_logstd) + 2*a_logstd - 2*a_logstd_targ - 1

            if per_dim:
                kl = 0.5 * kl_vec
            else:
                kl = 0.5 * tf.reduce_sum(kl_vec,axis=-1)
            
            return kl

    def get_weights(self,flat=False):
        """Returns parameter weights of current policy.
        
        Args:
            flat (bool): if True, returns weights as flattened np.ndarray
        
        Returns:
            list or np.ndarray of parameter weights.
        """
        weights = self._nn.get_weights()
        if not self.per_state_std:
            weights = weights + [self.logstd.numpy()]
        
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights of current policy.
        
        Args:
            weights (list, np.ndarray): list or np.ndarray of parameter weights
            from_flat (bool): if True, weights are flattened np.ndarray
            increment (bool): if True, weights are incremental values
        """
        if from_flat:
            weights = flat_to_list(self.trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights(flat=False)))

        if self.per_state_std:
            self._nn.set_weights(weights)
        else:
            model_weights = weights[:-1]
            logstd_weights = weights[-1]
            logstd_weights = np.maximum(logstd_weights,np.log(1e-3))
            
            self._nn.set_weights(model_weights)
            self.logstd.assign(logstd_weights)

    def update_targ(self,tau):
        """Updates target neural network weights."""
        for base_weight, targ_weight in zip(
            self._nn.variables,self._nn_targ.variables):
            targ_weight.assign((1-tau) * targ_weight + tau * base_weight)
        
        if not self.per_state_std:
            self.logstd_targ.assign(
                (1-tau) * self.logstd_targ + tau * self.logstd)

class GaussianActorwAdversary(GaussianActor):
    """Multivariate Gaussian policy with adversary."""

    def __init__(self,env,layers,activations,init_type,gain,layer_norm,
        std_mult=1.0,per_state_std=True,output_norm=False,adversary_prob=0.0):
        """Initializes multivariate Gaussian policy with adversary.

        Args:
            adversary_prob (float): probability of adversary action
        """

        super(GaussianActorwAdversary,self).__init__(env,layers,activations,
            init_type,gain,layer_norm,std_mult,per_state_std,output_norm)
        
        self.adversary_prob = adversary_prob
       
        if self.per_state_std:
            self._nn_adversary = create_nn(self.s_dim,2*self.a_dim,
                layers,activations,init_type,gain,layer_norm)
            self.adversary_trainable = self._nn_adversary.trainable_variables

            self._nn_adversary_targ = create_nn(self.s_dim,2*self.a_dim,
                layers,activations,init_type,gain,layer_norm)
        else:
            self._nn_adversary = create_nn(self.s_dim,self.a_dim,
                layers,activations,init_type,gain,layer_norm)
            self.logstd_adversary = tf.Variable(
                np.zeros_like(self.logstd_init),dtype=tf.float32)
            self.adversary_trainable = (self._nn_adversary.trainable_variables 
                + [self.logstd_adversary])

            self._nn_adversary_targ = create_nn(self.s_dim,self.a_dim,
                layers,activations,init_type,gain,layer_norm)
            self.logstd_adversary_targ = tf.Variable(
                np.zeros_like(self.logstd_init),dtype=tf.float32)

        self._nn_adversary_targ.set_weights(self._nn_adversary.get_weights())

    def _forward(self,s,targ=False,adversary=False):
        """Returns output of neural network."""
        s_feat = self._transform_state(s)
        
        if adversary:
            if targ:
                a_out = self._nn_adversary_targ(s_feat)
            else:
                a_out = self._nn_adversary(s_feat)
        else:        
            if targ:
                a_out = self._nn_targ(s_feat)
            else:
                a_out = self._nn(s_feat)
        
        if self.per_state_std:
            a_mean, a_std_out = tf.split(a_out,num_or_size_splits=2,axis=-1)
            a_std = tf.math.softplus(a_std_out)
            a_logstd = tf.math.log(a_std)
        else:
            a_mean = a_out

            if adversary:
                if targ:
                    a_logstd = self.logstd_adversary_targ * tf.ones_like(a_mean)
                else:
                    a_logstd = self.logstd_adversary * tf.ones_like(a_mean)
            else:
                if targ:
                    a_logstd = self.logstd_targ * tf.ones_like(a_mean)
                else:
                    a_logstd = self.logstd * tf.ones_like(a_mean)
        
        a_logstd = a_logstd + self.logstd_init
        a_logstd = tf.maximum(a_logstd,tf.math.log(1e-3))

        if self.output_norm:
            a_mean = self._output_normalization(a_mean)

        return a_mean, a_logstd

    def sample(self,s,deterministic=False,targ=False):
        """Samples an action from the current policy or adversary policy."""
        a, a_logstd = self._forward(s,targ=targ,adversary=False)
        
        if self.adversary_prob > 0.0:
            a_adv, a_adv_logstd = self._forward(s,targ=targ,adversary=True)
            u_adv = np.random.random(size=(np.shape(a)[0],1))
            use_adv = (u_adv < self.adversary_prob).astype('float32')

            a = (1-use_adv) * a + use_adv * a_adv
            a_logstd = (1-use_adv) * a_logstd + use_adv * a_adv_logstd

        if not deterministic:
            u = np.random.normal(size=np.shape(a))
            a = a + tf.exp(a_logstd) * u

        if np.shape(a)[0] == 1:
            a = tf.squeeze(a,axis=0)

        return a

    def sample_separate(self,s,deterministic=False,targ=False,adversary=False):
        """Returns action from the current policy or adversary policy."""
        a, a_logstd = self._forward(s,targ=targ,adversary=adversary)
        
        if not deterministic:
            u = np.random.normal(size=np.shape(a))
            a = a + tf.exp(a_logstd) * u

        if np.shape(a)[0] == 1:
            a = tf.squeeze(a,axis=0)

        return a

    def get_adversary_weights(self,flat=False):
        """Returns parameter weights of adversary policy."""
        weights = self._nn_adversary.get_weights()
        if not self.per_state_std:
            weights = weights + [self.logstd_adversary.numpy()]
        
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_adversary_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights of adversary policy."""
        if from_flat:
            weights = flat_to_list(self.adversary_trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_adversary_weights(flat=False)))

        if self.per_state_std:
            self._nn_adversary.set_weights(weights)
        else:
            model_weights = weights[:-1]
            logstd_weights = weights[-1]
            logstd_weights = np.maximum(logstd_weights,np.log(1e-3))
            
            self._nn_adversary.set_weights(model_weights)
            self.logstd_adversary.assign(logstd_weights)

    def update_adversary_targ(self,tau):
        """Updates target neural network weights for adversary."""
        for base_weight, targ_weight in zip(
            self._nn_adversary.variables,self._nn_adversary_targ.variables):
            targ_weight.assign((1-tau) * targ_weight + tau * base_weight)
        
        if not self.per_state_std:
            self.logstd_adversary_targ.assign(
                (1-tau) * self.logstd_adversary_targ + tau * self.logstd_adversary)