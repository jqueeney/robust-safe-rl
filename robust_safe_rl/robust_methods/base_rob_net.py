import gym
import tensorflow as tf

from robust_safe_rl.common.nn_utils import transform_features

class RobustNet:
    """Class for next state perturbations."""

    def __init__(self,env,rob_setup_kwargs,safety_kwargs):
        """Initializes RobustNet class.

        Args:
            env (object): environment
            rob_setup_kwargs (dict): robustness perturbation setup parameters
            safety_kwargs (dict): safety parameters
        """
        
        self._setup(rob_setup_kwargs,safety_kwargs)
        
        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)

        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

    def _setup(self,rob_setup_kwargs,safety_kwargs):
        """Sets up hyperparameters as class attributes."""
        self.rob_magnitude = rob_setup_kwargs['rob_magnitude']
        self.rob_out_max = rob_setup_kwargs['rob_out_max']
        if (self.rob_out_max is None):
            self.rob_out_max = 2 * self.rob_magnitude
        self.rob_reward_attitude = rob_setup_kwargs['rob_reward_attitude']

        self.safe = safety_kwargs['safe']

    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        self.s_rms, _, _, self.delta_rms, _, _ = all_rms

    def _transform_delta(self,s,sp):
        """Preprocesses state deltas."""
        delta = sp - s
        delta_norm = self.delta_rms.normalize(delta,center=False)
        delta_feat = transform_features(delta_norm)

        return delta_feat

    def _output_normalization(self,delta_rob,out_max):
        """Normalizes perturbation output via clipping."""
        return tf.clip_by_value(delta_rob,out_max*-1,out_max)

    def _calc_sp(self,s,delta,delta_perturb):
        """Returns perturbed next state (denormalized)."""
        # normalize output
        if self.rob_out_max:
            delta_perturb_clip = self._output_normalization(
                delta_perturb,self.rob_out_max)
        else:
            delta_perturb_clip = delta_perturb

        # apply perturbation
        delta_final = delta * (1. + delta_perturb_clip)

        # output raw next state
        delta_final = self.delta_rms.denormalize(delta_final,center=False)
        sp_final = s + delta_final
        
        return sp_final

    def _get_rob_loss(self,delta_rob):
        """Returns robustness perturbation loss per sample."""
        return tf.reduce_mean(tf.square(delta_rob),axis=-1)

    def get_rob_magnitude(self,delta_rob):
        """Returns average robustness magnitude per sample."""
        return tf.reduce_mean(tf.abs(delta_rob),axis=-1)

    def sample(self,s,a,sp):
        """Returns perturbed next state samples and perturbation outputs."""
        raise NotImplementedError

    def loss_and_targets(self,disc,rtg_next_all,ctg_next_all,
        delta_out,delta_out_cost):
        """Returns loss function and critic targets."""
        raise NotImplementedError

    def apply_updates(self,tape,rob_loss):
        """Applies gradient updates."""
        raise NotImplementedError

    def get_weights(self):
        """Returns parameter weights for robustness perturbation networks."""
        raise NotImplementedError

    def set_weights(self,weights):
        """Sets parameter weights for robustness perturbation networks."""
        raise NotImplementedError