"""Interface to actors."""
import gym

from robust_safe_rl.actors.continuous_actors import GaussianActor
from robust_safe_rl.actors.continuous_actors import GaussianActorwAdversary

def init_actor(env,actor_layers,actor_activations,actor_init_type,actor_gain,
    actor_layer_norm,actor_std_mult,actor_per_state_std,actor_output_norm,
    actor_adversary_prob=0.0,
    actor_weights=None,adversary_weights=None):
    """Creates actor.

    Args:
        env (object): environment
        actor_layers (list): list of hidden layer sizes for neural network
        actor_activations (list): list of activations for neural network
        actor_init_type (str): initialization type
        actor_gain (float): multiplicative factor for final layer initialization
        actor_layer_norm (bool): if True, first layer is layer norm
        actor_std_mult (float): multiplicative factor for diagonal covariance 
            initialization
        actor_per_state_std (bool): if True, state-dependent diagonal covariance
        actor_output_norm (bool): if True, normalize output magnitude
        actor_adversary_prob (float): probability of adversary action
        actor_weights (list): list of actor neural network weights
        adversary_weights (list): list of adversary neural network weights
    
    Returns:
        Actor.
    """

    if isinstance(env.action_space,gym.spaces.Box):
        if (actor_adversary_prob > 0.0):
            actor = GaussianActorwAdversary(env,
                actor_layers,actor_activations,actor_init_type,actor_gain,
                actor_layer_norm,actor_std_mult,actor_per_state_std,
                actor_output_norm,actor_adversary_prob)
        else:
            actor = GaussianActor(env,
                actor_layers,actor_activations,actor_init_type,actor_gain,
                actor_layer_norm,actor_std_mult,actor_per_state_std,
                actor_output_norm)
    else:
        raise TypeError('Only Gym Box action spaces supported')

    if actor_weights is not None:
        actor.set_weights(actor_weights)

    if (adversary_weights is not None) and (actor_adversary_prob > 0.0):
        actor.set_adversary_weights(adversary_weights)

    return actor