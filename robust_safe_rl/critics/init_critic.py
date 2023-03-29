"""Interface to critics."""
from robust_safe_rl.critics.critics import QCritic

def init_critics(env,critic_layers,critic_activations,critic_init_type,
    critic_gain,critic_layer_norm,critic_weights=None,cost_critic_weights=None):
    """Creates critics.
    
    Args:
        env (object): environment
        critic_layers (list): list of hidden layer sizes for neural network
        critic_activations (list): list of activations for neural network
        critic_init_type (str): initialization type
        critic_gain (float): multiplicative factor for final layer 
            initialization
        critic_layer_norm (bool): if True, first layer is layer norm
        critic_weights (list): list of reward critic neural network weights
        cost_critic_weights (list): list of cost critic neural network weights
    
    Returns:
        Critics for reward and cost.
    """

    critic = QCritic(env,critic_layers,critic_activations,
        critic_init_type,critic_gain,critic_layer_norm)
    cost_critic = QCritic(env,critic_layers,critic_activations,
        critic_init_type,critic_gain,critic_layer_norm,safety=True)

    if critic_weights is not None:
        critic.set_weights(critic_weights)
    if cost_critic_weights is not None:
        cost_critic.set_weights(cost_critic_weights)
    
    return critic, cost_critic