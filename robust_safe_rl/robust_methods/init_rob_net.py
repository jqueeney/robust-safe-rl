"""Interface to robust perturbation classes."""
from robust_safe_rl.robust_methods.otp import OTP
from robust_safe_rl.robust_methods.ramu import RAMU

def init_rob_net(env,robust_type,
    otp_layers,otp_activations,otp_init_type,otp_gain,otp_layer_norm,
    rob_weights,rob_setup_kwargs,safety_kwargs):
    """Creates robust perturbation class.

    Args:
        env (object): environment
        robust_type (str): robust perturbation type (otp, ramu)

        otp_layers (list): list of hidden layer sizes for OTP NN
        otp_activations (list): list of activations for OTP NN
        otp_init_type (str): initialization type for OTP NN
        otp_gain (float): mult factor for final layer OTP NN init
        otp_layer_norm (bool): if True, first layer of OTP NN is layer norm
        rob_weights (list): list of OTP NN weights

        rob_setup_kwargs (dict): robustness perturbation setup parameters
        safety_kwargs (dict): safety parameters
    
    Returns:
        Class that implements robust perturbations.
    """

    if robust_type == 'otp':
        rob_net = OTP(env,
            otp_layers,otp_activations,otp_init_type,otp_gain,otp_layer_norm,
            rob_setup_kwargs,safety_kwargs)
    elif robust_type == 'ramu':
        rob_net = RAMU(env,rob_setup_kwargs,safety_kwargs)
    else:
        raise ValueError('invalid perturbation type')

    if rob_weights is not None:
        rob_net.set_weights(rob_weights)

    return rob_net