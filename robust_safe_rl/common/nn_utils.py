"""Helper functions for creating and updating neural networks."""
import numpy as np
import tensorflow as tf

def create_activations(layers,activations_in):
    """Creates list of TensorFlow activations from string inputs."""
    if len(layers) > 1 and len(activations_in) == 1:
        activations_in = [activations_in[0] for layer in layers]
    activations = []
    for act_str in activations_in:
        if act_str == 'tanh':
            act = tf.keras.activations.tanh
        elif act_str == 'relu':
            act = tf.keras.activations.relu
        elif act_str == 'elu':
            act = tf.keras.activations.elu
        else:
            raise ValueError('activations must be tanh, relu or elu')

        activations.append(act)
    
    return activations

def create_initializer(init_type,gain=None):
    """Creates single TensorFlow initializer."""
    if init_type == 'orthogonal':
        # Standard on-policy initialization used w tanh activations
        if gain:
            init = tf.keras.initializers.Orthogonal(gain=gain)
        else:
            init = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
    elif init_type == 'var':
        # Default initialization used by Acme codebase
        if gain:
            init = tf.keras.initializers.VarianceScaling(
                distribution='uniform',mode='fan_out',scale=gain)
        else:
            init = tf.keras.initializers.VarianceScaling(
                distribution='uniform',mode='fan_out',scale=0.333)
    elif init_type == 'uniform':
        # Default initialization used by softlearning codebase
        init = 'glorot_uniform'
    else:
        raise ValueError('init_type must be orthogonal, var or uniform')
    
    return init

def create_initializations(layers,init_type,gain=0.01):
    """Creates TensorFlow initializations."""
    initializations = []
    for _ in range(len(layers)):
        init = create_initializer(init_type)
        initializations.append(init)
    
    init_final = create_initializer(init_type,gain)
    
    return initializations, init_final

def create_layer(units,initialization,activation=None,in_dim=None):
    """Creates single TensorFlow Dense layer."""

    if in_dim:
        layer = tf.keras.layers.Dense(
            units=units,
            kernel_initializer=initialization,
            activation=activation,
            input_shape=(in_dim,)
        )
    else:
        layer = tf.keras.layers.Dense(
            units=units,
            kernel_initializer=initialization,
            activation=activation
        )
    
    return layer

def transform_features(s):
    """Updates dtype and shape of inputs to neural networks."""   
    s = tf.cast(s,dtype=tf.float32)
    if len(s.shape) == 1:
        s = tf.expand_dims(s,axis=0)

    return s

def create_nn(in_dim,out_dim,layers,activations,init_type,gain,layer_norm=False):
    """Creates neural network.
    
    Args:
        in_dim (int): dimension of neural network input
        out_dim (int): dimension of neural network output
        layers (list): list of hidden layer sizes for neural network
        activations (list): list of activations for neural network
        init_type (str): initialization type
        gain (float): multiplicative factor for final layer initialization
        layer_norm (bool): if True, first layer is layer norm
    
    Returns:
        TensorFlow feedforward neural network.
    """
    nn = tf.keras.Sequential()

    activations = create_activations(layers,activations)
    initializations, init_final = create_initializations(layers,init_type,gain)
    assert len(activations) == len(layers), (
        'activations must be list of length len(layers)')
    
    for layer_idx in range(len(layers)):
        if layer_idx == 0:
            if layer_norm:
                nn.add(create_layer(
                    layers[layer_idx],
                    initializations[layer_idx],
                    None,
                    in_dim
                    )
                )
                nn.add(tf.keras.layers.LayerNormalization())
                nn.add(tf.keras.layers.Activation(tf.nn.tanh))
            else:
                nn.add(create_layer(
                    layers[layer_idx],
                    initializations[layer_idx],
                    activations[layer_idx],
                    in_dim
                    )
                )
        else:
            nn.add(create_layer(
                layers[layer_idx],
                initializations[layer_idx],
                activations[layer_idx]
                )
            )

    nn.add(create_layer(out_dim,init_final))

    return nn

def flat_to_list(trainable,weights):
    """Converts flattened array back into list."""
    shapes = [tf.shape(theta).numpy() for theta in trainable]
    sizes = [tf.size(theta).numpy() for theta in trainable]
    idxs = np.cumsum([0]+sizes)

    weights_list = []

    for i in range(len(shapes)):
        elem_flat = weights[idxs[i]:idxs[i+1]]
        elem = np.reshape(elem_flat,shapes[i])
        weights_list.append(elem)
    
    return weights_list

def list_to_flat(weights):
    """Flattens list into array."""
    weights_flat = np.concatenate(list(map(
                lambda y: np.reshape(y,[-1]),weights)),-1)
    
    return weights_flat

def soft_value(value):
    """Converts value into soft value for TF Variable initializations."""
    if value < 100:
        soft_value = np.log(np.exp(value)-1)
    else:
        soft_value = value
    
    return soft_value