"""Helper functions to set random seeds."""
import numpy as np
import tensorflow as tf
import random
import os

def init_seeds(seed,envs=None):
    """Sets random seed."""
    seed = int(seed)
    if envs is not None:
        envs_seeds = np.random.SeedSequence(seed).generate_state(len(envs))
        for idx, env in enumerate(envs):
            env.seed(int(envs_seeds[idx]))
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)