"""Entry point for creating video of results."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from robust_safe_rl.common.train_utils import gather_inputs
from robust_safe_rl.analysis.video_parser import create_video_parser, all_video_kwargs
from robust_safe_rl.analysis.video_utils import simulateRGB_setup
from robust_safe_rl.analysis.video_utils import simulateRGB, record_video

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def main():
    """Parses inputs, creates and saves video."""
    parser = create_video_parser()
    args = parser.parse_args()
    inputs_dict = gather_inputs(args,all_video_kwargs)

    setup_kwargs = inputs_dict['setup_kwargs']
    sim_kwargs = inputs_dict['sim_kwargs']
    video_kwargs = inputs_dict['video_kwargs']

    import_objects_all = simulateRGB_setup(setup_kwargs)

    rgb_all = []
    for import_objects in import_objects_all:
        env = import_objects['env']
        actor = import_objects['actor']
        rgb = simulateRGB(env,actor,**sim_kwargs)
        rgb_all.append(rgb)
    
    record_video(rgb_all,**video_kwargs)


if __name__=='__main__':
    main()