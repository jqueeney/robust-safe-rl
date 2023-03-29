from PIL import Image
import os
import pickle
import subprocess as sp
import numpy as np
from datetime import datetime

from robust_safe_rl.common.seeding import init_seeds
from robust_safe_rl.analysis.eval_utils import eval_setup


def simulateRGB_setup(setup_kwargs):
    """Sets up environments and actors."""
    import_path = setup_kwargs['import_path']
    import_files = setup_kwargs['import_files']
    import_indices = setup_kwargs['import_indices']

    if import_indices is None:
        import_indices = [0]*len(import_files)
    elif len(import_indices) == 1:
        import_indices = import_indices*len(import_files)
    
    if len(import_indices) < len(import_files):
        import_indices_append = [0]*(len(import_files)-len(import_indices))
        import_indices = import_indices + import_indices_append

    import_logs_all = []
    for idx in range(len(import_files)):
        import_filefull = os.path.join(import_path,import_files[idx])
        with open(import_filefull,'rb') as f:
            import_logs = pickle.load(f)
        import_log = import_logs[import_indices[idx]]
        import_logs_all.append(import_log)
        
    perturb_param_value = setup_kwargs['perturb_param_value']
    import_objects_all = eval_setup(
        perturb_param_value,setup_kwargs,import_logs_all)

    return import_objects_all

def simulateRGB(env,actor,seed,T,deterministic,camera_id,terminate,
    safety_budget,alpha):
    """Returns rendered simulation in RGB array format.
    
    Args:
        env (object): environment
        actor (object): actor
        seed (int): seed to initialize simulation for reproducibility
        T (int): length of simulation
        deterministic (bool): if True, use deterministic actor
        camera_id (int): camera ID for video
        terminate (bool): if True, freeze simulation after termination
        safety_budget (float): safety budget
        alpha (float): strength of safety notifications
    
    Returns:
        List of rendered frames in RGB array format.
    """   
    init_seeds(seed,[env])
    s = env.reset()
    rgb_logger = RGBLogger(env,camera_id,safety_budget,alpha)

    print('Running simulation...')
    sim_done = False
    sim_done_d = False
    J = 0
    Jc = 0
    c = 0
    for t in range(T):
        if terminate and sim_done_d:
            rgb_logger.repeat_frame()
        else:
            rgb_logger.capture_frame(c,Jc)

        s_old = s

        a = actor.sample(s_old,deterministic=deterministic).numpy()
        s, r, d, info = env.step(actor.clip(a))
        c = info.get('cost',np.zeros_like(r))
        J += r
        Jc += c

        if d:
            if not sim_done:
                print('J_r = %4.0f, J_c = %4.0f, t = %4d'%(J,Jc,t))
                sim_done = True
            if d and not sim_done_d:
                sim_done_d = True

    if not sim_done:
        print('J_r = %4.0f, J_c = %4.0f, t = %4d'%(J,Jc,t))
    print('Simulation complete.')
    return rgb_logger.dump()

class RGBLogger:
    """Class for rendering and storing simulations in RGB array format."""

    def __init__(self,env,camera_id,safety_budget,alpha):
        """Initializes RGBLogger."""
        self.env = env
        self.camera_id = camera_id
        self.safety_budget = safety_budget

        self.red = np.expand_dims(np.array([255,0,0],dtype='uint8'),axis=(0,1))
        self.yellow = np.expand_dims(np.array([255,255,0],dtype='uint8'),axis=(0,1))
        self.alpha = alpha

        self.all_frames = []
    
    def capture_frame(self,c=0,Jc=0):
        """Renders and stores frame."""
        frame = self.env.render(mode='rgb_array',camera_id=self.camera_id)
        if Jc > self.safety_budget:
            mix = np.tile(self.red,(self.frame.shape[0],self.frame.shape[1],1))
            frame_norm = self.alpha * (mix/255) + (1-self.alpha) * (frame / 255)
            frame = (frame_norm * 255).astype('uint8')
        elif c > 0:
            mix = np.tile(self.yellow,(self.frame.shape[0],self.frame.shape[1],1))
            frame_norm = self.alpha * (mix/255) + (1-self.alpha) * (frame / 255)
            frame = (frame_norm * 255).astype('uint8')
        
        self.frame = frame
        self.all_frames.append(self.frame)

    def repeat_frame(self):
        """Stores repeat of last rendered frame."""
        self.all_frames.append(self.frame)

    def dump(self):
        """Returns list of stored frames."""
        return self.all_frames

def record_video(rgb_all,save_path,save_file,video_type,fps,
    save_placeholder,image_type):
    """Creates and saves video recording from RGB inputs."""
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    save_file = '%s_%s'%(save_file,save_date)
    recorder = VideoRecorder(save_path,save_file,video_type,fps)
    print('Recording...')
    for idx in range(len(rgb_all[0])):
        rgb_active = []
        for rgb in rgb_all:
            rgb_active.append(rgb[idx])
        recorder.capture_frame(rgb_active)
        if save_placeholder and (idx == 0):
            frame = recorder.get_frame()
            img = Image.fromarray(frame)
            img.save(os.path.join(save_path,'%s.%s'%(save_file,image_type)))
    recorder.close()
    print('Recording complete.')

class VideoRecorder:
    """Class that converts RGB images into video recording."""
    def __init__(self,save_path,save_file,video_type,fps):
        os.makedirs(save_path,exist_ok=True)
        full_save_file = '%s.%s'%(save_file,video_type)
        self.path = os.path.join(save_path,full_save_file)

        self.frames_per_sec = fps
        self.output_frames_per_sec = fps

        self.encoder = None

    def capture_frame(self,frame_list):
        frame0 = frame_list[0]
        for idx in range(len(frame_list)):
            if idx == 0:
                self.frame = frame0
            else:
                white = np.ones((frame0.shape[0],int(frame0.shape[1]*0.1),
                    frame0.shape[2])) * 255
                white = white.astype('uint8')
                
                frame_cur = frame_list[idx]
                self.frame = np.concatenate((self.frame,white,frame_cur),axis=1)
        
        self._encode_image_frame(self.frame)

    def get_frame(self):
        return self.frame
    
    def _encode_image_frame(self,frame):
        if not self.encoder:
            self.encoder = ImageEncoder(self.path,frame.shape,
                self.frames_per_sec,self.output_frames_per_sec)
        
        self.encoder.capture_frame(frame)
    
    def close(self):
        if self.encoder:
            self.encoder.close()
            self.encoder = None

class ImageEncoder:
    """Class that encodes RGB images."""
    def __init__(self,path,frame_shape,frames_per_sec,output_frames_per_sec):
        self.path = path
        h, w, _ = frame_shape
        self.wh = (w,h)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec

        self.start()
    
    def start(self):
        self.cmdline = ('ffmpeg',
                     '-nostats',
                     '-loglevel', 'error', # suppress warnings
                     '-y',

                     # input
                     '-f', 'rawvideo',
                     '-s:v', '{}x{}'.format(*self.wh),
                     '-pix_fmt', 'rgb24',
                     '-framerate', '%d' % self.frames_per_sec,
                     '-i', '-',

                     # output
                     '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                     '-vcodec', 'libx264',
                     '-pix_fmt', 'yuv420p',
                     '-r', '%d' % self.output_frames_per_sec,
                     self.path
                     )
        
        self.proc = sp.Popen(self.cmdline, stdin=sp.PIPE)
    
    def capture_frame(self,frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            print("VideoRecorder encoder exited with status {}".format(ret))