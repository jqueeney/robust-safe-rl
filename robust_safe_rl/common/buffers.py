import numpy as np

def init_buffer(s_dim,a_dim,gamma,buffer_size=None):
    """Creates buffer."""
    return TrajectoryBuffer(s_dim,a_dim,gamma,buffer_size) 

def aggregate_data(data):
    """Combines data along first axis."""
    return np.concatenate(data,0)

class TrajectoryBuffer:
    """Class for storing and processing trajectory data."""

    def __init__(self,s_dim,a_dim,gamma,buffer_size=None):
        """Initializes TrajectoryBuffer class.

        Args:
            s_dim (int): state dimension
            a_dim (int): action dimension
            gamma (float): discount rate
            buffer_size (int): max number of steps to store, if not None
        """
        self.s_dim = s_dim
        self.a_dim = a_dim
        
        self.gamma = gamma

        self.buffer_size = buffer_size

        self.reset()
    
    def reset(self):
        """Resets data storage."""
        self.s_all = np.empty((0,self.s_dim),dtype=np.float32)     # states
        self.a_all = np.empty((0,self.a_dim),dtype=np.float32)     # actions
        self.r_all = np.empty((0,),dtype=np.float32)               # rewards
        self.sp_all = np.empty((0,self.s_dim),dtype=np.float32)    # next states
        self.d_all = np.empty((0,))                                # done flags
        self.c_all = np.empty((0,),dtype=np.float32)               # costs
        self.r_raw_all = np.empty((0,),dtype=np.float32)           # true rewards
        self.idx_all = np.empty((0,))                              # trajectory indices

        self.traj_total = 0         # total trajectory count added to buffer
        self.steps_total = 0        # total step count added to buffer
        self.current_size = 0       # current step count stored in buffer

    def add(self,s_traj,a_traj,r_traj,sp_traj,d_traj,c_traj,r_raw_traj):
        """Stores trajectory data.
        
        Args:
            s_traj (np.ndarray): states
            a_traj (np.ndarray): actions
            r_traj (np.ndarray): rewards
            sp_traj (np.ndarray): next states
            d_traj (np.ndarray): done flags
            c_traj (np.ndarray): costs
            r_raw_traj (np.ndarray): true rewards
        """
        idx_traj = np.ones_like(r_traj) * self.traj_total

        self.s_all = aggregate_data((self.s_all,s_traj))
        self.a_all = aggregate_data((self.a_all,a_traj))
        self.r_all = aggregate_data((self.r_all,r_traj))
        self.sp_all = aggregate_data((self.sp_all,sp_traj))
        self.d_all = aggregate_data((self.d_all,d_traj))
        self.c_all = aggregate_data((self.c_all,c_traj))
        self.r_raw_all = aggregate_data((self.r_raw_all,r_raw_traj))
        self.idx_all = aggregate_data((self.idx_all,idx_traj))

        if self.buffer_size and (len(self.r_all) > self.buffer_size):
            self.s_all = self.s_all[-self.buffer_size:]
            self.a_all = self.a_all[-self.buffer_size:]
            self.r_all = self.r_all[-self.buffer_size:]
            self.sp_all = self.sp_all[-self.buffer_size:]
            self.d_all = self.d_all[-self.buffer_size:]
            self.c_all = self.c_all[-self.buffer_size:]
            self.r_raw_all = self.r_raw_all[-self.buffer_size:]
            self.idx_all = self.idx_all[-self.buffer_size:]
        
        self.current_size = len(self.r_all)
        
        self.traj_total += 1
        self.steps_total += len(r_traj)

    def add_batch(self,s_batch,a_batch,r_batch,sp_batch,d_batch,c_batch,
        r_raw_batch):
        """Stores batch of trajectory data.
        
        Args:
            s_batch (np.ndarray): states
            a_batch (np.ndarray): actions
            r_batch (np.ndarray): rewards
            sp_batch (np.ndarray): next states
            d_batch (np.ndarray): done flags
            c_batch (np.ndarray): costs
            r_raw_batch (np.ndarray): true rewards
        """
        num_traj = len(r_batch)
        for idx in range(num_traj):
            s_traj = s_batch[idx]
            a_traj = a_batch[idx]
            r_traj = r_batch[idx]
            sp_traj = sp_batch[idx]
            d_traj = d_batch[idx]
            c_traj = c_batch[idx]
            r_raw_traj = r_raw_batch[idx]

            if np.any(d_traj):
                term_idx = np.argmax(d_traj) + 1
                
                s_traj = s_traj[:term_idx]
                a_traj = a_traj[:term_idx]
                r_traj = r_traj[:term_idx]
                sp_traj = sp_traj[:term_idx]
                d_traj = d_traj[:term_idx]
                c_traj = c_traj[:term_idx]
                r_raw_traj = r_raw_traj[:term_idx]
            
            self.add(s_traj,a_traj,r_traj,sp_traj,d_traj,c_traj,r_raw_traj)

    def get_offpolicy_info(self,batch_size=None):
        """Returns data needed to calculate off-policy actor and critic updates.
        
        Args:
            batch_size (int): number of samples to return, if not None
        
        Returns:
            States, actions, next states, discounts, rewards, and costs.
        """
        if batch_size:
            idx = np.random.randint(self.current_size,size=batch_size)
            s_sub = self.s_all[idx]
            a_sub = self.a_all[idx]
            sp_sub = self.sp_all[idx]
            d_sub = self.d_all[idx]
            r_sub = self.r_all[idx]
            c_sub = self.c_all[idx] * -1
            disc_sub = (1.-d_sub) * self.gamma
            return s_sub, a_sub, sp_sub, disc_sub, r_sub, c_sub
        else:
            c_all = self.c_all * -1
            disc_all = (1.-self.d_all) * self.gamma
            return (self.s_all, self.a_all, self.sp_all, disc_all, self.r_all, 
                c_all)