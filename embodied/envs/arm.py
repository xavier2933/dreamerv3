import os
import glob
import numpy as np
import elements
import embodied
from .reward_function import compute_reward  # Import the reward function



class Arm(embodied.Env):
    """
    Offline replay environment for DreamerV3.
    Uses preprocessed numpy arrays derived from ROS bag data.
    Supports loading multiple demonstrations from subdirectories.
    """

    def __init__(self, task, data_dir='offline', **kwargs):
        self.data_dir = data_dir
        
        # Load data from multiple demos
        demo_dirs = self._find_demo_dirs(data_dir)
        
        all_obs = {}
        all_actions = []
        all_rewards = []
        episode_ends = []
        cumulative_length = 0
        
        for demo_dir in demo_dirs:
            print(f"[INFO] Loading demo from: {demo_dir}")
            
            obs_data = np.load(os.path.join(demo_dir, "obs.npz"))
            actions = np.load(os.path.join(demo_dir, "actions.npy"))
            
            # COMPUTE REWARDS FROM OBSERVATIONS instead of loading
            rewards = self._compute_rewards_from_obs(obs_data)
            
            # ... rest of your loading code ...
            if not all_obs:
                for k in obs_data.keys():
                    all_obs[k] = []
            
            for k in obs_data.keys():
                all_obs[k].append(obs_data[k])
            
            all_actions.append(actions)
            all_rewards.append(rewards)
            
            cumulative_length += len(actions)
            episode_ends.append(cumulative_length)
            
            print(f"  Loaded {len(actions)} timesteps, computed reward: {rewards.sum():.2f}")
        
        # Concatenate all demos
        self.obs_data = {k: np.concatenate(v, axis=0) for k, v in all_obs.items()}
        self.actions = np.concatenate(all_actions, axis=0)
        self.rewards = np.concatenate(all_rewards, axis=0)
        self.episode_ends = np.array(episode_ends)
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.num_episodes = len(self.episode_ends)
        
        # Start with first episode
        self.current_episode = 0
        self.episode_start = self.episode_starts[0]
        self.episode_end = self.episode_ends[0]
        self.t = self.episode_start
        self.done = False

        # ---- Observation keys ----
        self.obs_keys = list(self.obs_data.keys())
        
        # ---- Define spaces ----
        # Build observation space from actual data shapes
        self._obs_space = {}
        for k in self.obs_keys:
            data_shape = self.obs_data[k].shape
            if len(data_shape) == 1:
                # Scalar per timestep
                obs_shape = ()
            else:
                # Vector per timestep - take shape from [T, ...] -> [...]
                obs_shape = data_shape[1:]
            self._obs_space[k] = elements.Space(np.float32, obs_shape)
        
        # Add standard DreamerV3 observation keys
        self._obs_space.update({
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        })

        self._act_space = {
            "reset": elements.Space(bool),
            "action": elements.Space(np.float32, (self.actions.shape[1],)),
        }

        # Print summary
        print(f"[INFO] Loaded {self.num_episodes} episodes with {len(self.actions)} total timesteps")
        ep_lengths = self.episode_ends - self.episode_starts
        print(f"[INFO] Episode lengths: min={ep_lengths.min()}, max={ep_lengths.max()}, mean={ep_lengths.mean():.1f}")
        print(f"[INFO] Total reward: {self.rewards.sum():.2f}")
        print(f"[INFO] Observation spaces:")
        for k, v in self._obs_space.items():
            if k not in ["reward", "is_first", "is_last", "is_terminal"]:
                print(f"  {k}: {v}")

    def _find_demo_dirs(self, data_dir):
        """
        Find demonstration directories. Returns list of paths containing demo data.
        Checks for:
        1. Direct files in data_dir (single demo)
        2. Subdirectories containing demo files (multiple demos)
        """
        # Check if data_dir itself contains demo files
        if (os.path.exists(os.path.join(data_dir, "obs.npz")) and
            os.path.exists(os.path.join(data_dir, "actions.npy"))):
            return [data_dir]
        
        # Look for subdirectories with demo files
        demo_dirs = []
        for root, dirs, files in os.walk(data_dir):
            if 'obs.npz' in files and 'actions.npy' in files:
                demo_dirs.append(root)
        
        if not demo_dirs:
            raise ValueError(f"No demo data found in {data_dir}. "
                           f"Expected either direct files (obs.npz, actions.npy) "
                           f"or subdirectories containing these files.")
        
        return demo_dirs

    # ----------------------------------------------------------------------

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def act_space(self):
        return self._act_space

    # ----------------------------------------------------------------------

    def _compute_rewards_from_obs(self, obs_data):
        """
        Compute rewards for an entire trajectory using the reward function.
        
        Args:
            obs_data: dict from np.load("obs.npz"), e.g. {'gripper_pos': array(T, 3), ...}
        
        Returns:
            np.ndarray: rewards of shape (T,)
        """
        # Get trajectory length
        first_key = list(obs_data.keys())[0]
        T = len(obs_data[first_key])
        
        rewards = np.zeros(T, dtype=np.float32)
        
        for t in range(T):
            # Extract observation at timestep t
            obs_t = {k: obs_data[k][t] for k in obs_data.keys()}
            
            # Compute reward using the reward function
            rewards[t] = compute_reward(obs_t)
        
        return rewards
    

    def reset(self):
        """Reset the environment to the start of the next episode."""
        # Cycle through episodes
        self.current_episode = (self.current_episode + 1) % self.num_episodes
        self.episode_start = self.episode_starts[self.current_episode]
        self.episode_end = self.episode_ends[self.current_episode]
        self.t = self.episode_start
        self.done = False
        return self._format_obs(self.t, is_first=True)


    def step(self, action):

        if self.t >= self.episode_end:
            print(f"[WARN] Step called past end of episode ({self.t} >= {self.episode_end}), auto-resetting.")
            self.done = True
            return self._format_obs(self.episode_end - 1, is_last=True, is_terminal=True)
        
        if action.get("reset", False) or self.done:
            # Move to next episode (cycle through all episodes)
            self.current_episode = (self.current_episode + 1) % self.num_episodes
            self.episode_start = self.episode_starts[self.current_episode]
            self.episode_end = self.episode_ends[self.current_episode]
            self.t = self.episode_start
            self.done = False
            return self._format_obs(self.t, is_first=True)

        # ---- Check for last timestep BEFORE incrementing ----
        if self.t >= self.episode_end - 1:
            self.done = True
            return self._format_obs(self.episode_end - 1, is_last=True, is_terminal=True)

        # ---- Safe to increment ----
        self.t += 1
        return self._format_obs(self.t, is_last=False, is_terminal=False)


    # ----------------------------------------------------------------------

    def _get_obs(self, idx):
        """Return a dict of numpy arrays at a specific timestep."""
        obs = {}
        for k in self.obs_keys:
            data = self.obs_data[k][idx]
            # Ensure we handle both scalar and vector observations
            if self.obs_data[k].shape == (len(self.obs_data[k]),):
                # Scalar observation
                obs[k] = np.float32(data)
            else:
                # Vector observation
                obs[k] = data.astype(np.float32)
        return obs

    def _format_obs(self, idx, is_first=False, is_last=False, is_terminal=False):
        obs = self._get_obs(idx)
        
        # Handle potential scalar vs array for rewards
        reward_val = self.rewards[idx]
        if isinstance(reward_val, np.ndarray):
            reward_val = reward_val.item() if reward_val.size == 1 else reward_val[0]
        
        obs.update({
            "reward": np.float32(reward_val),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        })
        return obs