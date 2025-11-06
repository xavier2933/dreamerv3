import os
import numpy as np
import elements
import embodied


class Arm(embodied.Env):
    """
    Offline replay environment for DreamerV3.
    Uses preprocessed numpy arrays derived from ROS bag data.
    """

    def __init__(self, data_dir, episode_length=None):
        """
        Args:
            data_dir: path containing npz or npy arrays, e.g.:
                - obs.npz: contains keys like arm_joints, block_pose, etc.
                - actions.npy: [T, action_dim]
                - rewards.npy (optional)
            episode_length: override length per episode, else inferred
        """
        self.data_dir = data_dir

        # ---- Load data ----
        self.obs_data = np.load(os.path.join(data_dir, "obs.npz"))
        self.actions = np.load(os.path.join(data_dir, "actions.npy"))
        self.rewards = (
            np.load(os.path.join(data_dir, "rewards.npy"))
            if os.path.exists(os.path.join(data_dir, "rewards.npy"))
            else np.zeros(len(self.actions), np.float32)
        )

        # infer episode length
        self.length = episode_length or len(self.actions)
        self.t = 0
        self.done = False

        # ---- Observation keys ----
        # Expected arrays in obs.npz:
        # arm_joints [T, N], block_pose [T, 7], target_pose [T, 7],
        # wrist_angle [T, 1], gripper_state [T, 1], contact [T, 1]
        self.obs_keys = list(self.obs_data.keys())
        example_obs = self._get_obs(0)

        # ---- Define spaces ----
        self._obs_space = {
            k: elements.Space(np.float32, v.shape[1:])
            for k, v in example_obs.items()
        }
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

    # ----------------------------------------------------------------------

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def act_space(self):
        return self._act_space

    # ----------------------------------------------------------------------

    def step(self, action):
        if action.pop("reset") or self.done:
            self.t = 0
            self.done = False
            return self._format_obs(self.t, is_first=True)

        self.t += 1
        if self.t >= self.length - 1:
            self.done = True

        return self._format_obs(self.t, is_last=self.done, is_terminal=self.done)

    # ----------------------------------------------------------------------

    def _get_obs(self, idx):
        """Return a dict of numpy arrays at a specific timestep."""
        obs = {k: self.obs_data[k][idx].astype(np.float32) for k in self.obs_keys}
        return obs

    def _format_obs(self, idx, is_first=False, is_last=False, is_terminal=False):
        obs = self._get_obs(idx)
        obs.update({
            "reward": np.float32(self.rewards[idx]),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        })
        return obs
