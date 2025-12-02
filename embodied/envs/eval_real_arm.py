import time
import json
import zmq
import numpy as np
import embodied
import elements
from embodied.envs import real_arm


class EvalRealArm(embodied.Env):
    """
    Evaluation version: does NOT move the robot.
    Just simulates a no-op environment to let Dreamer run evaluation episodes.
    """

    def __init__(self, real_env: real_arm.RealArm):
        # Copy spaces
        self._obs_space = real_env.obs_space
        self._act_space = real_env.act_space
        
        # Start from a real reset observation
        self._last_obs = real_env.reset()
        self.reward_fn = real_env.reward_fn

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def act_space(self):
        return self._act_space

    def reset(self):
        # Produce a fake-but-valid reset obs
        self.reward_fn.reset()
        
        obs = self._last_obs.copy()
        obs['reward'] = np.float32(0.0)
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        return obs

    def step(self, action):
        # DO NOT SEND ANY ACTION TO ARM
        # Just simulate “no change" observations

        obs = self._last_obs.copy()

        # recompute reward on the static obs
        obs['reward'] = np.float32(self.reward_fn(obs))
        obs['is_first'] = False
        obs['is_last'] = False
        obs['is_terminal'] = False

        self._last_obs = obs
        return obs

