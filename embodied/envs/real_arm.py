import time
import json
import zmq
import numpy as np
import embodied
import elements

class RealArm(embodied.Env):
    
    def __init__(self, task, ip='127.0.0.1', port_sub=5556, port_pub=5557, hz=10.0):
        self.hz = hz
        self.rate_duration = 1.0 / hz
        
        # Action scaling (must match inference.py)
        self.action_scale = np.array([0.041028, 0.055041, 0.046091, 13.39, 1.0])
        
        # Workspace limits
        self.pos_min = np.array([-2.0, -2.0, -2.0])
        self.pos_max = np.array([2.0, 2.0, 2.0])
        self.wrist_min = -180.0
        self.wrist_max = 180.0
        
        # Reward Function
        from embodied.envs.reward_function import DipLiftReward
        self.reward_fn = DipLiftReward()
        
        # ZMQ Setup
        print(f"[RealArm] Connecting to ZMQ bridge at {ip}...")
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{ip}:{port_sub}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{ip}:{port_pub}")
        
        # State tracking
        self.current_target_pos = np.zeros(3)
        self.current_wrist_angle = 0.0
        self.current_gripper_state = 0.0
        self.step_count = 0
        
        # Define spaces
        self._obs_space = {
            'arm_joints': elements.Space(np.float32, (6,)),
            'block_pose': elements.Space(np.float32, (7,)),
            'target_pose': elements.Space(np.float32, (7,)),
            'wrist_angle': elements.Space(np.float32, (1,)),
            'gripper_state': elements.Space(np.float32, (1,)),
            'left_contact': elements.Space(np.float32, (1,)),
            'right_contact': elements.Space(np.float32, (1,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
        
        self._act_space = {
            'action': elements.Space(np.float32, (5,)),
            'reset': elements.Space(bool),
        }
        
        print("[RealArm] Initialized")

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def act_space(self):
        return self._act_space

    def _receive_obs(self):
        """Blocking receive of observation."""
        while True:
            try:
                msg_str = self.sub.recv_string()
                obs_dict = json.loads(msg_str)
                
                # Check completeness
                required = ['arm_joints', 'block_pose', 'target_pose', 'wrist_angle',
                           'gripper_state', 'left_contact', 'right_contact']
                if all(k in obs_dict for k in required):
                    return obs_dict
            except Exception as e:
                print(f"[RealArm] Error receiving obs: {e}")
                time.sleep(0.1)

    def _parse_obs(self, obs_dict, is_first=False):
        obs = {}
        for k, v in obs_dict.items():
            if k in self._obs_space:
                val = np.array(v, dtype=np.float32)
                if k == 'arm_joints' and val.shape[0] > 6:
                    val = val[:6]
                obs[k] = val
        
        # Ensure shapes match space
        for k, space in self._obs_space.items():
            if k not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if k not in obs:
                    obs[k] = np.zeros(space.shape, dtype=space.dtype)
                else:
                    # Handle scalar vs vector mismatch
                    if obs[k].shape != space.shape:
                        obs[k] = obs[k].reshape(space.shape)

        # Compute reward using stateful function
        reward = self.reward_fn(obs)
        
        obs['reward'] = np.float32(reward)
        obs['is_first'] = is_first
        obs['is_last'] = False
        obs['is_terminal'] = False
        
        # Update internal state for integration
        if 'target_pose' in obs_dict:
            self.current_target_pos = np.array(obs_dict['target_pose'][:3])
        if 'wrist_angle' in obs_dict:
            self.current_wrist_angle = obs_dict['wrist_angle'][0]
        if 'gripper_state' in obs_dict:
            self.current_gripper_state = obs_dict['gripper_state'][0]
            
        return obs

    def step(self, action):
        loop_start = time.time()
        
        # 1. Process Action
        if action.get('reset', False):
            return self.reset()
            
        act = action['action']
        
        # Un-normalize
        delta = act * self.action_scale
        
        # Integrate
        new_pos = self.current_target_pos + delta[:3]
        new_wrist = self.current_wrist_angle + delta[3]
        new_gripper = np.clip(self.current_gripper_state + delta[4], 0.0, 1.0)
        
        # Clip
        new_pos = np.clip(new_pos, self.pos_min, self.pos_max)
        new_wrist = np.clip(new_wrist, self.wrist_min, self.wrist_max)
        
        # 2. Send Action
        action_msg = {
            "action": [
                float(new_pos[0]),
                float(new_pos[1]),
                float(new_pos[2]),
                float(new_wrist),
                float(new_gripper)
            ]
        }
        self.pub.send_string(json.dumps(action_msg))
        
        # 3. Wait for cycle time (rate limiting)
        elapsed = time.time() - loop_start
        sleep_time = self.rate_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        # 4. Receive Next Observation
        obs_dict = self._receive_obs()
        self.step_count += 1
        
        return self._parse_obs(obs_dict, is_first=False)

    def reset(self):
        print("[RealArm] Resetting...")
        
        # Reset reward function state
        self.reward_fn.reset()
        
        # Send reset command twice to ensure it goes through
        self.pub.send_string(json.dumps({"reset": True}))
        time.sleep(2.0)
        
        self.pub.send_string(json.dumps({"reset": True}))
        time.sleep(2.0)
        
        obs_dict = self._receive_obs()
        self.step_count = 0
        return self._parse_obs(obs_dict, is_first=True)
