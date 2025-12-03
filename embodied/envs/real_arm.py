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
    
        # Reduced to 5mm (0.005) for stability on 2025-11-30
        self.action_scale = np.array([0.005, 0.005, 0.005, 2.0, 1.0])
        
        # Workspace limits (Must match bridge.py clamping!)
        # Bridge: X[-0.2, 0.2], Y[0.15, 0.5], Z[0.2, 0.5]
        self.pos_min = np.array([-0.2, 0.15, 0.2])
        self.pos_max = np.array([0.2, 0.5, 0.5])
        self.wrist_min = -180.0
        self.wrist_max = 180.0
        
        # Reward Function
        from embodied.envs.reward_function import SimpleReachReward
        self.reward_fn = SimpleReachReward(target_pos=np.array([0.1, 0.35, 0.35]))
        
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
            'actual_pose': elements.Space(np.float32, (7,)),
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


    def _parse_obs(self, obs_dict, is_first=False):
        obs = {}
        
        # Handle missing block_pose by defaulting to zeros
        if 'block_pose' not in obs_dict:
            obs_dict['block_pose'] = [0.0] * 7
            
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
        if 'actual_pose' in obs_dict:
            self.current_target_pos = np.array(obs_dict['actual_pose'][:3])
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
        
        # 2. Send Action (Raw Normalized Deltas)
        # Bridge expects: [dx, dy, dz, dwrist, dgrip] in normalized space
        # We simply pass the agent's output directly.
        action_msg = {
            "action": [float(x) for x in act]
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

    def _receive_obs(self, timeout_ms=5000):
        """Blocking receive of observation with a custom timeout."""
        self.sub.setsockopt(zmq.RCVTIMEO, timeout_ms)
        while True:
            try:
                msg_str = self.sub.recv_string()
                obs_dict = json.loads(msg_str)
                
                # Check completeness - removed block_pose from strict requirement
                required = ['arm_joints', 'actual_pose', 'wrist_angle',
                            'gripper_state', 'left_contact', 'right_contact']
                if all(k in obs_dict for k in required):
                    # Restore default timeout for subsequent calls (e.g., from step())
                    self.sub.setsockopt(zmq.RCVTIMEO, 5000) 
                    return obs_dict
            except zmq.Again:
                # This exception is raised if RCVTIMEO is set and no message arrives
                self.sub.setsockopt(zmq.RCVTIMEO, 5000)
                raise TimeoutError("Timeout waiting for observation during reset.")
            except Exception as e:
                print(f"[RealArm] Error receiving obs: {e}")
                time.sleep(0.1)

    def reset(self):
        print("[RealArm] Resetting...")
        
        # Reset reward function state
        self.reward_fn.reset()
        
        # --- 1. Send reset command ---
        self.pub.send_string(json.dumps({"reset": True}))
        # Wait a very short moment for the command to be picked up by the bridge
        time.sleep(0.1)
        
        # --- 2. Clear all observations BEFORE and DURING the reset motion ---
        print("[RealArm] Clearing stale/transient messages...")
        
        # Temporarily set a short timeout for polling
        self.sub.setsockopt(zmq.RCVTIMEO, 100) # 100ms timeout
        
        clear_start_time = time.time()
        # The bridge implements a 0.5s + 3.0s = 3.5s wait, use 4.0s to be safe
        timeout_duration = 4.0 
        
        # Continuously poll and clear the socket until the motion is expected to be complete
        msgs_cleared = 0
        while (time.time() - clear_start_time) < timeout_duration:
            try:
                # Read without blocking until it times out (100ms)
                # We don't need to parse the JSON, just drain the queue
                self.sub.recv_string()
                msgs_cleared += 1
            except zmq.Again:
                # Timeout occurred, meaning the queue is empty *for now*. 
                # Keep looping until the total timeout is reached.
                time.sleep(0.01) # Small sleep to avoid hogging the CPU
                
        print(f"[RealArm] Cleared {msgs_cleared} stale/transient messages.")
        
        # --- 3. Receive the guaranteed fresh observation ---
        # Note: self._receive_obs() will automatically restore the 5000ms timeout.
        print("[RealArm] Waiting for first post-reset observation...")
        
        try:
            # Wait with a generous timeout for the first guaranteed fresh message
            obs_dict = self._receive_obs(timeout_ms=10000) # 10 second safety timeout
        except TimeoutError:
            print("[RealArm] FATAL: Failed to receive post-reset observation.")
            # Depending on your setup, you might want to exit or raise here.
            raise
        
        self.step_count = 0
        
        print("[RealArm] Reset complete and current pose confirmed!")
        return self._parse_obs(obs_dict, is_first=True)

