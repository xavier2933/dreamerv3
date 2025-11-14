#!/usr/bin/env python3
"""
DreamerV3 client that communicates with ROS via ZeroMQ bridge.
Runs in Python 3.11 venv, bridge runs in Python 3.10 ROS venv.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from pathlib import Path
import yaml
import json
import zmq
import time


class DreamerZMQClient:
    def __init__(self, logdir, hz=10.0):
        self.hz = hz
        self.rate_duration = 1.0 / hz
        
        print(f"[INFO] Loading DreamerV3 agent from {logdir}")
        self.agent = self._create_agent(logdir)
        
        # Agent state (batch_size=1)
        try:
            self.agent_state = self.agent.init_policy(batch_size=1)
        except TypeError:
            self.agent_state = self.agent.init_policy(1)
        
        # Current end-effector state for delta integration
        self.current_target_pos = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.current_wrist_angle = 0.0
        self.current_gripper_state = 0.0
        
        # Action normalization factors (set to 1.0 for now, tune later)
        self.action_scale = np.ones(5)  # [x, y, z, wrist, gripper]
        
        # Track initialization
        self.initialized = False
        self.step_count = 0
        
        # ZeroMQ setup
        print("[INFO] Setting up ZeroMQ connections...")
        ctx = zmq.Context()
        
        # Subscribe to observations from ROS bridge
        self.sub = ctx.socket(zmq.SUB)
        self.sub.connect("tcp://127.0.0.1:5556")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Publish actions to ROS bridge
        self.pub = ctx.socket(zmq.PUB)
        self.pub.connect("tcp://127.0.0.1:5557")
        
        # Give ZMQ time to establish connections
        time.sleep(0.5)
        
        print("[INFO] DreamerV3 ZMQ client initialized")
        print(f"[INFO] Running at {hz} Hz")
        print("[INFO] Waiting for observations from ROS bridge...")
    
    def _create_agent(self, logdir):
        """Create and load the trained agent (same as inference script)."""
        from dreamerv3 import agent as dreamer_agent
        from embodied.envs import arm
        
        ckpt_path = Path(logdir)
        config_file = ckpt_path / "config.yaml"
        
        # Load base config
        default_config_path = Path(__file__).parent / "dreamerv3" / "configs.yaml"
        with open(default_config_path, 'r') as f:
            base_configs = yaml.load(f, Loader=yaml.FullLoader)
        
        def deep_merge(base, override):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        # Build config
        config = base_configs.get('defaults', {}).copy()
        if 'size1m' in base_configs:
            config = deep_merge(config, base_configs['size1m'])
        if 'arm' in base_configs:
            arm_config = base_configs['arm'].copy()
            arm_config.pop('<<', None)
            config = deep_merge(config, arm_config)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
            config = deep_merge(config, checkpoint_config)
        
        # Unflatten config
        def unflatten_config(flat_config):
            result = {}
            for key, value in flat_config.items():
                if '.' not in key and not key.startswith('.*'):
                    if isinstance(value, dict):
                        result[key] = unflatten_config(value)
                    else:
                        result[key] = value
                elif key.startswith('.*\\.'):
                    result[key] = value
                else:
                    parts = key.split('.')
                    current = result
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        elif not isinstance(current[part], dict):
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
            return result
        
        config = unflatten_config(config)
        
        if 'enc' not in config and 'agent' in config and isinstance(config['agent'], dict):
            agent_cfg = config.pop('agent')
            config.update(agent_cfg)
        
        if not isinstance(config, dict):
            config = {}
        
        if 'jax' not in config:
            config['jax'] = {}
        if isinstance(config['jax'], dict) and 'platform' not in config.get('jax', {}):
            config['jax']['platform'] = 'cpu'
        
        # Enable JAX cache
        jax_cache_dir = os.path.join(logdir, 'jax_cache')
        os.makedirs(jax_cache_dir, exist_ok=True)
        os.environ['JAX_COMPILATION_CACHE_DIR'] = jax_cache_dir
        os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
        print(f"[INFO] JAX cache: {jax_cache_dir}")
        
        # Create dummy env to get spaces
        env = arm.Arm(task="arm_offline", data_dir="/home/xavie/dreamer/dreamerv3/data/demos/")
        obs_space = env.obs_space
        act_space = env.act_space
        
        if 'reset' in act_space:
            act_space = {k: v for k, v in act_space.items() if k != 'reset'}
        
        print(f"[INFO] Obs space keys: {list(obs_space.keys())}")
        print(f"[INFO] Act space keys: {list(act_space.keys())}")
        
        # Convert config to attribute-accessible dict
        class ConfigDict(dict):
            def __getattr__(self, key):
                try:
                    value = self[key]
                    if isinstance(value, list):
                        return tuple(value)
                    return value
                except KeyError:
                    raise AttributeError(f"No attribute '{key}'")
            def __setattr__(self, key, value):
                self[key] = value
            def __getitem__(self, key):
                value = super().__getitem__(key)
                if isinstance(value, list):
                    return tuple(value)
                return value
            def items(self):
                for k, v in super().items():
                    if isinstance(v, list):
                        yield k, tuple(v)
                    else:
                        yield k, v
        
        def dictify(d):
            if isinstance(d, dict):
                result = ConfigDict()
                for k, v in d.items():
                    result[k] = dictify(v)
                return result
            elif isinstance(d, list):
                return tuple(dictify(item) for item in d)
            elif isinstance(d, str):
                try:
                    if 'e' in d.lower() or '.' in d:
                        return float(d)
                    return int(d)
                except (ValueError, AttributeError):
                    return d
            return d
        
        config = dictify(config)
        
        print("[INFO] Creating agent")
        agent = dreamer_agent.Agent(obs_space, act_space, config)
        
        # Load checkpoint
        ckpt_dir = ckpt_path / "ckpt"
        if ckpt_dir.exists():
            latest_file = ckpt_dir / "latest"
            latest_name = None
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        latest_name = content
            
            if not latest_name:
                subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir()]
                if subdirs:
                    latest_name = sorted(subdirs)[-1].name
            
            if latest_name:
                agent_pkl = ckpt_dir / latest_name / "agent.pkl"
                if agent_pkl.exists():
                    print(f"[INFO] Loading weights from {agent_pkl}")
                    import pickle
                    with open(agent_pkl, 'rb') as f:
                        agent_state = pickle.load(f)
                    
                    if hasattr(agent, 'load_state'):
                        agent.load_state(agent_state)
                    elif hasattr(agent, 'load'):
                        agent.load(agent_state)
                    else:
                        agent.__dict__.update(agent_state)
                    print("[INFO] Checkpoint loaded!")
        
        return agent
    
    def _parse_obs_from_ros(self, obs_dict):
        """Convert ROS observations to DreamerV3 format (batched)."""
        obs = {}
        
        # Convert each observation to numpy array with batch dimension
        for key in ['arm_joints', 'block_pose', 'target_pose', 'wrist_angle', 
                    'gripper_state', 'left_contact', 'right_contact']:
            if key in obs_dict:
                data = np.array(obs_dict[key], dtype=np.float32)
                obs[key] = data[np.newaxis, ...]  # Add batch dim
            else:
                # Provide defaults if missing
                if key == 'arm_joints':
                    obs[key] = np.zeros((1, 6), dtype=np.float32)
                elif key in ['block_pose', 'target_pose']:
                    obs[key] = np.zeros((1, 7), dtype=np.float32)
                elif key == 'wrist_angle':
                    obs[key] = np.zeros((1, 1), dtype=np.float32)
                elif key in ['gripper_state', 'left_contact', 'right_contact']:
                    obs[key] = np.zeros((1, 1), dtype=np.float32)
        
        # Add standard DreamerV3 keys
        obs['is_first'] = np.array([self.step_count == 0], dtype=bool)
        obs['is_last'] = np.array([False], dtype=bool)
        obs['is_terminal'] = np.array([False], dtype=bool)
        obs['reward'] = np.array([0.0], dtype=np.float32)
        
        # Update current target position from observations
        if 'target_pose' in obs_dict:
            self.current_target_pos = np.array(obs_dict['target_pose'][:3])
        if 'wrist_angle' in obs_dict:
            self.current_wrist_angle = obs_dict['wrist_angle'][0]
        if 'gripper_state' in obs_dict:
            self.current_gripper_state = obs_dict['gripper_state'][0]
        
        return obs
    
    def _check_obs_complete(self, obs_dict):
        """Check if all required observations are present."""
        required = ['arm_joints', 'block_pose', 'target_pose', 'wrist_angle',
                   'gripper_state', 'left_contact', 'right_contact']
        return all(key in obs_dict for key in required)
    
    def run(self):
        """Main control loop."""
        print("\n[INFO] Starting control loop...")
        
        last_step_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                # Receive observations from ROS bridge (non-blocking)
                try:
                    msg_str = self.sub.recv_string(flags=zmq.NOBLOCK)
                    obs_dict = json.loads(msg_str)
                    
                    # Check if we have all observations
                    if not self._check_obs_complete(obs_dict):
                        if not self.initialized:
                            print("[INFO] Waiting for complete observations...", end='\r')
                        continue
                    
                    if not self.initialized:
                        print("\n[INFO] All observations received! Starting inference.")
                        self.initialized = True
                    
                    # Convert to DreamerV3 format
                    obs = self._parse_obs_from_ros(obs_dict)
                    
                    # Run policy
                    result = self.agent.policy(self.agent_state, obs, mode='eval')
                    
                    if len(result) == 3:
                        self.agent_state, act, _ = result
                    elif len(result) == 2:
                        self.agent_state, act = result
                    else:
                        print(f"[ERROR] Unexpected policy return format: {len(result)} values")
                        continue
                    
                    # Extract action
                    if isinstance(act, dict):
                        action = act.get('action', act)
                    else:
                        action = act
                    
                    # Remove batch dim and ensure 1-D
                    if hasattr(action, 'ndim') and action.ndim > 1:
                        action = action[0]
                    action = np.array(action, dtype=np.float32).flatten()
                    
                    # Action is normalized delta: [dx, dy, dz, d_wrist, d_gripper]
                    # Un-normalize
                    delta = action * self.action_scale
                    
                    # Integrate to get absolute commands
                    new_pos = self.current_target_pos + delta[:3]
                    new_wrist = self.current_wrist_angle + delta[3]
                    new_gripper = np.clip(self.current_gripper_state + delta[4], 0.0, 1.0)
                    
                    # Package action for ROS bridge
                    # Bridge expects [x, y, z, wrist, gripper] as absolute positions
                    action_msg = {
                        "action": [
                            float(new_pos[0]),
                            float(new_pos[1]),
                            float(new_pos[2]),
                            float(new_wrist),
                            float(new_gripper)
                        ]
                    }
                    
                    # Send action to ROS bridge
                    self.pub.send_string(json.dumps(action_msg))
                    
                    # Update current state
                    self.current_target_pos = new_pos
                    self.current_wrist_angle = new_wrist
                    self.current_gripper_state = new_gripper
                    
                    self.step_count += 1
                    
                    if self.step_count % 50 == 0:
                        print(f"[Step {self.step_count}] pos={np.round(new_pos, 3)}, "
                              f"wrist={new_wrist:.2f}, gripper={new_gripper:.2f}")
                
                except zmq.Again:
                    # No message available, continue
                    pass
                
                except Exception as e:
                    print(f"[ERROR] Control loop error: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = self.rate_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, help='Path to DreamerV3 checkpoint')
    parser.add_argument('--hz', type=float, default=10.0, help='Control frequency')
    args = parser.parse_args()
    
    client = DreamerZMQClient(args.logdir, args.hz)
    client.run()


if __name__ == "__main__":
    main()