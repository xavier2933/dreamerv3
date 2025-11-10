#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import zmq
import embodied
import json
import time
from pathlib import Path


def tf_to_array(data):
    return np.array(data, dtype=np.float32)


class DreamerInferenceBridge:
    def __init__(self, logdir: str, hz: float = 10.0):
        print(f"[DreamerBridge] Loading DreamerV3 agent from {logdir}")

        # --- Load agent from checkpoint ---
        from dreamerv3 import agent as dreamer_agent
        from embodied.envs import arm
        
        # Load the checkpoint configuration
        ckpt_path = Path(logdir)
        config_file = ckpt_path / "config.yaml"
        
        # Load config from YAML
        import yaml
        
        # Use FullLoader to properly parse scientific notation
        yaml_loader = yaml.FullLoader
        
        # First load the base config from dreamerv3/configs.yaml
        default_config_path = Path(__file__).parent / "dreamerv3" / "configs.yaml"
        print(f"[DreamerBridge] Loading base config from {default_config_path}")
        with open(default_config_path, 'r') as f:
            base_configs = yaml.load(f, Loader=yaml_loader)
        
        # Helper function to deep merge dicts
        def deep_merge(base, override):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        # Start with defaults
        config = base_configs.get('defaults', {}).copy()
        print(f"[DreamerBridge] Loaded defaults, keys: {list(config.keys())[:10]}...")
        
        # Merge with size1m preset if it exists
        if 'size1m' in base_configs:
            config = deep_merge(config, base_configs['size1m'])
            print(f"[DreamerBridge] Merged size1m preset")
        
        # Merge with arm-specific config if it exists
        if 'arm' in base_configs:
            arm_config = base_configs['arm'].copy()
            # Remove the anchor reference if present
            arm_config.pop('<<', None)
            config = deep_merge(config, arm_config)
            print(f"[DreamerBridge] Merged arm config")
        
        # Finally, merge with checkpoint config if it exists
        if config_file.exists():
            with open(config_file, 'r') as f:
                checkpoint_config = yaml.load(f, Loader=yaml_loader)
            config = deep_merge(config, checkpoint_config)
            print(f"[DreamerBridge] Merged with checkpoint config from {config_file}")
        else:
            print(f"[DreamerBridge] No checkpoint config found at {config_file}, using base config")
        
        # Debug: check if critical keys exist
        print(f"[DreamerBridge] Top-level config keys: {list(config.keys())}")
        if 'agent' in config:
            print(f"[DreamerBridge] Agent config type: {type(config['agent'])}")
            if isinstance(config['agent'], dict):
                print(f"[DreamerBridge] Agent keys: {list(config['agent'].keys())[:10]}")
        
        # The config uses dot notation - we need to flatten it properly
        # Convert dot-notation keys to nested structure
        def unflatten_config(flat_config):
            result = {}
            for key, value in flat_config.items():
                if '.' not in key and not key.startswith('.*'):
                    # Direct key
                    if isinstance(value, dict):
                        result[key] = unflatten_config(value)
                    else:
                        result[key] = value
                elif key.startswith('.*\\.'):
                    # Pattern-based config (like .*\.rssm)
                    # Store as-is for now
                    result[key] = value
                else:
                    # Dot-notation key like 'env.arm.data_dir'
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
        print(f"[DreamerBridge] After unflattening, top keys: {list(config.keys())}")
        if 'agent' in config:
            print(f"[DreamerBridge] Agent keys after unflatten: {list(config.get('agent', {}).keys())[:10]}")
        
        if 'enc' not in config and 'agent' in config and isinstance(config['agent'], dict):
            # The enc/dec configs might be inside agent
            print(f"[DreamerBridge] Moving agent subconfigs to top level")
            agent_cfg = config.pop('agent')
            config.update(agent_cfg)
        
        if 'enc' not in config:
            print(f"[DreamerBridge] ERROR: Still no 'enc' in config! Keys: {list(config.keys())}")
            print(f"[DreamerBridge] This might be a config structure issue. Check configs.yaml format.")
        
        # Ensure config is a dict
        if not isinstance(config, dict):
            config = {}
        
        # Make sure jax config exists
        if 'jax' not in config:
            config['jax'] = {}
        if isinstance(config['jax'], dict) and 'platform' not in config.get('jax', {}):
            config['jax']['platform'] = 'cpu'  # Default to CPU for inference

        # Initialize env to extract its spaces (same as training)
        print("[DreamerBridge] Creating arm environment to get spaces")
        env = arm.Arm(task="arm_offline", data_dir="/home/xavie/dreamer/dreamerv3/data/demos/")
        
        obs_space = env.obs_space
        act_space = env.act_space  # Keep the full act_space dict
        
        print(f"[DreamerBridge] Obs space keys: {list(obs_space.keys())}")
        print(f"[DreamerBridge] Act space keys: {list(act_space.keys())}")
        
        # Remove 'reset' from action space if present (agent doesn't like it)
        if 'reset' in act_space:
            print("[DreamerBridge] Removing 'reset' from action space")
            act_space = {k: v for k, v in act_space.items() if k != 'reset'}
        
        print(f"[DreamerBridge] Final act space keys: {list(act_space.keys())}")
        
        # Create a simple config wrapper that supports attribute access
        class ConfigDict(dict):
            def __getattr__(self, key):
                try:
                    value = self[key]
                    # Convert lists to tuples for ninjax compatibility
                    if isinstance(value, list):
                        return tuple(value)
                    return value
                except KeyError:
                    raise AttributeError(f"No attribute '{key}'")
            def __setattr__(self, key, value):
                self[key] = value
            def __getitem__(self, key):
                value = super().__getitem__(key)
                # Convert lists to tuples for ninjax compatibility
                if isinstance(value, list):
                    return tuple(value)
                return value
            def items(self):
                # Convert lists to tuples in items as well
                for k, v in super().items():
                    if isinstance(v, list):
                        yield k, tuple(v)
                    else:
                        yield k, v
        
        # Convert nested dicts to ConfigDict for attribute access
        def dictify(d):
            if isinstance(d, dict):
                result = ConfigDict()
                for k, v in d.items():
                    result[k] = dictify(v)
                return result
            elif isinstance(d, list):
                # Convert lists to tuples for ninjax compatibility
                return tuple(dictify(item) for item in d)
            elif isinstance(d, str):
                # Try to convert strings that look like numbers
                try:
                    # Check for scientific notation or float
                    if 'e' in d.lower() or '.' in d:
                        return float(d)
                    return int(d)
                except (ValueError, AttributeError):
                    return d
            return d
        
        config = dictify(config)
        print("[DreamerBridge] Creating agent")
        self.agent = dreamer_agent.Agent(obs_space, act_space, config)

        # Load weights from checkpoint
        ckpt_dir = ckpt_path / "ckpt"
        
        if not ckpt_dir.exists():
            print(f"[DreamerBridge] Warning: no checkpoint found at {ckpt_dir}")
        else:
            print(f"[DreamerBridge] Found checkpoint directory at {ckpt_dir}")
            try:
                # Read the 'latest' file to find the checkpoint subdirectory
                latest_file = ckpt_dir / "latest"
                latest_name = None
                if latest_file.exists():
                    with open(latest_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            latest_name = content
                            print(f"[DreamerBridge] Latest checkpoint from file: {latest_name}")
                
                # If latest file is empty or missing, find the most recent directory
                if not latest_name:
                    subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir()]
                    if subdirs:
                        latest_name = sorted(subdirs)[-1].name
                        print(f"[DreamerBridge] Found latest checkpoint directory: {latest_name}")
                
                if latest_name:
                    agent_pkl = ckpt_dir / latest_name / "agent.pkl"
                    if agent_pkl.exists():
                        print(f"[DreamerBridge] Loading agent weights from {agent_pkl}")
                        import pickle
                        with open(agent_pkl, 'rb') as f:
                            agent_state = pickle.load(f)
                        
                        # Load the state into the agent
                        if hasattr(self.agent, 'load_state'):
                            self.agent.load_state(agent_state)
                        elif hasattr(self.agent, 'load'):
                            self.agent.load(agent_state)
                        else:
                            # Direct state loading
                            self.agent.__dict__.update(agent_state)
                        print("[DreamerBridge] Successfully loaded checkpoint!")
                    else:
                        print(f"[DreamerBridge] Warning: agent.pkl not found at {agent_pkl}")
                else:
                    print("[DreamerBridge] Could not determine latest checkpoint")
                    
            except Exception as e:
                print(f"[DreamerBridge] Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()

        # Initialize agent state
        self.agent_state = None

        self.rate = hz
        self.dt = 1.0 / hz

        # --- ZeroMQ setup ---
        ctx = zmq.Context()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.connect("tcp://127.0.0.1:5556")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self.pub = ctx.socket(zmq.PUB)
        self.pub.connect("tcp://127.0.0.1:5557")

        print("[DreamerBridge] Connected to ROS bridge via ZeroMQ.")
        self.obs_cache = {}

    def run(self):
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        print("[DreamerBridge] Running inference loop.")
        while True:
            socks = dict(poller.poll(timeout=int(self.dt * 1000)))
            if self.sub in socks:
                msg = self.sub.recv_string()
                data = json.loads(msg)
                self.obs_cache.update(data)

            required = [
                "arm_joints", "block_pose", "target_pose",
                "wrist_angle", "gripper_state", "left_contact", "right_contact"
            ]
            if not all(k in self.obs_cache for k in required):
                continue

            obs = self.obs_cache
            
            # Build observation dictionary matching the space definition
            obs_dict = {
                "arm_joints": tf_to_array(obs["arm_joints"]),  # (9,)
                "block_pose": tf_to_array(obs["block_pose"]),  # (7,)
                "target_pose": tf_to_array(obs["target_pose"]),  # (7,)
                "wrist_angle": tf_to_array(obs["wrist_angle"]).reshape(1),  # (1,)
                "gripper_state": tf_to_array(obs["gripper_state"]).reshape(1),  # (1,)
                "left_contact": tf_to_array(obs["left_contact"]).reshape(1),  # (1,)
                "right_contact": tf_to_array(obs["right_contact"]).reshape(1),  # (1,)
            }
            
            # Add is_first flag for agent state management
            if self.agent_state is None:
                obs_dict['is_first'] = np.array(True, dtype=bool)
            else:
                obs_dict['is_first'] = np.array(False, dtype=bool)

            try:
                # Call agent policy
                action_dict, self.agent_state = self.agent.policy(
                    obs_dict, self.agent_state, mode='eval'
                )

                if isinstance(action_dict, dict):
                    act = action_dict.get("action", None)
                    if act is None:
                        continue
                    act = np.array(act, dtype=np.float32).flatten()
                else:
                    act = np.array(action_dict, dtype=np.float32).flatten()

                msg = json.dumps({"action": act.tolist()})
                self.pub.send_string(msg)
                
            except Exception as e:
                print(f"[DreamerBridge] Error during inference: {e}")
                continue
            
            time.sleep(self.dt)


if __name__ == "__main__":
    logdir = os.getenv("DREAMER_LOGDIR", "/home/xavie/logdir/dreamer/20251109T171831")
    bridge = DreamerInferenceBridge(logdir=logdir, hz=10.0)
    bridge.run()