#!/usr/bin/env python3
"""
Simple script to run DreamerV3 inference on a saved demo.
Tests the agent without ROS complexity.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from pathlib import Path
import yaml


def load_demo(demo_dir):
    """Load a single demo from disk."""
    obs_data = np.load(os.path.join(demo_dir, "obs.npz"))
    actions = np.load(os.path.join(demo_dir, "actions.npy"))
    
    print(f"[INFO] Loaded demo from {demo_dir}")
    print(f"[INFO] Timesteps: {len(actions)}")
    print(f"[INFO] Observation keys: {list(obs_data.keys())}")
    print(f"[INFO] Action shape: {actions.shape}")
    
    return obs_data, actions


def create_agent(logdir):
    """Create and load the trained agent."""
    print(f"[INFO] Loading agent from {logdir}")
    
    from dreamerv3 import agent as dreamer_agent
    from embodied.envs import arm
    
    # Load config
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


def format_observation(obs_data, t):
    """
    Format a single timestep from the demo data into the agent's expected format.
    This should match what the training environment returns.
    """
    obs = {}
    
    # Extract each observation component and add batch dimension
    for key in obs_data.keys():
        data = obs_data[key][t]
        # Ensure proper dtype and add batch dimension
        obs[key] = data.astype(np.float32)[np.newaxis, ...]  # Add batch dim
    
    # Add standard DreamerV3 keys with batch dimension
    obs['is_first'] = np.array([t == 0], dtype=bool)  # Shape: (1,)
    obs['is_last'] = np.array([False], dtype=bool)     # Shape: (1,)
    obs['is_terminal'] = np.array([False], dtype=bool) # Shape: (1,)
    obs['reward'] = np.array([0.0], dtype=np.float32)  # Shape: (1,)
    
    return obs


def run_closed_loop(agent, rollout_steps=300):
    """
    Run Dreamer in closed-loop mode using its own predicted actions.
    Accepts an environment that returns plain numpy scalars/arrays and
    converts them into batched arrays for agent.policy().
    """
    from embodied.envs import arm
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Create environment (same as training wrapper)
    env = arm.Arm(task="arm_offline", data_dir="/home/xavie/dreamer/dreamerv3/data/demos/")

    # Reset env and prepare initial obs as batched dict
    raw_obs = env.reset()

    # Helper: convert env obs (scalars or arrays) -> batched numpy arrays (batch dim = 1)
    def batchify_obs(raw):
        batched = {}
        for k, v in raw.items():
            # If already numpy array, use it; otherwise convert
            if not isinstance(v, np.ndarray):
                try:
                    v = np.array(v)
                except Exception:
                    # fallback for weird types
                    v = np.asarray(v, dtype=np.float32)

            # Ensure array has at least 1 dimension for JAX sharding.
            # Scalars (0-d) -> convert to shape (1,)
            if v.ndim == 0:
                v = v.reshape((1,))
            else:
                # Add batch dim at axis=0
                v = v[np.newaxis, ...]

            # Standardize dtypes for Dreamer/JAX: booleans -> bool, rewards -> float32
            if k in ("is_first", "is_last", "is_terminal"):
                v = v.astype(bool)
            elif k == "reward":
                v = v.astype(np.float32)
            else:
                # keep numeric types as float32 where appropriate
                if np.issubdtype(v.dtype, np.floating):
                    v = v.astype(np.float32)
            batched[k] = v
        return batched

    obs = batchify_obs(raw_obs)

    # Initialize agent state (batch_size = 1)
    try:
        agent_state = agent.init_policy(batch_size=1)
    except TypeError:
        # some agents expect integer or tuple; try fallback
        agent_state = agent.init_policy(1)

    actions_list, rewards_list = [], []
    total_reward = 0.0

    print(f"\n[INFO] Starting Dreamer closed-loop rollout for {rollout_steps} steps...")

    for t in range(rollout_steps):
        try:
            # Call policy with batched obs
            result = agent.policy(agent_state, obs, mode='eval')

            if len(result) == 3:
                agent_state, act, _ = result
            elif len(result) == 2:
                agent_state, act = result
            else:
                raise ValueError(f"Unexpected policy return format: {len(result)} values")

            # Extract raw action array from whatever format agent returned
            if isinstance(act, dict):
                action = act.get('action', act)
            else:
                action = act

            # Remove batch dim if present and ensure 1-D action vector
            if hasattr(action, 'ndim') and action.ndim > 1:
                action = action[0]
            action = np.array(action, dtype=np.float32).flatten()
            actions_list.append(action)

            # Step environment with raw action (your env.step expects the array)
            # If your env.step previously expected a dict like {"action": ...}, adapt here.
            next_raw = env.step({"action": action, "reset": False})

            # env.step returns either (obs) or (obs, reward, done, info) in some wrappers.
            # Your Arm.step returns obs dict (with reward/is_last inside), so handle both.
            if isinstance(next_raw, tuple) and len(next_raw) in (3, 4):
                # classic (obs, reward, done, [info])
                next_obs_raw = next_raw[0]
                reward_val = next_raw[1]
                done = bool(next_raw[2])
            else:
                # env returns an obs dict with reward/is_last keys (your implementation)
                next_obs_raw = next_raw
                # Pull reward from obs dict (ensure numeric)
                reward_val = next_obs_raw.get("reward", 0.0)
                done = bool(next_obs_raw.get("is_last", False) or next_obs_raw.get("is_terminal", False))

            # Convert reward to float
            try:
                reward_val = float(np.array(reward_val).item())
            except Exception:
                reward_val = float(reward_val)

            rewards_list.append(reward_val)
            total_reward += reward_val

            if t % 10 == 0 or done:
                print(f"[Step {t:3d}] Reward: {reward_val:.3f} | Action: {np.round(action, 3)}")

            # Prepare obs for next policy call
            obs = batchify_obs(next_obs_raw)

            if done:
                print(f"[INFO] Episode finished early at step {t}")
                break

        except Exception as e:
            print(f"\n[ERROR] Rollout crashed at step {t}: {e}")
            import traceback; traceback.print_exc()
            break

    # Close env if it exposes close()
    try:
        env.close()
    except Exception:
        pass

    # Convert lists -> arrays
    actions = np.array(actions_list)
    rewards = np.array(rewards_list)

    print(f"\n[INFO] Closed-loop rollout complete.")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Steps: {len(actions)}")
    print(f"  Action shape: {actions.shape}")

    # Avoid min/max on empty arrays
    if rewards.size > 0:
        print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    else:
        print("  No rewards recorded (zero-length).")

    # Save results
    out_dir = Path("/home/xavie/dreamer/dreamerv3/data/rollouts")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "closed_loop_rollout.npz", actions=actions, rewards=rewards)
    print(f"[INFO] Saved rollout data to {out_dir}/closed_loop_rollout.npz")

    # Plot rewards if present
    if rewards.size > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(rewards, label="Reward", linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title("Dreamer Closed-Loop Rewards")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("[INFO] Skipping reward plot (no rewards).")

    return actions, rewards



def find_demo_dirs(data_dir):
    """Find all demo directories in the data directory."""
    demo_dirs = []
    data_path = Path(data_dir)
    
    # Check if data_dir itself is a demo
    if (data_path / "obs.npz").exists() and (data_path / "actions.npy").exists():
        return [str(data_path)]
    
    # Look for subdirectories with demo files
    for item in data_path.iterdir():
        if item.is_dir():
            if (item / "obs.npz").exists() and (item / "actions.npy").exists():
                demo_dirs.append(str(item))
    
    return sorted(demo_dirs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='/home/xavie/logdir/dreamer/20251109T171831')
    parser.add_argument('--data_dir', default='/home/xavie/dreamer/dreamerv3/data/demos/success',
                       help='Directory containing demo folders')
    parser.add_argument('--demo', default=None, 
                       help='Specific demo directory (overrides data_dir)')
    parser.add_argument('--steps', type=int, default=1000, 
                       help='Number of steps to run (0 for all)')
    parser.add_argument('--list', action='store_true',
                       help='List available demos and exit')
    args = parser.parse_args()
    
    # Find demos
    if args.demo:
        demo_dirs = [args.demo]
    else:
        demo_dirs = find_demo_dirs(args.data_dir)
    
    if args.list:
        print(f"\n[INFO] Found {len(demo_dirs)} demos in {args.data_dir}:")
        for i, demo_dir in enumerate(demo_dirs):
            print(f"  [{i}] {demo_dir}")
        return
    
    if not demo_dirs:
        print(f"[ERROR] No demos found in {args.data_dir}")
        print(f"Expected structure: {args.data_dir}/demo_N/obs.npz and actions.npy")
        return
    
    # Use first demo if not specified
    demo_path = demo_dirs[0]
    print(f"\n[INFO] Using demo: {demo_path}")
    if len(demo_dirs) > 1:
        print(f"[INFO] (Found {len(demo_dirs)} total demos, use --demo to specify)")
    
    # Load demo
    obs_data, gt_actions = load_demo(demo_path)
    
    # Create agent
    agent = create_agent(args.logdir)
    
    # Run inference
    num_steps = None if args.steps == 0 else args.steps
    pred_actions, rewards = run_closed_loop(agent, rollout_steps=args.steps or 300)
    
    if pred_actions is not None:
        # Compare with ground truth
        T = min(len(pred_actions), len(gt_actions))
        mse = np.mean((pred_actions[:T] - gt_actions[:T]) ** 2)
        mae = np.mean(np.abs(pred_actions[:T] - gt_actions[:T]))
        print(f"\n[INFO] Comparison with ground truth:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        
        # Per-dimension comparison
        print(f"\n[INFO] Per-dimension MAE:")
        for i in range(pred_actions.shape[1]):
            dim_mae = np.mean(np.abs(pred_actions[:T, i] - gt_actions[:T, i]))
            print(f"  Dim {i}: {dim_mae:.4f}")
        
        # Save predictions
        output_path = Path(demo_path) / "predicted_actions.npy"
        np.save(output_path, pred_actions)
        print(f"\n[INFO] Saved predictions to {output_path}")


if __name__ == "__main__":
    main()