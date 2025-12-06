import warnings
import functools
import pathlib
import sys
import ruamel.yaml as yaml
import elements
import embodied
import numpy as np
from dreamerv3 import agent as agent_module
import dreamerv3
from embodied.envs import real_arm
import argparse
import matplotlib.pyplot as plt
import csv

import os

# ... (Previous imports and wrappers remain same) ...

class FilterObs(embodied.Wrapper):
    def __init__(self, env, keys_to_remove):
        super().__init__(env)
        self._keys_to_remove = keys_to_remove
        self._obs_space = {k: v for k, v in env.obs_space.items() if k not in keys_to_remove}
                           
    @property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        obs = self.env.step(action)
        for k in self._keys_to_remove:
            if k in obs:
                del obs[k]
        return obs

    def reset(self):
        obs = self.env.reset()
        for k in self._keys_to_remove:
            if k in obs:
                del obs[k]
        return obs

class SimplifyAction(embodied.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        # Original: 5 (dx, dy, dz, dwrist, dgrip)
        # New: 3 (dx, dy, dz)
        self._act_space = {
            'action': elements.Space(np.float32, (3,)),
            'reset': env.act_space['reset']
        }

    @property
    def act_space(self):
        return self._act_space

    def step(self, action):
        # Pad action with 0.0 for wrist and gripper
        if 'action' in action:
            action = action.copy()  # Copy first to avoid modifying original
            act = action['action']

            # [dx, dy, dz] -> [dx, dy, dz, 0.0, 0.0]
            padded = np.concatenate([act, np.zeros(2, dtype=np.float32)], axis=0)
            action['action'] = padded
        return self.env.step(action)

class RandomAgent:
    def __init__(self, act_space):
        self.act_space = act_space

    def init_policy(self, batch_size):
        return None

    def policy(self, carry, obs, mode='eval'):
        batch_size = len(next(iter(obs.values())))
        action = {}
        for k, v in self.act_space.items():
            if np.issubdtype(v.dtype, np.floating):
                # Assume normalized action space [-1, 1] for random agent
                act = np.random.uniform(-1.0, 1.0, (batch_size,) + v.shape).astype(v.dtype)
            else:
                # Fallback for non-float (e.g. reset bool)
                # We generally don't want to trigger reset randomly
                if v.dtype == bool:
                     act = np.zeros((batch_size,) + v.shape, dtype=bool)
                else:
                     act = np.random.uniform(v.low, v.high, (batch_size,) + v.shape).astype(v.dtype)
            action[k] = act
        return carry, action, {}

def run_episode(env, agent, mode='eval'):
    obs = env.reset()
    agent_state = agent.init_policy(batch_size=1)
    
    done = False
    step = 0
    total_reward = 0
    
    while not done:
        # Add batch dimension
        obs_batch = {k: np.array([v]) for k, v in obs.items()}
        
        # Policy step
        agent_state, action_batch, _ = agent.policy(agent_state, obs_batch, mode=mode)
        
        # Remove batch dimension
        action = {k: v[0] for k, v in action_batch.items()}
        action['reset'] = False
        
        # Env step
        obs = env.step(action)
        
        reward = obs['reward']
        total_reward += reward
        
        if obs['is_last']:
            done = True
        
        step += 1
        
    return total_reward, step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to load')
    parser.add_argument('--target', type=float, nargs=3, default=None, help='Override target position [x, y, z]')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run per policy')
    parser.add_argument('--length', type=int, default=200, help='Episode length')
    
    args_cli = parser.parse_args()
    checkpoint_path = os.path.expanduser(args_cli.checkpoint)

    print(f"Loading checkpoint from: {checkpoint_path}")

    # 1. Load and Setup Config
    root = pathlib.Path(dreamerv3.__file__).parent
    configs = elements.Path(root / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs)
    config = elements.Config(configs['defaults'])
    config = config.update(configs['size1m'])

    # Apply same updates as training
    updates = {
        'batch_size': 32,
        'batch_length': 32,
        'report_length': 16,
        'jax.prealloc': False,
        'jax.platform': 'cuda', 
        'agent.dyn.rssm.deter': 512,
        'agent.dyn.rssm.hidden': 512,
        'agent.dyn.rssm.stoch': 32,
        'agent.dyn.rssm.classes': 32,
        'agent.imag_length': 15,
        'agent.policy.minstd': 0.1,
        'replay.online': True,
        'replay.size': 500000,
        'replay.chunksize': 1024,
    }
    config = config.update(updates)
    
    # 2. Create Environment
    print("Initializing Environment...")
    # We use RealArm directly
    env = real_arm.RealArm(task='online_reach', hz=10.0, target_pos=args_cli.target)
    env = embodied.wrappers.TimeLimit(env, args_cli.length)
    
    keys_to_remove = [
        'gripper_state', 'left_contact', 'right_contact',
        'block_pose', 'wrist_angle'
    ]
    env = FilterObs(env, keys_to_remove)
    env = SimplifyAction(env)

    # 3. Create Agents
    print("Initializing Agents...")
    
    # Trained Agent
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}

    agent_config = elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )
    trained_agent = agent_module.Agent(obs_space, act_space, agent_config)
    
    # Load Weights
    print(f"Loading weights from {checkpoint_path}...")
    elements.checkpoint.load(checkpoint_path, {'agent': trained_agent.load})
    
    # Random Agent
    random_agent = RandomAgent(act_space)

    # 4. Run Evaluation Loop
    results = {'Trained': [], 'Random': []}
    
    for agent_name, agent in [('Trained', trained_agent), ('Random', random_agent)]:
        print(f"\nEvaluating {agent_name} Policy...")
        rewards = []
        durations = []
        
        for i in range(args_cli.episodes):
            print(f"  Episode {i+1}/{args_cli.episodes}...", end='\r')
            total_reward, duration = run_episode(env, agent)
            rewards.append(total_reward)
            durations.append(duration)
            print(f"  Episode {i+1}/{args_cli.episodes}: Reward={total_reward:.2f}, Steps={duration}")
            
        results[agent_name] = {'rewards': rewards, 'durations': durations}

    env.close()
    
    # 5. Report and Plot
    print("\n=== Evaluation Report ===")
    
    # CSV Logging
    csv_filename = 'eval_results.csv'
    print(f"Logging results to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Agent', 'Episode', 'Reward', 'Duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for name, data in results.items():
            avg_reward = np.mean(data['rewards'])
            cum_reward = np.sum(data['rewards'])
            avg_duration = np.mean(data['durations'])
            print(f"{name}: Avg Reward={avg_reward:.2f}, Cumulative={cum_reward:.2f}, Avg Duration={avg_duration:.1f}")
            
            for i in range(len(data['rewards'])):
                writer.writerow({
                    'Agent': name,
                    'Episode': i + 1,
                    'Reward': data['rewards'][i],
                    'Duration': data['durations'][i]
                })

    # Plotting
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Reward Plot
    ax = axes[0]
    for name, data in results.items():
        ax.plot(data['rewards'], label=name, marker='o')
    ax.set_title('Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True)
    
    # Duration Plot
    ax = axes[1]
    for name, data in results.items():
        ax.plot(data['durations'], label=name, marker='x')
    ax.set_title('Episode Durations')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('eval_results.png')
    print("Saved plots to eval_results.png")
    print("Done.")

if __name__ == '__main__':
    main()
