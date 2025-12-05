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
from embodied.envs import real_arm, eval_real_arm
import argparse
import os
import matplotlib.pyplot as plt
import jax

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

# --- Reusing classes from train_online_simple.py ---
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

class SimplifyAction(embodied.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._act_space = {
            'action': elements.Space(np.float32, (3,)),
            'reset': env.act_space['reset']
        }

    @property
    def act_space(self):
        return self._act_space

    def step(self, action):
        if 'action' in action:
            action = action.copy()
            act = action['action']
            padded = np.concatenate([act, np.zeros(2, dtype=np.float32)], axis=0)
            action['action'] = padded
        return self.env.step(action)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='Path to log directory containing checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint name (e.g. checkpoint.ckpt)')
    args_cli = parser.parse_args()

    logdir = pathlib.Path(os.path.expanduser(args_cli.logdir))
    
    # Load config
    config_path = logdir / 'config.yaml'
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        return

    configs = yaml.YAML(typ='safe').load(config_path.read_text())
    config = elements.Config(configs)

    # Force report settings
    config = config.update({
        'jax.platform': 'cpu', # Use CPU for visualization to avoid OOM/conflicts
        'batch_size': 1,       # Single batch for visualization
        'batch_length': 64,
        'report_length': 64,
    })

    # Recreate Environment (needed for spaces)
    # We use a dummy target since we are just replaying data
    def make_env():
        env = real_arm.RealArm(task='online_reach', hz=10.0, target_pos=[0,0,0])
        env = embodied.wrappers.TimeLimit(env, 200)
        keys_to_remove = [
            'gripper_state', 'left_contact', 'right_contact',
            'block_pose', 'wrist_angle'
        ]
        env = FilterObs(env, keys_to_remove)
        env = SimplifyAction(env)
        return env

    env = make_env()
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()

    # Create Agent
    print("Creating agent...")
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
    agent = agent_module.Agent(obs_space, act_space, agent_config)

    # Load Checkpoint
    print(f"Loading checkpoint...")
    if args_cli.checkpoint:
        # Load specific file
        path = pathlib.Path(args_cli.checkpoint)
        if not path.exists():
             # Try relative to logdir
             path = logdir / args_cli.checkpoint
        
        if not path.exists():
            print(f"Checkpoint file not found: {path}")
            return
            
        print(f"Loading specific checkpoint: {path}")
        # elements.checkpoint.load expects a dict of loaders
        # The checkpoint file (e.g. agent.pkl) contains the data directly if it's a component checkpoint
        # OR it might be a full checkpoint. 
        # Based on file listing 'agent.pkl', it seems to be component-wise.
        # Let's try loading it directly into the agent.
        elements.checkpoint.load(str(path), {'agent': agent.load})
    else:
        # Load latest from ckpt directory
        ckpt_dir = logdir / 'ckpt'
        if not ckpt_dir.exists():
             print(f"Checkpoint directory not found: {ckpt_dir}")
             return
        
        cp = elements.Checkpoint(ckpt_dir)
        cp.agent = agent
        cp.load()

    # Load Replay Buffer
    print("Loading replay buffer...")
    replay_dir = logdir / 'replay_train'
    replay = embodied.replay.Replay(
        length=config.batch_length + config.replay_context,
        capacity=config.replay.size,
        directory=replay_dir,
        online=False,
        chunksize=config.replay.chunksize,
    )
    print(f"Replay directory: {replay_dir}")
    replay.load()
    print(f"Loaded {len(replay)} items from replay buffer.")
    
    # Sample Batch
    print("Sampling batch...")
    # We need a stream to properly format the batch
    def make_stream(replay):
        fn = functools.partial(replay.sample, config.batch_size, 'train')
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream,
            length=config.batch_length,
            consec=config.consec_train,
            prefix=config.replay_context,
            strict=True,
            contiguous=True
        )
        return stream

    stream = make_stream(replay)
    stream = agent.stream(stream)
    iterator = iter(stream)
    batch = next(iterator)

    # Run Report
    print("Running agent report...")
    # Initialize agent state
    carry = agent.init_report(config.batch_size)
    
    # Run report
    # We need to wrap this in jax.jit or call it carefully if it expects JAX arrays
    # The agent.report method handles JAX internals, we just pass the batch
    # But batch comes from replay as numpy, agent expects numpy or jax?
    # embodied agents usually handle conversion. Let's try passing directly.
    
    # The report method signature: report(self, carry, data)
    # It returns (carry, metrics)
    _, metrics = agent.report(carry, batch)

    # Extract and Plot
    print("Generating plots...")
    os.makedirs('vis_plots', exist_ok=True)

    # Keys to plot
    keys_to_plot = ['actual_pose']

    for key in keys_to_plot:
        true_key = f'openloop/{key}_true'
        pred_key = f'openloop/{key}_pred'

        if true_key not in metrics or pred_key not in metrics:
            print(f"Key {key} not found in metrics. Available keys: {list(metrics.keys())}")
            continue

        # Convert to numpy
        true_data = np.array(metrics[true_key])
        pred_data = np.array(metrics[pred_key])

        # Shape: [Batch, Time, Dims]
        # We used batch_size=1, so take first element
        true_seq = true_data[0]
        pred_seq = pred_data[0]

        # Plot each dimension
        dims = true_seq.shape[-1]
        fig, axes = plt.subplots(dims, 1, figsize=(10, 2 * dims), sharex=True)
        if dims == 1:
            axes = [axes]

        time_steps = np.arange(true_seq.shape[0])

        # Split point for context vs imagination (usually half of report_length)
        # In agent.py: firsthalf = T // 2. 
        # The report logic concatenates observed (first half) and imagined (second half)
        # But wait, agent.py logic:
        # pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
        # obsrecons is on first half (context), imgrecons is on second half (imagination)
        context_len = true_seq.shape[0] // 2

        for i in range(dims):
            ax = axes[i]
            ax.plot(time_steps, true_seq[:, i], 'k-', label='Ground Truth', linewidth=2)
            ax.plot(time_steps, pred_seq[:, i], 'r--', label='Model Prediction', linewidth=2)
            
            # Mark context boundary
            ax.axvline(x=context_len, color='b', linestyle=':', label='Imagination Start')
            
            ax.set_ylabel(f'Dim {i}')
            if i == 0:
                ax.legend()

        plt.xlabel('Time Step')
        plt.suptitle(f'World Model Prediction: {key}')
        plt.tight_layout()
        
        save_path = f'vis_plots/{key}.png'
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()

if __name__ == '__main__':
    main()
