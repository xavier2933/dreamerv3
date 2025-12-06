import argparse
import functools
import os
import pathlib
import sys
import ruamel.yaml as yaml
import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import elements
import embodied
import dreamerv3
from dreamerv3 import agent as agent_module
from embodied.envs import real_arm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_checkpoint', type=str, required=True, help='Path to checkpoint to load')
    parser.add_argument('--logdir', type=str, required=True, help='Path to logdir (where replay is)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for visualization')
    parser.add_argument('--save_path', type=str, default='predictions.png', help='Path to save the plot')
    args_cli = parser.parse_args()

    args_cli.from_checkpoint = os.path.expanduser(args_cli.from_checkpoint)
    args_cli.logdir = os.path.expanduser(args_cli.logdir)

    # Load configs (copied from train_online_simple.py logic)
    root = pathlib.Path(dreamerv3.__file__).parent
    configs = elements.Path(root / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs)
    config = elements.Config(configs['defaults'])
    config = config.update(configs['size1m'])

    # Apply same updates as train_online_simple.py to ensure model matches
    updates = {
        'logdir': args_cli.logdir,
        'batch_size': args_cli.batch_size,
        'batch_length': 32, # Match training length
        'jax.prealloc': False,
        'jax.platform': 'cpu', # Use CPU for inference/plotting to avoid OOM if training is running
        'run.train_ratio': 2,
        'run.log_every': 60,
        'run.save_every': 500,
        'run.envs': 1,
        'run.eval_envs': 1,
        'run.report_every': 1000,
        'agent.opt.lr': 1e-4,
        'agent.opt.eps': 1e-6,
        'agent.opt.agc': 0.3,
        'agent.opt.warmup': 2000,
        'agent.dyn.rssm.deter': 512,
        'agent.dyn.rssm.hidden': 512,
        'agent.dyn.rssm.stoch': 32,
        'agent.dyn.rssm.classes': 32,
        'agent.imag_length': 15,
        'agent.policy.minstd': 0.1,
        'agent.policy.maxstd': 1.0,
        'agent.policy.unimix': 0.1,
        'agent.loss_scales.policy': 1.0,
        'agent.loss_scales.value': 1.0,
        'agent.loss_scales.rep': 1.0,
        'agent.loss_scales.repval': 0.5,
        'agent.imag_loss.actent': 0.2,
        'agent.imag_loss.lam': 0.95,
        'agent.slowvalue.rate': 0.005,
        'agent.retnorm.impl': 'perc',
        'replay.online': True,
        'replay.size': 500000,
        'replay.chunksize': 128,
        'run.from_checkpoint': args_cli.from_checkpoint,
    }
    config = config.update(updates)
   
    # Define spaces manually since we don't want to spin up a real env
    # These must match RealArm spaces
    obs_space = {
        'arm_joints': elements.Space(np.float32, (6,)),
        'block_pose': elements.Space(np.float32, (7,)),
        'actual_pose': elements.Space(np.float32, (7,)),
        'wrist_angle': elements.Space(np.float32, (1,)),
        'target_error': elements.Space(np.float32, (3,)),
        'gripper_state': elements.Space(np.float32, (1,)),
        'left_contact': elements.Space(np.float32, (1,)),
        'right_contact': elements.Space(np.float32, (1,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
   
    # Filter keys as done in train_online_simple.py
    keys_to_remove = [
        'gripper_state', 'left_contact', 'right_contact',
        'block_pose', 'wrist_angle'
    ]
    obs_space = {k: v for k, v in obs_space.items() if k not in keys_to_remove}
   
    act_space = {
        'action': elements.Space(np.float32, (3,)), # Simplified action space
        # 'reset': elements.Space(bool), # Agent usually doesn't see reset in act space for policy
    }

    print("Initializing Agent...")
    # Create agent config
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
   
    # Manually disable transfer guard as Agent setup enables it
    jax.config.update("jax_transfer_guard", None)
   
    # Resolve checkpoint path if it points to a directory with 'latest'
    if os.path.isdir(args_cli.from_checkpoint):
        latest_path = os.path.join(args_cli.from_checkpoint, 'latest')
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                latest_dir = f.read().strip()
            print(f"Found 'latest' file pointing to: {latest_dir}")
            args_cli.from_checkpoint = os.path.join(args_cli.from_checkpoint, latest_dir)
   
    # Load checkpoint
    print(f"Loading checkpoint from {args_cli.from_checkpoint}...")
    elements.checkpoint.load(args_cli.from_checkpoint, dict(agent=agent.load))

    # Load Replay
    print(f"Loading Replay from {config.logdir}...")
    directory = elements.Path(config.logdir) / 'replay_train'
    replay = embodied.replay.Replay(
        length=config.batch_length + config.replay_context,
        capacity=int(config.replay.size),
        directory=directory,
        online=config.replay.online,
        chunksize=config.replay.chunksize,
    )
   
    # Load Replay data
    print("Loading replay data from disk...")
    replay.load()
    print(f"Replay loaded with {len(replay)} items.")
   
    # Create stream
    print("Creating stream...")
    # Use 'report' mode to get sequential data if possible, or just sample
    # We want a batch
    batch_size = args_cli.batch_size
   
    print("Sampling batch...")
    # replay.sample returns a dict of numpy arrays [B, T, ...]
    batch = replay.sample(batch_size)
   
    # Run Prediction (Open Loop)
    print("Running prediction...")
   
    # We need to use ninjax to call the model with parameters
    import ninjax as nj
   
    def predict_open_loop(carry, data):
        # Use agent.model to handle context and logic
        # agent.model._apply_replay_context expects data to have 'is_first' etc.
        # It returns (carry, obs, prevact, stepid)
        carry, obs, prevact, _ = agent.model._apply_replay_context(carry, data)
       
        (enc_carry, dyn_carry, dec_carry) = carry
       
        # We need to split into context and target
        # agent.report logic:
        B, T = obs['is_first'].shape
        # We assume batch_size is small, so we use all of it
        RB = B
       
        # Helper to split
        # We want to predict the second half
        context_len = T // 2
       
        firsthalf = lambda xs: jax.tree.map(lambda x: x[:, :context_len], xs)
        secondhalf = lambda xs: jax.tree.map(lambda x: x[:, context_len:], xs)
       
        # 1. Observe Context
        # We need to run encoder first?
        # In Agent.report:
        # _, (new_carry, entries, outs, mets) = self.loss(..., training=False)
        # But that runs loss. We want open loop.
        # Agent.report open loop section:
        # dyn_carry, _, obsfeat = self.dyn.observe(dyn_carry, firsthalf(outs['tokens']), ...)
        # It uses 'outs['tokens']' which comes from self.enc called in self.loss.
       
        # So we must run encoder on context first.
        # enc returns (carry, entries, tokens)
        # We need to handle the carry carefully.
       
        # Let's just run observe on the first half using the encoder
        # agent.model.enc(enc_carry, obs, reset, training=False)
       
        # Context (First Half)
        obs_ctx = firsthalf(obs)
        reset_ctx = firsthalf(obs['is_first'])
        prevact_ctx = firsthalf(prevact)
       
        # Encode context
        # enc: (carry, obs, reset, training) -> (carry, entry, tokens)
        _, _, tokens_ctx = agent.model.enc(enc_carry, obs_ctx, reset_ctx, training=False)
       
        # Observe context
        # dyn.observe: (carry, tokens, prevact, reset, training) -> (carry, entry, feat)
        dyn_carry_post, _, obsfeat = agent.model.dyn.observe(
            dyn_carry, tokens_ctx, prevact_ctx, reset_ctx, training=False)
           
        # 2. Imagine Target (Second Half)
        # We use the actions from the second half
        prevact_tgt = secondhalf(prevact)
       
        # dyn.imagine: (carry, prevact, length, training) -> (carry, feat, prevact)
        # Note: imagine starts from the *current* state of dyn_carry_post
        _, imgfeat, _ = agent.model.dyn.imagine(
            dyn_carry_post, prevact_tgt, length=T - context_len, training=False)
           
        # 3. Decode
        # We decode both for plotting
       
        # Decode Context (Reconstruction)
        _, _, obsrecons = agent.model.dec(
            dec_carry, obsfeat, reset_ctx, training=False)
           
        # Decode Target (Prediction)
        # For decoding imagination, we usually pass zeros for reset
        reset_tgt = jnp.zeros_like(secondhalf(obs['is_first']))
        _, _, imgrecons = agent.model.dec(
            dec_carry, imgfeat, reset_tgt, training=False)
           
        return obs, obsrecons, imgrecons

    # Wrap with nj.pure
    predict_pure = nj.pure(predict_open_loop)
   
    # Prepare inputs
    # Convert batch to JAX arrays
    batch_jax = jax.tree.map(jnp.array, batch)
   
    # Add missing keys required by _apply_replay_context
    # 'consec': [B, T] int32. If we treat this as a single chunk, we can set it to 0?
    # Actually, _apply_replay_context uses it to check for first chunk.
    # If we set it to all 0s, it might think it's the first chunk.
    # Let's set it to zeros.
    B, T = batch_jax['is_first'].shape
    batch_jax['consec'] = jnp.zeros((B, T), dtype=jnp.int32)
   
    # 'stepid': [B, T, ...] unique ID.
    # We can just generate dummy IDs.
    # agent.py expects stepid to be present.
    # In replay.py, stepid is uint8.
    batch_jax['stepid'] = jnp.zeros((B, T, 16), dtype=jnp.uint8) # Dummy UUIDs
   
    # Get params
    params = agent.params
    rng = jax.random.PRNGKey(0)
   
    # Initialize carry
    carry = agent.init_report(batch_size)
   
    # Call function
    print("Executing JAX function...")
    # pure returns (new_params, result)
    # Pass seed as keyword argument!
    _, (obs_all, obsrecons, imgrecons) = predict_pure(params, carry, batch_jax, seed=rng)
   
    # Process results for plotting
    keys_to_plot = ['actual_pose', 'arm_joints']
    results = {}
   
    # We need to combine reconstruction and imagination
    # obsrecons is dict of distributions/modes
    # imgrecons is dict of distributions/modes
   
    for key in keys_to_plot:
        if key not in obs_space:
            continue
           
        # Ground Truth (Full sequence)
        # obs_all is the full observation dict processed by _apply_replay_context
        truth = obs_all[key][0] # [T, D]
       
        # Prediction
        # Recon: [B, T_ctx, D]
        # Imag: [B, T_tgt, D]
        recon = obsrecons[key].pred()
        imag = imgrecons[key].pred()
       
        pred = jnp.concatenate([recon, imag], axis=1)
        pred = np.array(pred[0]) # [T, D]
        truth = np.array(truth)
       
        results[key] = {'truth': truth, 'pred': pred}
       
    context_len = config.batch_length // 2
       
    # Plotting
    print(f"Plotting to {args_cli.save_path}...")
    num_keys = len(results)
    if num_keys == 0:
        print("No keys to plot found!")
        return

    fig, axes = plt.subplots(num_keys, 1, figsize=(10, 5 * num_keys), squeeze=False)
   
    for i, (key, data) in enumerate(results.items()):
        ax = axes[i, 0]
        truth = data['truth']
        pred = data['pred']
       
        # Define labels
        if key == 'actual_pose':
            labels = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
        elif key == 'arm_joints':
            labels = [f'joint{j+1}' for j in range(dims)]
        else:
            labels = [f'dim{d}' for d in range(dims)]

        # Plot each dimension
        dims = truth.shape[-1]
        for d in range(dims):
            color = plt.cm.tab10(d % 10)
            label = labels[d] if d < len(labels) else f'dim{d}'
            
            ax.plot(truth[:, d], label=label, linestyle='-', color=color, alpha=0.7)
            ax.plot(pred[:, d], label=None, linestyle='--', color=color, alpha=1.0)
           
        # Draw vertical line at context split
        ax.axvline(x=context_len, color='k', linestyle=':', label='Context/Imag Split')
       
        ax.set_title(f'{key} (Solid: Truth, Dashed: Pred)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
       
    plt.tight_layout()
    plt.savefig(args_cli.save_path)
    print("Done.")

if __name__ == '__main__':
    main()
