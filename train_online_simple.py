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
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
import argparse
import os


class FilterObs(embodied.Wrapper):
    def __init__(self, env, keys_to_remove):
        super().__init__(env)
        self._keys_to_remove = keys_to_remove
        self._obs_space = {k: v for k, v in env.obs_space.items()
                           if k not in keys_to_remove}
                           
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
            act = action['action']

            # [dx, dy, dz] -> [dx, dy, dz, 0.0, 0.0]
            padded = np.concatenate([act, np.zeros(2, dtype=np.float32)], axis=0)
            action = action.copy()
            action['action'] = padded
        return self.env.step(action)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_checkpoint', type=str, default=None, help='Path to checkpoint to load')

    args_cli = parser.parse_args()
    if args_cli.from_checkpoint:
        args_cli.from_checkpoint = os.path.expanduser(args_cli.from_checkpoint)

    # Load configs
    root = pathlib.Path(dreamerv3.__file__).parent
    configs = elements.Path(root / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs)
    config = elements.Config(configs['defaults'])
    config = config.update(configs['size1m'])

    updates = {
        # General Overrides
        'logdir': '~/dreamer/dreamerv3/log_data/online_training_simple',
        'batch_size': 16,
        
        # JAX Overrides (from size1m/debug, but simplified for clean look)
        'jax.prealloc': False,
        'jax.platform': 'cuda',
        
        # Run Overrides (from your desired settings)
        'run.train_ratio': 8, # Set back to 8 for simple environment/CPU use
        'run.log_every': 60,
        'run.envs': 1,
        'run.eval_envs': 0,
        
        # Agent Overrides (Entropy/Exploration)
        # CRITICAL FIX: Use the actual key for action entropy coefficient
        # This replaces the non-existent 'agent.entropy_coeff'
        'agent.imag_loss.actent': 0.005, 
    }

    if args_cli.from_checkpoint:

        updates['run.from_checkpoint'] = args_cli.from_checkpoint

    config = config.update(updates)
    logdir = elements.Path(config.logdir)
    logdir.mkdir()
    print('Logdir:', logdir)
    config.save(logdir / 'config.yaml')

    def make_env(config, index, **overrides):

        env = real_arm.RealArm(task='online_reach', hz=10.0)
        env = embodied.wrappers.TimeLimit(env, 1000)

        keys_to_remove = ['gripper_state', 'left_contact', 'right_contact', 'block_pose', 'wrist_angle']
        env = FilterObs(env, keys_to_remove)

        env = SimplifyAction(env)
        return env

    def make_agent(config):

        env = make_env(config, 0)
        # Filter spaces
        notlog = lambda k: not k.startswith('log/')
        obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
        act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
        env.close()
   
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

        return agent

    def make_replay(config, folder, mode='train'):
        # Simplified replay creation
        directory = elements.Path(config.logdir) / folder

        replay = embodied.replay.Replay(
            length=config.batch_length + config.replay_context,
            capacity=int(1e5),
            directory=directory,
            online=True,
            chunksize=1024,
        )

        return replay

    def make_logger(config):

        step = elements.Counter()
        logdir = config.logdir

        outputs = [
            elements.logger.TerminalOutput(config.logger.filter, 'Agent'),
            elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
            elements.logger.TensorBoardOutput(logdir, config.logger.fps),
        ]
        return elements.Logger(step, outputs, multiplier=1)

    def make_stream(config, replay, mode):
        length = config.batch_length if mode == 'train' else config.report_length
        consec = config.consec_train if mode == 'train' else config.consec_report
        fn = functools.partial(replay.sample, config.batch_size, mode)
        stream = embodied.streams.Stateless(fn)

        stream = embodied.streams.Consec(
            stream,
            length=length,
            consec=consec,
            prefix=config.replay_context,
            strict=(mode == 'train'),
            contiguous=True
        )
        return stream

    args = elements.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * 16,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        consec_report=config.consec_report,
        from_checkpoint_regex='.*',
    )

    embodied.run.train(
        functools.partial(make_agent, config),
        functools.partial(make_replay, config, 'replay'),
        functools.partial(make_env, config),
        functools.partial(make_stream, config),
        functools.partial(make_logger, config),
        args
    )

if __name__ == '__main__':

    main()