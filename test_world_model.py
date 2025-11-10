import os
import numpy as np
import yaml
import elements
import embodied
from dreamerv3 import agent as dreamer


def make_action_space(actions):
    """Create an action space compatible with any embodied version."""
    act_shape = actions.shape[1:] if len(actions.shape) > 1 else (1,)

    # Try all common import paths
    if hasattr(embodied, "Space"):
        SpaceClass = embodied.Space
    elif hasattr(embodied, "core") and hasattr(embodied.core, "Space"):
        SpaceClass = embodied.core.Space
    elif hasattr(embodied, "envs") and hasattr(embodied.envs, "Space"):
        SpaceClass = embodied.envs.Space
    else:
        raise ImportError("Could not find embodied.Space definition in your version.")

    return SpaceClass(np.float32, act_shape, low=-1, high=1)


def test_world_model(checkpoint_dir, demo_obs_path, demo_actions_path):
    # --- Load Dreamer config ---
    config_path = os.path.join(os.path.dirname(checkpoint_dir), "config.yaml")
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    config = elements.Config(raw_config)

    # --- Load demo data ---
    obs = np.load(demo_obs_path)
    actions = np.load(demo_actions_path)
    obs_dict = {k: v for k, v in obs.items()}
    print(f"Loaded {len(actions)} actions and {len(obs_dict)} observation tensors")

    # --- Define action space dynamically ---
    act_space = make_action_space(actions)

    # --- Initialize Dreamer agent ---
    agent = dreamer.Agent(act_space, config)
    step = embodied.Counter()

    # --- Load checkpoint ---
    checkpoint = elements.Checkpoint(checkpoint_dir)
    checkpoint.agent = agent
    checkpoint.load_or_save()
    print(f"✅ Loaded checkpoint from {checkpoint_dir}")

    # --- Initialize model state ---
    carry = agent.init_train(batch_size=1)

    total_loss = 0.0
    count = 0

    # --- Evaluate world model predictions ---
    for t in range(len(actions) - 1):
        batch = {
            "obs": {k: v[t:t+1] for k, v in obs_dict.items()},
            "action": actions[t:t+1],
            "is_terminal": np.zeros((1,), dtype=bool),
        }
        carry, outs, mets = agent.train(carry, batch)
        if "train/loss/dyn" in mets:
            total_loss += mets["train/loss/dyn"]
            count += 1

    avg_loss = total_loss / max(count, 1)
    print(f"📊 Average dynamics prediction loss over demo: {avg_loss:.4f}")


if __name__ == "__main__":
    test_world_model(
        checkpoint_dir="/home/xavie/logdir/dreamer/20251109T171831/ckpt",
        demo_obs_path="/home/xavie/dreamer/dreamerv3/data/demos/success/demo_rosbag2_2025_11_09-15_33_27/obs.npz",
        demo_actions_path="/home/xavie/dreamer/dreamerv3/data/demos/success/demo_rosbag2_2025_11_09-15_33_27/actions.npy",
    )
