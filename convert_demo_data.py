import numpy as np
import os
import uuid
import datetime
import elements
from embodied.envs.reward_function import compute_reward

def convert_data(obs_path, act_path, output_dir):
    print(f"Loading data from:\n  {obs_path}\n  {act_path}")
    
    # Load raw data
    with np.load(obs_path) as data:
        obs_data = {k: data[k] for k in data.files}
    actions = np.load(act_path)
    
    length = actions.shape[0]
    print(f"Trajectory length: {length}")
    
    # Prepare output dictionary
    episode = {}
    
    # 1. Process Observations
    for k, v in obs_data.items():
        # Slice arm_joints if needed (9 -> 6)
        if k == 'arm_joints' and v.shape[1] > 6:
            v = v[:, :6]
        episode[k] = v.astype(np.float32)
        
    # 2. Process Actions
    episode['action'] = actions.astype(np.float32)
    # episode['reset'] = np.zeros(length, dtype=bool) # Agent doesn't want this
    
    # 3. Compute Rewards and Generate StepIDs
    rewards = []
    stepids = []
    
    # Generate a full UUID for this chunk
    chunk_uuid = uuid.uuid4()
    chunk_uuid_bytes = chunk_uuid.bytes
    
    for t in range(length):
        # Construct single-step obs for reward function
        step_obs = {k: episode[k][t] for k in obs_data.keys()}
        r = compute_reward(step_obs)
        rewards.append(r)
        
        # Generate stepid: 16 bytes UUID + 4 bytes index (big endian)
        index_bytes = t.to_bytes(4, 'big')
        stepid_bytes = chunk_uuid_bytes + index_bytes
        stepids.append(np.frombuffer(stepid_bytes, dtype=np.uint8))
        
    episode['reward'] = np.array(rewards, dtype=np.float32)
    episode['stepid'] = np.array(stepids, dtype=np.uint8)
    
    # 4. Add Metadata
    episode['is_first'] = np.zeros(length, dtype=bool)
    episode['is_first'][0] = True
    
    episode['is_last'] = np.zeros(length, dtype=bool)
    episode['is_last'][-1] = True
    
    episode['is_terminal'] = np.zeros(length, dtype=bool)
    
    # episode['discount'] = np.ones(length, dtype=np.float32) # Agent doesn't want this
    
    # 5. Add Initial State (dyn/deter, dyn/stoch)
    # Standard DreamerV3 config: deter=512, stoch=(32, 32)
    # BUT size1m config uses classes=4, so stoch=(32, 4)
    episode['dyn/deter'] = np.zeros((length, 512), dtype=np.float32)
    episode['dyn/stoch'] = np.zeros((length, 32, 4), dtype=np.float32)

    
    # 5. Save as Chunk
    # Format: timestamp-uuid-succ-length.npz
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    
    # Use elements.UUID to generate compatible IDs
    # We need to convert our chunk_uuid (standard UUID) to elements.UUID
    # But elements.UUID(int) failed for large ints.
    # Safest bet: Just generate a new elements.UUID() for the filename
    # The stepids inside can still use the standard UUID bytes, that's fine.
    
    unique_id = str(elements.UUID())
    succ_id = str(elements.UUID())
    
    filename = f"{timestamp}-{unique_id}-{succ_id}-{length}.npz"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Save compressed
    np.savez_compressed(output_path, **episode)
    print(f"Saved converted episode to: {output_path}")
    
    # Verify load
    with np.load(output_path) as f:
        print(f"Verification - Keys: {list(f.keys())}")

if __name__ == "__main__":
    # Base directory containing all demo folders
    base_dir = os.path.expanduser("~/dreamer/dreamerv3/log_data/replay")
    output_dir = os.path.expanduser("~/dreamer/dreamerv3/log_data/online_training/replay")
    
    print(f"Scanning for demos in: {base_dir}")
    
    # Iterate over all items in base_dir
    count = 0
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Check if it's a directory and looks like a demo
        if os.path.isdir(item_path) and item.startswith("demo_rosbag2"):
            print(f"\nProcessing: {item}")
            
            obs_path = os.path.join(item_path, "obs.npz")
            act_path = os.path.join(item_path, "actions.npy")
            
            if os.path.exists(obs_path) and os.path.exists(act_path):
                try:
                    convert_data(obs_path, act_path, output_dir)
                    count += 1
                except Exception as e:
                    print(f"FAILED to convert {item}: {e}")
            else:
                print(f"Skipping {item}: Missing obs.npz or actions.npy")
                
    print(f"\nDone! Converted {count} trajectories.")
