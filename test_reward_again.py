import numpy as np

def compute_reward(obs):
    """
    FIXED reward function for block pickup task.
    
    Key fixes:
    1. Reduced distance penalty to not dominate
    2. Increased success rewards to make them attractive
    3. Added height-based shaping reward
    4. Made gripper closing rewarding when near block
    """
    reward = 0.0

    # === Extract positions ===
    target_pos = None
    block_pos = None

    if 'target_pose' in obs:
        target_pose = np.atleast_1d(obs['target_pose'])
        if len(target_pose) >= 3:
            target_pos = target_pose[:3]

    if 'block_pose' in obs:
        block_pose = np.atleast_1d(obs['block_pose'])
        if len(block_pose) >= 3:
            block_pos = block_pose[:3]

    # === Extract contact and gripper state ===
    left_contact = float(np.atleast_1d(obs.get('left_contact', [0.0]))[0])
    right_contact = float(np.atleast_1d(obs.get('right_contact', [0.0]))[0])
    gripper_state = float(np.atleast_1d(obs.get('gripper_state', [1.0]))[0])

    # === Compute holding state ===
    has_contact = (left_contact > 0.5 and right_contact > 0.5)
    gripper_closed = (gripper_state < 0.5)
    holding_block = has_contact and gripper_closed

    # === Distance-based reward (only when not holding) ===
    if target_pos is not None and block_pos is not None:
        distance = np.linalg.norm(target_pos - block_pos)
        
        if not holding_block:
            # Smaller penalty, shaped to encourage approach
            # Cap at 1.0m distance to avoid huge penalties
            capped_distance = min(distance, 1.0)
            reward += -0.5 * capped_distance
        else:
            # Small bonus for maintaining good grip
            reward += 0.5

    # === Contact rewards ===
    if has_contact:
        reward += 2.0  # Reward making contact
        
        # Extra reward if gripper is closed while in contact
        if gripper_closed:
            reward += 3.0  # Total +5.0 for grasping

    # === Height-based rewards (the main goal) ===
    if block_pos is not None and holding_block:
        block_height = block_pos[2]
        
        # Progressive height rewards
        if block_height > 0.05:  # Lifted off table
            reward += 5.0
        
        if block_height > 0.15:  # Partially lifted
            reward += 5.0
        
        if block_height > 0.25:  # Success!
            reward += 20.0  # Big reward for success
            
        # Additional shaping: reward any height gain
        # (assumes table is at z=0)
        reward += 10.0 * max(0, block_height)

    return float(reward)


class BlockPickupReward:
    """Reward function wrapper for block pickup task."""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def compute_reward(self, obs, prev_obs=None):
        """Use the standalone compute_reward function."""
        return compute_reward(obs)
    
    def compute_trajectory_rewards(self, obs_dict):
        """Compute rewards for entire trajectory."""
        first_key = list(obs_dict.keys())[0]
        T = len(obs_dict[first_key])
        
        rewards = np.zeros(T, dtype=np.float32)
        
        for t in range(T):
            obs_t = {k: v[t] for k, v in obs_dict.items()}
            rewards[t] = compute_reward(obs_t)
        
        return rewards


# Test to show reward progression
if __name__ == "__main__":
    print("Testing reward function on pickup sequence:")
    print("-" * 60)
    
    # Simulate a successful pickup sequence
    scenarios = [
        ("Start: arm far from block", {
            'target_pose': np.array([0.5, 0.5, 0.3, 0, 0, 0, 1]),
            'block_pose': np.array([0.3, 0.3, 0.0, 0, 0, 0, 1]),
            'gripper_state': np.array([1.0]),  # open
            'left_contact': np.array([0.0]),
            'right_contact': np.array([0.0])
        }),
        ("Approaching block", {
            'target_pose': np.array([0.3, 0.3, 0.05, 0, 0, 0, 1]),
            'block_pose': np.array([0.3, 0.3, 0.0, 0, 0, 0, 1]),
            'gripper_state': np.array([1.0]),
            'left_contact': np.array([0.0]),
            'right_contact': np.array([0.0])
        }),
        ("Contact made, gripper open", {
            'target_pose': np.array([0.3, 0.3, 0.02, 0, 0, 0, 1]),
            'block_pose': np.array([0.3, 0.3, 0.0, 0, 0, 0, 1]),
            'gripper_state': np.array([1.0]),
            'left_contact': np.array([1.0]),
            'right_contact': np.array([1.0])
        }),
        ("Grasping (closed gripper)", {
            'target_pose': np.array([0.3, 0.3, 0.02, 0, 0, 0, 1]),
            'block_pose': np.array([0.3, 0.3, 0.0, 0, 0, 0, 1]),
            'gripper_state': np.array([0.0]),  # closed
            'left_contact': np.array([1.0]),
            'right_contact': np.array([1.0])
        }),
        ("Lifted slightly (5cm)", {
            'target_pose': np.array([0.3, 0.3, 0.07, 0, 0, 0, 1]),
            'block_pose': np.array([0.3, 0.3, 0.05, 0, 0, 0, 1]),
            'gripper_state': np.array([0.0]),
            'left_contact': np.array([1.0]),
            'right_contact': np.array([1.0])
        }),
        ("Lifted to 15cm", {
            'target_pose': np.array([0.3, 0.3, 0.17, 0, 0, 0, 1]),
            'block_pose': np.array([0.3, 0.3, 0.15, 0, 0, 0, 1]),
            'gripper_state': np.array([0.0]),
            'left_contact': np.array([1.0]),
            'right_contact': np.array([1.0])
        }),
        ("SUCCESS: Lifted above 25cm", {
            'target_pose': np.array([0.3, 0.3, 0.27, 0, 0, 0, 1]),
            'block_pose': np.array([0.3, 0.3, 0.25, 0, 0, 0, 1]),
            'gripper_state': np.array([0.0]),
            'left_contact': np.array([1.0]),
            'right_contact': np.array([1.0])
        }),
    ]
    
    total = 0.0
    for desc, obs in scenarios:
        r = compute_reward(obs)
        total += r
        print(f"{desc:35s} Reward: {r:7.2f}  (Cumulative: {total:7.2f})")
    
    print("-" * 60)
    print(f"Total trajectory reward: {total:.2f}")
    print("\nThis should be strongly positive for successful demonstrations!")