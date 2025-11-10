import numpy as np

def compute_reward(obs):
    """
    Compute reward for block pickup task from a single observation.
    
    Based on your ROS bag data structure:
    - arm_joints: joint positions (array)
    - block_pose: [x, y, z, qx, qy, qz, qw] - block position and orientation
    - target_pose: [x, y, z, qx, qy, qz, qw] - end effector target
    - wrist_angle: wrist joint angle (scalar)
    - gripper_state: gripper open/close state (0=open, 1=closed)
    - left_contact: left finger contact (0 or 1)
    - right_contact: right finger contact (0 or 1)
    
    Args:
        obs: dict with observation keys from your ROS data
    
    Returns:
        float: reward value
    """
    reward = 0.0

    # === Extract positions ===
    target_pos = None
    block_pos = None

    if 'target_pose' in obs:
        target_pose = np.atleast_1d(obs['target_pose'])
        if len(target_pose) >= 3:
            target_pos = target_pose[:3]  # [x, y, z]

    if 'block_pose' in obs:
        block_pose = np.atleast_1d(obs['block_pose'])
        if len(block_pose) >= 3:
            block_pos = block_pose[:3]  # [x, y, z]

    # === Extract contact and gripper state ===
    left_contact = float(np.atleast_1d(obs.get('left_contact', [0.0]))[0])
    right_contact = float(np.atleast_1d(obs.get('right_contact', [0.0]))[0])
    gripper_state = float(np.atleast_1d(obs.get('gripper_state', [1.0]))[0])  # 1=open, 0=closed (adjust if reversed)

    # === Distance reward ===
    # Only penalize distance while NOT holding the block
    if target_pos is not None and block_pos is not None:
        distance = np.linalg.norm(target_pos - block_pos)

        holding_block = (gripper_state < 0.5 and left_contact > 0.5 and right_contact > 0.5)

        if not holding_block:
            reward += -1.9 * distance  # Approach block
        else:
            reward += 0.2  # Small constant bonus for maintaining grip stability

    # === Contact reward ===
    if left_contact > 0.5 and right_contact > 0.5:
        reward += 5.0  # Reward both contacts

    # === Success reward ===
    if block_pos is not None and left_contact > 0.5 and right_contact > 0.5:
        block_height = block_pos[2]

        # Reward lifting with gripper closed
        if gripper_state < 0.5 and block_height > 0.25:
            reward += 5.0

    return float(reward)



class BlockPickupReward:
    """
    Reward function wrapper for block pickup task.
    Can be used in both offline preprocessing and online env.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def compute_reward(self, obs, prev_obs=None):
        """Use the standalone compute_reward function."""
        return compute_reward(obs)
    
    def compute_trajectory_rewards(self, obs_dict):
        """
        Compute rewards for entire trajectory.
        
        Args:
            obs_dict: dict of observation arrays, e.g. {'gripper_pos': array(T, 3), ...}
        
        Returns:
            np.ndarray: rewards of shape (T,)
        """
        # Get trajectory length
        first_key = list(obs_dict.keys())[0]
        T = len(obs_dict[first_key])
        
        rewards = np.zeros(T, dtype=np.float32)
        
        for t in range(T):
            # Extract observation at time t
            obs_t = {k: v[t] for k, v in obs_dict.items()}
            rewards[t] = compute_reward(obs_t)
        
        return rewards


# Example usage:
if __name__ == "__main__":
    # Test with dummy data matching your ROS structure
    
    # Test 1: Arm far from block, gripper open
    obs = {
        'arm_joints': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'target_pose': np.array([0.5, 0.5, 0.2, 0, 0, 0, 1]),  # end effector
        'block_pose': np.array([0.5, 0.5, 0.0, 0, 0, 0, 1]),   # block on table
        'wrist_angle': np.array([0.0]),
        'gripper_state': np.array([0.0]),  # open
        'left_contact': np.array([0.0]),
        'right_contact': np.array([0.0])
    }
    print(f"Reward (far, open, no contact): {compute_reward(obs):.3f}")
    
    # Test 2: Close to block with contacts
    obs['target_pose'] = np.array([0.5, 0.5, 0.01, 0, 0, 0, 1])  # close
    obs['left_contact'] = np.array([1.0])
    obs['right_contact'] = np.array([1.0])
    print(f"Reward (close, open, both contacts): {compute_reward(obs):.3f}")
    
    # Test 3: Grasping
    obs['gripper_state'] = np.array([1.0])  # closed
    print(f"Reward (close, closed, grasped): {compute_reward(obs):.3f}")
    
    # Test 4: Lifted
    obs['block_pose'] = np.array([0.5, 0.5, 0.20, 0, 0, 0, 1])  # lifted
    print(f"Reward (lifted and grasped): {compute_reward(obs):.3f}")