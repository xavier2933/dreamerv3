import numpy as np

class MoveAndGraspReward:
    """
    Simple stateful reward for: Move end-effector to target position + close gripper.
    
    Philosophy (from Danijar):
    - Dense, smooth rewards (avoid sparse 0/1 rewards)
    - Reward progress, not just goal achievement
    - Use simple distance-based shaping
    - Let the world model learn the dynamics
    """
    def __init__(self, target_pos=None, target_gripper=1.0):
        """
        Args:
            target_pos: [x, y, z] target position. If None, uses target_pose from obs
            target_gripper: Target gripper state (1.0 = closed, 0.0 = open)
        """
        self.target_pos = np.array(target_pos) if target_pos is not None else None
        self.target_gripper = target_gripper
        
        # Track best distance for progress reward
        self.best_distance = float('inf')
        
    def reset(self):
        self.best_distance = float('inf')
        print("[MoveAndGraspReward] Reset")
        
    def __call__(self, obs):
        reward = 0.0
        
        # Extract current end-effector position
        # Using target_pose which tracks the commanded position
        if 'target_pose' not in obs:
            return 0.0
            
        target_pose = np.atleast_1d(obs['target_pose'])
        if len(target_pose) < 3:
            return 0.0
            
        current_pos = target_pose[:3]
        
        # If no fixed target, use the target_pose from environment (for reaching tasks)
        if self.target_pos is None:
            # For now, let's use a fixed target - modify as needed
            self.target_pos = np.array([0.0, 0.3, 0.0])
        
        # === POSITION REWARD ===
        # Simple distance-based reward (dense and smooth)
        distance = np.linalg.norm(current_pos - self.target_pos)
        
        # Reward: Exponential decay from 1.0 at target to ~0 far away
        # Scale factor controls the "reach" - smaller = more forgiving
        position_reward = np.exp(-3.0 * distance)
        
        # Progress bonus: reward improvement over best distance
        if distance < self.best_distance:
            progress_bonus = 0.5 * (self.best_distance - distance)
            self.best_distance = distance
            reward += progress_bonus
        
        reward += position_reward
        
        # === GRIPPER REWARD ===
        # Only reward gripper when close to target (staged reward)
        if distance < 0.1:  # Within 10cm
            if 'gripper_state' in obs:
                gripper_state = np.atleast_1d(obs['gripper_state'])[0]
                
                # Reward moving gripper toward target state
                gripper_error = abs(gripper_state - self.target_gripper)
                gripper_reward = 0.5 * (1.0 - gripper_error)
                
                reward += gripper_reward
                
                # Success bonus: close to target AND gripper in correct state
                if distance < 0.05 and gripper_error < 0.1:
                    reward += 2.0
        
        return float(reward)


class SimpleReachReward:
    def __init__(self, target_pos=np.array([0.0, 0.3, 0.0])):
        self.target_pos = np.array(target_pos)
        self.best_distance = float('inf')
        self.consecutive_success = 0
        
    def reset(self):
        self.best_distance = float('inf')
        self.consecutive_success = 0
        print(f"[SimpleReachReward] Reset - Target: {self.target_pos}")
        
    def __call__(self, obs):
        if 'target_pose' not in obs:
            return 0.0
            
        current_pos = np.atleast_1d(obs['target_pose'])[:3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        
        # Core distance reward: [0, 1] range
        # At target (0m): 1.0, at 20cm: ~0.25, at 40cm: ~0.06
        reward = np.exp(-3.0 * distance)
        
        # Small progress bonus (keeps total < 1.5)
        if self.best_distance == float('inf'):
            self.best_distance = distance
        elif distance < self.best_distance:
            improvement = self.best_distance - distance
            reward += np.clip(improvement * 2.0, 0, 0.3)  # Max +0.3
            self.best_distance = distance
            
        # Success bonus: Bigger reward for being at target
        if distance < 0.05:
            self.consecutive_success += 1
            # Extra 0.5 per step at target (makes total ~1.5)
            reward += 0.5
            
            # Small growing bonus for persistence (caps at +0.2)
            persistence_bonus = min(0.2, self.consecutive_success * 0.001)
            reward += persistence_bonus
            
            if self.consecutive_success % 100 == 0:
                print(f"[SUCCESS] Held target for {self.consecutive_success} steps")
        else:
            self.consecutive_success = 0
            
        return float(reward)