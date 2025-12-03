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
            target_pos: [x, y, z] target position. If None, uses actual_pose from obs
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
        # Using actual_pose which tracks the robot position
        if 'actual_pose' not in obs:
            return 0.0
            
        actual_pose = np.atleast_1d(obs['actual_pose'])
        if len(actual_pose) < 3:
            return 0.0
            
        current_pos = actual_pose[:3]
        
        # If no fixed target, use the actual_pose from environment (for reaching tasks)
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
    def __init__(self, target_pos=np.array([0.1, 0.35, 0.35])):
        self.target_pos = np.array(target_pos)
        # ⬅️ Added state for tracking progress
        self.best_distance = float('inf') 

    def reset(self):
        print("[SmoothReachReward] Target:", self.target_pos)
        self.best_distance = float('inf') # Reset best distance

    def __call__(self, obs):
        if 'actual_pose' not in obs:
            return 0.0

        current_pos = np.atleast_1d(obs['actual_pose'])[:3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        
        reward = 0.0

        # 1. Exponential Proximity: Sharp, smooth signal toward goal
        # Increased factor from -6.0 to -10.0 for sharper falloff
        proximity = np.exp(-10.0 * distance)
        reward += proximity
        
        # 2. Progress Bonus: Reward movement that is closer than ever before
        if distance < self.best_distance:
            if self.best_distance != float('inf'): # ⬅️ ADD THIS CHECK
                progress_bonus = 1.0 * (self.best_distance - distance)
                reward += progress_bonus
                
            self.best_distance = distance
        
        # 3. Hold Bonus: Encourages stable position holding
        hold_bonus = 0.0
        if distance < 0.04: # within 4 cm
            hold_bonus += 0.1          # Increased from 0.02
        
        reward += hold_bonus

        # 4. Final Success Bonuses
        if distance < 0.02: # within 2 cm
            reward += 1.0
        if distance < 0.01: # within 1 cm
            reward += 2.0
            
        MAX_REWARD_CAP = 4.0 
        
        # Ensure the returned value is a float and is capped
        return float(np.clip(reward, 0.0, MAX_REWARD_CAP))
