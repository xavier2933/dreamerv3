import numpy as np

class DipLiftReward:
    """
    Stateful reward function for 'Dip and Lift' validation task.
    
    Task:
    1. Dip: Move arm down (Y < 0.25)
    2. Lift: Move arm up (Y > 0.35)
    
    Axis: Y-axis (based on user logs)
    """
    def __init__(self):
        self.phase = 'dip'
        
    def reset(self):
        self.phase = 'dip'
        print("[DipLiftReward] Reset to DIP phase")
        
    def __call__(self, obs):
        reward = 0.0
        
        # Extract Y position from target_pose
        # target_pose is [x, y, z, qx, qy, qz, qw]
        if 'target_pose' not in obs:
            return 0.0
            
        target_pose = np.atleast_1d(obs['target_pose'])
        if len(target_pose) < 2:
            return 0.0
            
        # Y-axis is index 1
        current_y = target_pose[1]
        
        if self.phase == 'dip':
            # Target: Go down to Y=0.20
            # Range of motion seems to be ~0.2 to ~0.4 based on logs
            
            # Distance to target 0.20
            dist = abs(current_y - 0.20)
            
            # Reward: 1.0 at target, decays as we move away
            reward = 1.0 - np.tanh(dist * 5.0)
            
            # Transition condition: Y < 0.25
            if current_y < 0.25:
                self.phase = 'lift'
                reward += 5.0 # Bonus for transition
                print(f"[DipLiftReward] Phase Switch: DIP -> LIFT (y={current_y:.3f})")
                
        elif self.phase == 'lift':
            # Target: Go up to Y=0.40
            
            # Distance to target 0.40
            dist = abs(current_y - 0.40)
            
            # Reward: 1.0 at target
            reward = 1.0 - np.tanh(dist * 5.0)
            
            # Success condition: Y > 0.35
            if current_y > 0.35:
                reward += 2.0 # Bonus for holding high
                if np.random.rand() < 0.1: # Print 10% of the time to avoid spam
                    print(f"[DipLiftReward] SUCCESS: LIFTED (y={current_y:.3f})")
                
        return float(reward)

# Keep original for reference/fallback
def compute_reward(obs):
    """Original block pickup reward (stateless)."""
    # ... (original implementation omitted for brevity if not used)
    return 0.0