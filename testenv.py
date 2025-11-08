# test_dreamer_env.py
import sys
import numpy as np
sys.path.insert(0, '.')

from embodied.envs.arm import Arm

def test_dreamer_compatibility(data_dir):
    print("=" * 60)
    print("DREAMERV3 COMPATIBILITY TEST")
    print("=" * 60)
    
    # 1. Load environment
    print("\n1. Loading environment...")
    env = Arm(data_dir)
    print(f"✓ Environment loaded")
    print(f"   Episode length: {env.length}")
    print(f"   Action dim: {env.actions.shape[1]}")
    
    # 2. Check observation and action spaces
    print("\n2. Checking spaces...")
    print("   Observation space:")
    for key, space in env.obs_space.items():
        if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
            print(f"      {key}: {space.shape}")
    print(f"   Action space: {env.act_space['action'].shape}")
    
    # 3. Test episode rollout
    print("\n3. Testing episode rollout...")
    obs = env.step({"reset": True, "action": np.zeros(env.actions.shape[1])})
    
    trajectory = [obs]
    for i in range(min(20, env.length - 1)):
        obs = env.step({"reset": False, "action": np.zeros(env.actions.shape[1])})
        trajectory.append(obs)
    
    print(f"✓ Collected {len(trajectory)} steps")
    
    # 4. Check data quality
    print("\n4. Data quality checks...")
    
    # Check for NaN/Inf in observations
    has_nan = False
    for key in env.obs_keys:
        if np.any(np.isnan(env.obs_data[key])):
            print(f"   ✗ {key} has NaN values!")
            has_nan = True
        if np.any(np.isinf(env.obs_data[key])):
            print(f"   ✗ {key} has Inf values!")
            has_nan = True
    
    if not has_nan:
        print("   ✓ No NaN/Inf in observations")
    
    # Check rewards
    print(f"\n5. Reward analysis:")
    print(f"   Shape: {env.rewards.shape}")
    print(f"   Min: {env.rewards.min():.4f}")
    print(f"   Max: {env.rewards.max():.4f}")
    print(f"   Mean: {env.rewards.mean():.4f}")
    print(f"   Std: {env.rewards.std():.4f}")
    print(f"   Unique values: {len(np.unique(env.rewards))}")
    
    if np.allclose(env.rewards, env.rewards[0]):
        print("   ✗ WARNING: Rewards are constant! Learning may be impossible.")
    else:
        print("   ✓ Rewards vary")
    
    # Check actions
    print(f"\n6. Action analysis:")
    print(f"   Shape: {env.actions.shape}")
    for i in range(env.actions.shape[1]):
        print(f"   Dim {i}: min={env.actions[:,i].min():.3f}, "
              f"max={env.actions[:,i].max():.3f}, "
              f"std={env.actions[:,i].std():.3f}")
    
    action_std = env.actions.std(axis=0)
    if np.any(action_std < 1e-6):
        print(f"   ✗ WARNING: Some action dimensions are constant!")
    else:
        print("   ✓ All action dimensions vary")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    # UPDATE THIS PATH
    data_dir = "data/demo2"
    test_dreamer_compatibility(data_dir)