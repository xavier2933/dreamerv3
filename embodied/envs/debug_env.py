import numpy as np
from arm import Arm

def verify_env_setup(data_dir):
    """Basic checks for environment setup"""
    print("=" * 60)
    print("ENVIRONMENT SETUP VERIFICATION")
    print("=" * 60)
    
    # 1. Initialize environment
    print("\n1. Initializing environment...")
    try:
        env = Arm(data_dir)
        print("✓ Environment initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False
    
    # 2. Check data shapes
    print("\n2. Data shapes:")
    print(f"   Actions: {env.actions.shape}")
    print(f"   Rewards: {env.rewards.shape}")
    print(f"   Episode length: {env.length}")
    for key in env.obs_keys:
        print(f"   {key}: {env.obs_data[key].shape}")
    
    # 3. Check spaces
    print("\n3. Observation space:")
    for key, space in env.obs_space.items():
        print(f"   {key}: {space.dtype} {space.shape}")
    
    print("\n   Action space:")
    for key, space in env.act_space.items():
        print(f"   {key}: {space.dtype} {space.shape}")
    
    # 4. Test reset
    print("\n4. Testing reset...")
    obs = env.step({"reset": True, "action": np.zeros(env.actions.shape[1])})
    assert obs["is_first"], "Reset should set is_first=True"
    assert not obs["is_last"], "Reset should set is_last=False"
    print("✓ Reset works correctly")
    print(f"   First observation keys: {list(obs.keys())}")
    
    # 5. Test episode rollout
    print("\n5. Testing full episode rollout...")
    obs = env.step({"reset": True, "action": np.zeros(env.actions.shape[1])})
    steps = 0
    reward_sum = 0
    
    while not obs["is_last"] and steps < env.length + 10:  # safety limit
        action = {"reset": False, "action": np.zeros(env.actions.shape[1])}
        obs = env.step(action)
        reward_sum += obs["reward"]
        steps += 1
    
    print(f"✓ Episode completed in {steps} steps")
    print(f"   Total reward: {reward_sum:.4f}")
    print(f"   is_last flag set: {obs['is_last']}")
    print(f"   is_terminal flag set: {obs['is_terminal']}")
    
    # 6. Data quality checks
    print("\n6. Data quality checks:")
    
    # Check for NaNs/Infs
    has_issues = False
    for key in env.obs_keys:
        data = env.obs_data[key]
        if np.any(np.isnan(data)):
            print(f"   ✗ WARNING: {key} contains NaN values")
            has_issues = True
        if np.any(np.isinf(data)):
            print(f"   ✗ WARNING: {key} contains Inf values")
            has_issues = True
    
    if np.any(np.isnan(env.actions)):
        print(f"   ✗ WARNING: actions contain NaN values")
        has_issues = True
    if np.any(np.isinf(env.actions)):
        print(f"   ✗ WARNING: actions contain Inf values")
        has_issues = True
    
    if not has_issues:
        print("   ✓ No NaN/Inf values detected")
    
    # Check action ranges
    print(f"\n   Action statistics:")
    print(f"   - Min: {env.actions.min(axis=0)}")
    print(f"   - Max: {env.actions.max(axis=0)}")
    print(f"   - Mean: {env.actions.mean(axis=0)}")
    print(f"   - Std: {env.actions.std(axis=0)}")
    
    # Check if actions are constant (potential issue)
    if np.allclose(env.actions.std(axis=0), 0):
        print("   ✗ WARNING: Actions appear to be constant!")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    return True

if __name__ == "__main__":
    data_dir = "data/demo1"  # Update this
    verify_env_setup(data_dir)