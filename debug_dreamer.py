#!/usr/bin/env python3
"""
Comprehensive debugging script for DreamerV3 training issues.
Run this to diagnose action space, observation space, and reward problems.
NO ROS DEPENDENCIES - runs in DreamerV3 Python 3.11 environment.
"""
import numpy as np
import os
import sys
from pathlib import Path

def check_action_statistics(data_dir):
    """Check action statistics across all demos."""
    print("\n" + "="*80)
    print("CHECKING ACTION STATISTICS")
    print("="*80)
    
    all_actions = []
    demo_dirs = []
    
    # Find demo directories
    for root, dirs, files in os.walk(data_dir):
        if 'actions.npy' in files:
            demo_dirs.append(root)
    
    for demo_dir in demo_dirs:
        actions = np.load(os.path.join(demo_dir, "actions.npy"))
        all_actions.append(actions)
        print(f"\n{demo_dir}:")
        print(f"  Shape: {actions.shape}")
        print(f"  Mean: {np.mean(actions, axis=0)}")
        print(f"  Std:  {np.std(actions, axis=0)}")
        print(f"  Min:  {np.min(actions, axis=0)}")
        print(f"  Max:  {np.max(actions, axis=0)}")
        print(f"  Range: {np.max(actions, axis=0) - np.min(actions, axis=0)}")
    
    # Combined statistics
    if all_actions:
        combined = np.vstack(all_actions)
        print(f"\n{'COMBINED STATISTICS':^80}")
        print(f"Total timesteps: {len(combined)}")
        print(f"Action dim: {combined.shape[1]}")
        print(f"Mean:  {np.mean(combined, axis=0)}")
        print(f"Std:   {np.std(combined, axis=0)}")
        print(f"Min:   {np.min(combined, axis=0)}")
        print(f"Max:   {np.max(combined, axis=0)}")
        print(f"Range: {np.max(combined, axis=0) - np.min(combined, axis=0)}")
        
        # Check if actions are near zero (normalized deltas should be ~[-1, 1])
        mean_abs = np.mean(np.abs(combined), axis=0)
        print(f"\nMean absolute: {mean_abs}")
        if np.all(mean_abs < 0.1):
            print("⚠️  WARNING: Actions are very small! Model may not learn meaningful movements.")
        
        # Check normalization
        max_abs = np.max(np.abs(combined), axis=0)
        print(f"Max absolute: {max_abs}")
        if not np.allclose(max_abs, 1.0, atol=0.1):
            print("⚠️  WARNING: Actions don't appear normalized to [-1, 1] range!")
            print("   Expected max_abs ≈ 1.0 for each dimension")
    
    return combined if all_actions else None


def check_observations(data_dir):
    """Check observation shapes and statistics."""
    print("\n" + "="*80)
    print("CHECKING OBSERVATIONS")
    print("="*80)
    
    demo_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if 'obs.npz' in files:
            demo_dirs.append(root)
    
    obs_shapes = {}
    
    for i, demo_dir in enumerate(demo_dirs):
        obs_data = np.load(os.path.join(demo_dir, "obs.npz"))
        print(f"\n{demo_dir}:")
        
        for key in obs_data.keys():
            shape = obs_data[key].shape
            print(f"  {key}: {shape}")
            
            if key not in obs_shapes:
                obs_shapes[key] = shape
            elif obs_shapes[key] != shape:
                print(f"    ⚠️  WARNING: Shape mismatch! Expected {obs_shapes[key]}")
    
    return obs_shapes


def check_rewards(data_dir):
    """Check reward statistics and distribution."""
    print("\n" + "="*80)
    print("CHECKING REWARDS")
    print("="*80)
    
    demo_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if 'obs.npz' in files:
            demo_dirs.append(root)
    
    # Import reward function
    sys.path.append(str(Path(__file__).parent))
    try:
        from embodied.envs.reward_function import compute_reward
    except:
        print("⚠️  Could not import reward function!")
        return
    
    all_rewards = []
    
    for demo_dir in demo_dirs:
        obs_data = np.load(os.path.join(demo_dir, "obs.npz"))
        
        # Compute rewards
        T = len(obs_data[list(obs_data.keys())[0]])
        rewards = np.zeros(T)
        
        for t in range(T):
            obs_t = {k: obs_data[k][t] for k in obs_data.keys()}
            rewards[t] = compute_reward(obs_t)
        
        all_rewards.append(rewards)
        
        print(f"\n{demo_dir}:")
        print(f"  Total reward: {rewards.sum():.2f}")
        print(f"  Mean reward:  {rewards.mean():.3f}")
        print(f"  Max reward:   {rewards.max():.3f}")
        print(f"  Min reward:   {rewards.min():.3f}")
        print(f"  Positive steps: {(rewards > 0).sum()} / {len(rewards)}")
        print(f"  Zero steps:     {(rewards == 0).sum()} / {len(rewards)}")
        print(f"  Negative steps: {(rewards < 0).sum()} / {len(rewards)}")
    
    if all_rewards:
        combined = np.concatenate(all_rewards)
        print(f"\n{'COMBINED REWARD STATISTICS':^80}")
        print(f"Total reward: {combined.sum():.2f}")
        print(f"Mean: {combined.mean():.3f}")
        print(f"Std:  {combined.std():.3f}")
        
        if combined.sum() < 0:
            print("⚠️  WARNING: Total reward is negative! Model will learn to end episodes quickly.")
        if np.abs(combined.mean()) < 0.01:
            print("⚠️  WARNING: Mean reward near zero! Reward signal may be too weak.")


def check_trajectory_properties(data_dir):
    """Check if trajectories show successful task completion."""
    print("\n" + "="*80)
    print("CHECKING TRAJECTORY PROPERTIES")
    print("="*80)
    
    demo_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if 'obs.npz' in files:
            demo_dirs.append(root)
    
    for demo_dir in demo_dirs:
        obs_data = np.load(os.path.join(demo_dir, "obs.npz"))
        print(f"\n{demo_dir}:")
        
        # Check block height progression
        if 'block_pose' in obs_data:
            block_pos = obs_data['block_pose'][:, :3]
            block_z = block_pos[:, 2]
            
            print(f"  Block Z: start={block_z[0]:.3f}, end={block_z[-1]:.3f}, max={block_z.max():.3f}")
            
            if block_z[-1] > 0.25:
                print(f"  ✓ Block lifted successfully!")
            else:
                print(f"  ⚠️  Block not lifted above 0.25m in this demo")
            
            # Check if block moves at all
            block_movement = np.linalg.norm(block_pos[-1] - block_pos[0])
            print(f"  Block total movement: {block_movement:.3f}m")
        
        # Check gripper contacts
        if 'left_contact' in obs_data and 'right_contact' in obs_data:
            left = obs_data['left_contact']
            right = obs_data['right_contact']
            both_contact = (left > 0.5) & (right > 0.5)
            
            print(f"  Contact timesteps: {both_contact.sum()} / {len(left)}")
            if both_contact.sum() > 0:
                print(f"  ✓ Gripper made contact")
            else:
                print(f"  ⚠️  No gripper contact detected!")
        
        # Check gripper state
        if 'gripper_state' in obs_data:
            gripper = obs_data['gripper_state']
            closed_steps = (gripper < 0.5).sum()
            print(f"  Gripper closed: {closed_steps} / {len(gripper)} timesteps")


def check_action_space_inference(data_dir, action_scale):
    """Check if action scale matches training normalization."""
    print("\n" + "="*80)
    print("CHECKING ACTION SCALE FOR INFERENCE")
    print("="*80)
    
    # Compute actual normalization used in training
    demo_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if 'actions.npy' in files:
            demo_dirs.append(root)
    
    print("Computing normalization from training data...")
    
    # Load all demos and find the normalization that was used
    all_effector_deltas = []
    
    for demo_dir in demo_dirs:
        obs_data = np.load(os.path.join(demo_dir, "obs.npz"))
        
        # Reconstruct effector positions
        target_pose = obs_data.get('target_pose', np.zeros((len(obs_data['arm_joints']), 7)))
        wrist_angle = obs_data.get('wrist_angle', np.zeros((len(obs_data['arm_joints']), 1)))
        gripper_state = obs_data.get('gripper_state', np.zeros((len(obs_data['arm_joints']), 1)))
        
        eff = np.hstack([
            target_pose[:, :3],
            wrist_angle,
            gripper_state,
        ])
        
        # Compute deltas (same as in extract_rosbag.py)
        deltas = np.diff(eff, axis=0, prepend=eff[0:1])
        all_effector_deltas.append(deltas)
    
    combined_deltas = np.vstack(all_effector_deltas)
    actual_scale = np.abs(combined_deltas).max(axis=0)
    actual_scale[actual_scale == 0] = 1.0
    
    print(f"\nActual normalization scale used in training:")
    print(f"  {actual_scale}")
    
    print(f"\nYour inference action_scale:")
    print(f"  {action_scale}")
    
    print(f"\nRatio (yours / actual):")
    ratio = action_scale / actual_scale
    print(f"  {ratio}")
    
    if not np.allclose(ratio, 1.0, atol=0.1):
        print("\n⚠️  WARNING: Action scale mismatch!")
        print("   Your action_scale doesn't match training normalization!")
        print("   This will cause wrong action magnitudes during inference.")
        print("\n   SOLUTION: Update action_scale in dreamer_zmq_client.py to:")
        print(f"   self.action_scale = np.array({list(actual_scale)})")
    else:
        print("\n✓ Action scale matches training normalization")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to demo data directory')
    parser.add_argument('--action_scale', type=float, nargs=5, 
                       default=[0.021972, 0.026301, 0.041695, 11.62197, 1.0],
                       help='Action scale from inference (x y z wrist gripper)')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: {args.data_dir} does not exist!")
        return
    
    print(f"Analyzing demos in: {args.data_dir}")
    
    # Run all checks
    actions = check_action_statistics(args.data_dir)
    obs_shapes = check_observations(args.data_dir)
    check_rewards(args.data_dir)
    check_trajectory_properties(args.data_dir)
    
    if actions is not None:
        check_action_space_inference(args.data_dir, np.array(args.action_scale))
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nCheck the warnings above to identify issues.")


if __name__ == "__main__":
    main()