#!/usr/bin/env python3
"""
Test script for reward function.
Loads demo data, computes rewards, and plots them.

Usage:
    python test_reward_function.py --data_dir ./offline
    python test_reward_function.py --data_dir ./offline/demo1
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from reward_function import compute_reward


def find_demo_dirs(data_dir):
    """
    Find demonstration directories.
    Returns list of paths containing demo data.
    """
    # Check if data_dir itself contains demo files
    if (os.path.exists(os.path.join(data_dir, "obs.npz")) and
        os.path.exists(os.path.join(data_dir, "actions.npy"))):
        return [data_dir]
    
    # Look for subdirectories with demo files
    demo_dirs = []
    for subdir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if os.path.isdir(subdir):
            if (os.path.exists(os.path.join(subdir, "obs.npz")) and
                os.path.exists(os.path.join(subdir, "actions.npy"))):
                demo_dirs.append(subdir)
    
    if not demo_dirs:
        raise ValueError(f"No demo data found in {data_dir}")
    
    return demo_dirs


def compute_rewards_from_obs(obs_data):
    """
    Compute rewards for entire trajectory.
    
    Args:
        obs_data: dict from np.load("obs.npz")
    
    Returns:
        np.ndarray: rewards of shape (T,)
    """
    first_key = list(obs_data.keys())[0]
    T = len(obs_data[first_key])
    
    rewards = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        obs_t = {k: obs_data[k][t] for k in obs_data.keys()}
        rewards[t] = compute_reward(obs_t)
    
    return rewards


def print_observation_info(obs_data):
    """Print information about observation structure."""
    print("\n" + "="*60)
    print("OBSERVATION STRUCTURE")
    print("="*60)
    for key, value in obs_data.items():
        print(f"{key:20s}: shape={value.shape}, dtype={value.dtype}")
        print(f"{'':20s}  min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")
    print("="*60 + "\n")


def plot_rewards(all_rewards, demo_names, save_path=None):
    """
    Plot rewards vs timestep for all demos.
    
    Args:
        all_rewards: list of reward arrays
        demo_names: list of demo names
        save_path: optional path to save figure
    """
    num_demos = len(all_rewards)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_demos + 1, 1, figsize=(12, 4 * (num_demos + 1)))
    if num_demos == 1:
        axes = [axes[0], axes[1]]
    
    # Plot individual demos
    for i, (rewards, name) in enumerate(zip(all_rewards, demo_names)):
        ax = axes[i]
        timesteps = np.arange(len(rewards))
        
        ax.plot(timesteps, rewards, linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Reward')
        ax.set_title(f'{name} | Total Reward: {rewards.sum():.2f} | Mean: {rewards.mean():.3f}')
        
        # Add statistics text
        stats_text = f'Min: {rewards.min():.2f}\nMax: {rewards.max():.2f}\nStd: {rewards.std():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot all demos together
    ax = axes[-1]
    offset = 0
    colors = plt.cm.tab10(np.linspace(0, 1, num_demos))
    
    for i, (rewards, name) in enumerate(zip(all_rewards, demo_names)):
        timesteps = np.arange(len(rewards)) + offset
        ax.plot(timesteps, rewards, label=name, linewidth=1.5, alpha=0.8, color=colors[i])
        offset += len(rewards)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Timestep (concatenated)')
    ax.set_ylabel('Reward')
    ax.set_title('All Demos Combined')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[INFO] Plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test reward function on demo data')
    parser.add_argument('--data_dir', type=str, default='offline',
                        help='Directory containing demo data')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Path to save plot (e.g., rewards.png)')
    args = parser.parse_args()
    
    print(f"[INFO] Loading demos from: {args.data_dir}")
    
    # Find all demo directories
    demo_dirs = find_demo_dirs(args.data_dir)
    print(f"[INFO] Found {len(demo_dirs)} demo(s)")
    
    all_rewards = []
    demo_names = []
    
    # Process each demo
    for demo_dir in demo_dirs:
        demo_name = os.path.basename(demo_dir) if demo_dir != args.data_dir else "demo"
        print(f"\n[INFO] Processing: {demo_name}")
        print(f"  Path: {demo_dir}")
        
        # Load observations and actions
        obs_data = np.load(os.path.join(demo_dir, "obs.npz"))
        actions = np.load(os.path.join(demo_dir, "actions.npy"))
        
        # Print observation info (only for first demo)
        if len(all_rewards) == 0:
            print_observation_info(obs_data)
        
        # Compute rewards
        print(f"  Computing rewards for {len(actions)} timesteps...")
        rewards = compute_rewards_from_obs(obs_data)
        
        # Print reward statistics
        print(f"  Reward statistics:")
        print(f"    Total:  {rewards.sum():.2f}")
        print(f"    Mean:   {rewards.mean():.4f}")
        print(f"    Std:    {rewards.std():.4f}")
        print(f"    Min:    {rewards.min():.4f}")
        print(f"    Max:    {rewards.max():.4f}")
        print(f"    Median: {np.median(rewards):.4f}")
        
        # Check if any rewards are non-zero
        nonzero_count = np.count_nonzero(rewards)
        print(f"    Non-zero timesteps: {nonzero_count}/{len(rewards)} ({100*nonzero_count/len(rewards):.1f}%)")
        
        all_rewards.append(rewards)
        demo_names.append(demo_name)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_timesteps = sum(len(r) for r in all_rewards)
    total_reward = sum(r.sum() for r in all_rewards)
    print(f"Total demos:     {len(all_rewards)}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Total reward:    {total_reward:.2f}")
    print(f"Average reward per timestep: {total_reward/total_timesteps:.4f}")
    print("="*60)
    
    # Plot results
    print("\n[INFO] Generating plots...")
    plot_rewards(all_rewards, demo_names, save_path=args.save_plot)


if __name__ == "__main__":
    main()