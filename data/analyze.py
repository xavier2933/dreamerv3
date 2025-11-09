#!/usr/bin/env python3
"""
Analyze and visualize DreamerV3 dataset converted from ROS bag.

Usage:
  python analyze_ros_dreamer_dataset.py --data data/demo1
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def summarize_array(name, arr):
    """Print shape and basic stats for an array."""
    arr = np.asarray(arr)
    print(f"{name:>15s}: shape={arr.shape}, dtype={arr.dtype}")
    if np.issubdtype(arr.dtype, np.number):
        print(f"    min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}, std={arr.std():.3f}")


def analyze_dataset(data_dir):
    obs_path = os.path.join(data_dir, "obs.npz")
    act_path = os.path.join(data_dir, "actions.npy")
    rew_path = os.path.join(data_dir, "rewards.npy")

    assert os.path.exists(obs_path), f"Missing {obs_path}"
    assert os.path.exists(act_path), f"Missing {act_path}"

    obs = np.load(obs_path)
    actions = np.load(act_path)
    rewards = np.load(rew_path) if os.path.exists(rew_path) else None

    print("=== Dataset Summary ===")
    print(f"Loaded from: {data_dir}")
    print(f"Keys in obs: {list(obs.keys())}\n")

    # --- Print summaries ---
    for key in obs.keys():
        summarize_array(key, obs[key])
    summarize_array("actions", actions)
    if rewards is not None:
        summarize_array("rewards", rewards)

    # --- Sanity check lengths ---
    T_obs = min(v.shape[0] for v in obs.values())
    T_act = actions.shape[0]
    print(f"\nAligned sequence length: {T_obs}")
    if abs(T_obs - T_act) > 1:
        print(f"⚠️ Warning: observation length ({T_obs}) and actions length ({T_act}) differ")

    # --- Plot helpers ---
    os.makedirs(os.path.join(data_dir, "plots"), exist_ok=True)

    def plot_series(arr, title, fname, labels=None):
        plt.figure(figsize=(8, 3))
        
        if arr.ndim == 1:
            plt.plot(arr)
        else:
            # Plot each column with a label
            for i in range(arr.shape[1]):
                label = labels[i] if labels and i < len(labels) else f"Dim {i}"
                plt.plot(arr[:, i], label=label)
            plt.legend()
        
        plt.title(title)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "plots", fname))
        plt.close()

    def plot_contacts_overlay(left, right, fname):
        plt.figure(figsize=(8, 3))
        plt.step(range(len(left)), left.squeeze(), where="post", label="Left Contact")
        plt.step(range(len(right)), right.squeeze(), where="post", label="Right Contact")
        plt.title("Gripper Contacts")
        plt.xlabel("Timestep")
        plt.ylabel("Contact (1=True)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "plots", fname))
        plt.close()

    # --- Generate plots ---
    print("\nGenerating plots...")
    if "block_pose" in obs:
        plot_series(obs["block_pose"][:, :3], "Block Position XYZ", "block_pose_xyz.png", labels=['X', 'Y', 'Z'])
    if "target_pose" in obs:
        plot_series(obs["target_pose"][:, :3], "Target Position XYZ", "target_pose_xyz.png")
    if "arm_joints" in obs:
        plot_series(obs["arm_joints"][:, :7], "Arm Joint Angles", "arm_joints.png")
    if "wrist_angle" in obs:
        plot_series(obs["wrist_angle"], "Wrist Angle", "wrist_angle.png")
    if "gripper_state" in obs:
        plot_series(obs["gripper_state"], "Gripper Command", "gripper_state.png")

    # New: contact plots
    left = obs["left_contact"] if "left_contact" in obs else None
    right = obs["right_contact"] if "right_contact" in obs else None
    if left is not None:
        plot_series(left, "Left Contact Detected", "left_contact.png")
    if right is not None:
        plot_series(right, "Right Contact Detected", "right_contact.png")
    if left is not None and right is not None:
        plot_contacts_overlay(left, right, "gripper_contacts_overlay.png")

    plot_series(actions, "Actions", "actions.png")
    if rewards is not None:
        plot_series(rewards, "Rewards", "rewards.png")

    print("✅ Analysis complete. Plots saved under:", os.path.join(data_dir, "plots"))
    print("Open them to visually check contact timing and grasp alignment.")
    print("\nNext: if data looks consistent, we can normalize and feed into Dreamer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory with obs.npz, actions.npy, rewards.npy")
    args = parser.parse_args()
    analyze_dataset(args.data)
