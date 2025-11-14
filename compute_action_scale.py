#!/usr/bin/env python3
"""
Compute action scaling factors from training demonstrations.
This calculates the max_abs values used during training.
"""
import numpy as np
from pathlib import Path
import argparse


def compute_scaling_from_demo(demo_dir):
    """Load a demo and compute its action scaling factors."""
    actions = np.load(Path(demo_dir) / "actions.npy")
    
    # Actions are already normalized deltas in the saved demos
    # We need to find what they were divided by
    # Looking at your extraction script, you compute:
    # actions = np.diff(eff, axis=0, prepend=eff[0:1])
    # max_abs = np.abs(actions).max(axis=0)
    # actions /= max_abs
    
    # So to reverse it, we need the original unnormalized deltas
    # Let's load the observations and recompute
    obs_data = np.load(Path(demo_dir) / "obs.npz")
    
    # Reconstruct end-effector state
    target_pose = obs_data.get("target_pose", np.zeros((len(actions) + 1, 7)))
    wrist_angle = obs_data.get("wrist_angle", np.zeros((len(actions) + 1, 1)))
    gripper_state = obs_data.get("gripper_state", np.zeros((len(actions) + 1, 1)))
    
    eff = np.hstack([
        target_pose[:, :3],  # x, y, z
        wrist_angle,
        gripper_state,
    ])
    
    # Compute raw deltas
    raw_deltas = np.diff(eff, axis=0, prepend=eff[0:1])
    
    # Get max absolute value per dimension
    max_abs = np.abs(raw_deltas).max(axis=0)
    max_abs[max_abs == 0] = 1.0  # Avoid division by zero
    
    return max_abs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                       default='/home/xavie/dreamer/dreamerv3/data/demos/success',
                       help='Directory containing demo folders')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    
    # Find all demo directories
    demo_dirs = []
    for item in data_path.iterdir():
        if item.is_dir() and (item / "obs.npz").exists() and (item / "actions.npy").exists():
            demo_dirs.append(item)
    
    if not demo_dirs:
        print(f"[ERROR] No demos found in {args.data_dir}")
        return
    
    print(f"[INFO] Found {len(demo_dirs)} demos")
    
    # Compute scaling for each demo
    all_scales = []
    for demo_dir in demo_dirs:
        try:
            scale = compute_scaling_from_demo(demo_dir)
            all_scales.append(scale)
            print(f"[INFO] {demo_dir.name}: scale={scale}")
        except Exception as e:
            print(f"[WARNING] Failed to process {demo_dir.name}: {e}")
    
    if not all_scales:
        print("[ERROR] No valid demos processed")
        return
    
    # Use maximum scale across all demos (most conservative)
    final_scale = np.max(all_scales, axis=0)
    
    print(f"\n[RESULT] Recommended action_scale:")
    print(f"  np.array({final_scale.tolist()})")
    print(f"\nCopy this into your dreamer_zmq_client.py:")
    print(f"  self.action_scale = np.array({final_scale.tolist()})")
    
    # Also show per-dimension info
    print(f"\nPer-dimension breakdown:")
    dims = ['x', 'y', 'z', 'wrist', 'gripper']
    for i, (dim, scale) in enumerate(zip(dims, final_scale)):
        print(f"  {dim:8s}: {scale:.6f}")


if __name__ == "__main__":
    main()