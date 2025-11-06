#!/usr/bin/env python3
import numpy as np, os, argparse
from scipy.signal import savgol_filter

def normalize_quat(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q

def fix_quat_sign(q):
    """Ensure continuous quaternion sign (avoid sudden flips)."""
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] = -q[i]
    return q

def smooth_positions(xyz, window=9, poly=3):
    if window % 2 == 0: window += 1
    out = np.zeros_like(xyz)
    for i in range(xyz.shape[1]):
        out[:, i] = savgol_filter(xyz[:, i], window, poly, mode="nearest")
    return out

def majority_filter(arr, window=5):
    out = np.copy(arr)
    for i in range(len(arr)):
        lo = max(0, i-window//2)
        hi = min(len(arr), i+window//2+1)
        out[i] = np.round(np.mean(arr[lo:hi]))
    return out

def main(data_dir, out_dir):
    obs = dict(np.load(os.path.join(data_dir, "obs.npz")))
    os.makedirs(out_dir, exist_ok=True)

    for key in ("block_pose", "target_pose"):
        if key not in obs: continue
        pose = obs[key].astype(np.float64)
        pos, quat = pose[:, :3], pose[:, 3:7]
        quat = normalize_quat(quat)
        quat = fix_quat_sign(quat)
        pos = smooth_positions(pos)
        obs[key] = np.hstack([pos, quat])

    for key in ("arm_joints",):
        if key in obs:
            obs[key] = smooth_positions(obs[key])

    for key in ("gripper_state", "contact"):
        if key in obs:
            arr = obs[key].astype(np.float64).squeeze()
            obs[key] = majority_filter(arr).reshape(-1, 1)

    # Recompute actions from target pose + wrist + gripper
    T = len(next(iter(obs.values())))
    pos = obs["target_pose"][:, :3]
    wrist = obs["wrist_angle"]
    grip = obs["gripper_state"]
    delta = np.vstack([np.zeros((1, 3)), np.diff(pos, axis=0)])
    delta_wrist = np.vstack([np.zeros((1, 1)), np.diff(wrist, axis=0)])
    delta_grip = np.vstack([np.zeros((1, 1)), np.diff(grip, axis=0)])
    actions = np.hstack([delta, delta_wrist, delta_grip])
    actions /= np.max(np.abs(actions), axis=0, where=(np.abs(actions) > 0), initial=1.0)

    np.savez_compressed(os.path.join(out_dir, "obs.npz"), **obs)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    if os.path.exists(os.path.join(data_dir, "rewards.npy")):
        os.system(f"cp {os.path.join(data_dir, 'rewards.npy')} {out_dir}/")
    print(f"Saved cleaned data to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(args.data, args.out)
