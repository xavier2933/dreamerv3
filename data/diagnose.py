#!/usr/bin/env python3
import numpy as np
import argparse
import os

def stats(name, arr):
    arr = np.asarray(arr)
    print(f"{name:>15s}: shape={arr.shape}, dtype={arr.dtype}")
    if np.issubdtype(arr.dtype, np.number):
        print(f"    min={arr.min():.6g}, max={arr.max():.6g}, mean={arr.mean():.6g}, std={arr.std():.6g}")
    # basic delta stats
    if arr.shape[0] > 1 and arr.ndim >= 2:
        d = np.diff(arr, axis=0)
        print(f"    Δ mean={d.mean():.6g}, Δ std={d.std():.6g}, Δ max={d.max():.6g}, Δ 99p={np.percentile(np.abs(d),99):.6g}")
    # uniqueness in time series per-dimension
    if arr.ndim == 2:
        uniq_counts = [len(np.unique(arr[:,i])) for i in range(min(arr.shape[1], 6))]
        print(f"    unique counts (first dims) = {uniq_counts}")

def detect_spikes(arr, thresh=5.0):
    d = np.abs(np.diff(arr, axis=0))
    big = (d > thresh)
    if big.any():
        idxs = np.unique(np.where(big)[0])
        print(f"  Found {len(idxs)} spike timesteps (Δ>{thresh}) indices (showing up to 10): {idxs[:10]}")
    else:
        print("  No spikes above threshold")

def main(data_dir):
    obs = np.load(os.path.join(data_dir, "obs.npz"))
    actions = np.load(os.path.join(data_dir, "actions.npy"))
    rewards = np.load(os.path.join(data_dir, "rewards.npy")) if os.path.exists(os.path.join(data_dir, "rewards.npy")) else None

    print("Keys:", list(obs.keys()))
    for k in obs.files:
        stats(k, obs[k])
        if obs[k].ndim == 2:
            detect_spikes(obs[k])
        print("")

    stats("actions", actions)
    detect_spikes(actions)
    if rewards is not None:
        stats("rewards", rewards)

    # Check simple time consistency: large fraction of identical consecutive rows?
    def fraction_constant(arr):
        arr = np.asarray(arr)
        if arr.shape[0] < 2: return 0.0
        same = (arr[1:] == arr[:-1])
        # for multi-dim, require all dims equal
        if same.ndim == 2:
            same_all = same.all(axis=1)
            return same_all.mean()
        return same.mean()
    print("\nFraction of timesteps that are identical to previous (per-channel):")
    for k in obs.files:
        print(f"  {k}: {fraction_constant(obs[k]):.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()
    main(args.data)
