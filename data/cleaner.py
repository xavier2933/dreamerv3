 #!/usr/bin/env python3
import os
import argparse
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt


def pose_to_numpy(msg):
    # Handles geometry_msgs/Pose and raw array-like fallback
    try:
        return np.array([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ])
    except AttributeError:
        arr = np.array(msg)
        if arr.size == 7:
            return arr
        else:
            raise ValueError(f"Unexpected pose message format: {type(msg)} with size {arr.size}")


def extract_rosbag(bag_path, topics, hz=10.0):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Collect all available topics and types
    available = {t.name: t.type for t in reader.get_all_topics_and_types()}
    print("[INFO] Available topics:")
    for k, v in available.items():
        print(f"  {k}: {v}")

    # --- Load data ---
    data = {k: [] for k in topics.keys()}
    msg_types = {v: get_message(available[v]) for v in topics.values() if v in available}

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    while reader.has_next():
        topic, serialized, t = reader.read_next()
        if topic in topics.values():
            key = [k for k, v in topics.items() if v == topic][0]
            msg_type = msg_types[topic]
            msg = deserialize_message(serialized, msg_type)
            data[key].append((t * 1e-9, msg))

    for key, samples in data.items():
        print(f"[INFO] {key}: {len(samples)} msgs")

    # --- Determine global time range ---
    all_times = [t for arr in data.values() for (t, _) in arr if arr]
    t0, t1 = min(all_times), max(all_times)
    t_uniform = np.arange(t0, t1, 1.0 / hz)

    # --- Extract numeric arrays + interpolate ---
    obs = {}
    for key, samples in data.items():
        if not samples:
            continue
        ts = np.array([t for t, _ in samples])
        if key == "arm_joints":
            vals = np.array([msg.position for _, msg in samples])
        elif key in ["block_pose", "target_pose"]:
            vals = []
            for _, msg in samples:
                if hasattr(msg, "transform"):
                    vals.append([
                        msg.transform.translation.x,
                        msg.transform.translation.y,
                        msg.transform.translation.z,
                        msg.transform.rotation.x,
                        msg.transform.rotation.y,
                        msg.transform.rotation.z,
                        msg.transform.rotation.w,
                    ])
                elif hasattr(msg, "position"):
                    vals.append(pose_to_numpy(msg))
                else:
                    raise ValueError(f"Unexpected message type for {key}: {type(msg)}")
            vals = np.array(vals)
        elif key == "wrist_angle":
            vals = np.array([[msg.data] for _, msg in samples])
        elif key in ["gripper_state", "left_contact", "right_contact"]:
            vals = np.array([[float(msg.data)] for _, msg in samples])
        else:
            continue

        if key in ["gripper_state", "left_contact", "right_contact"]:
            # Discrete binary signal: nearest neighbor
            vals = (vals > 0.5).astype(float)
            interp = interp1d(
                ts, vals, axis=0, bounds_error=False,
                fill_value="extrapolate",  # Changed this
                kind="nearest"
            )
        else:
            interp = interp1d(
                ts, vals, axis=0, bounds_error=False,
                fill_value="extrapolate", kind="linear"
            )

        obs[key] = interp(t_uniform)

    # --- Compute actions (fixed global scaling instead of per-trajectory) ---
    eff = np.hstack([
        obs.get("target_pose", np.zeros((len(t_uniform), 7)))[:, :3],
        obs.get("wrist_angle", np.zeros((len(t_uniform), 1))),
        obs.get("gripper_state", np.zeros((len(t_uniform), 1))),
    ])
    actions = np.diff(eff, axis=0, prepend=eff[0:1])
    
    # Use FIXED scaling factors (matching bridge.py)
    # Position: 5cm = 0.05m per unit action
    # Wrist: 2° per unit action
    # Gripper: binary, no scaling needed
    position_scale = 0.05  # meters
    wrist_scale = 2.0      # degrees
    
    actions[:, 0:3] /= position_scale  # Normalize position deltas
    actions[:, 3] /= wrist_scale       # Normalize wrist deltas
    # actions[:, 4] is gripper (binary), keep as-is

    # No combined contact or merged reward — only left/right contact
    left_contact = obs.get("left_contact", np.zeros((len(t_uniform), 1)))
    right_contact = obs.get("right_contact", np.zeros((len(t_uniform), 1)))
    rewards = np.zeros_like(left_contact)  # dummy reward (you can replace later)

    return t_uniform, obs, actions, rewards


def save_dreamer_format(out_dir, obs, actions, rewards):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "obs.npz"), **obs)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    np.save(os.path.join(out_dir, "rewards.npy"), rewards)
    print(f"[INFO] Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", default="data/demo1")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    topics = {
        "arm_joints": "/joint_states",              # Vector<Angles [rads]>
        "block_pose": "/block_pose",                # Pose
        "target_pose": "/unity_target_pose",        # X, Y, Z position
        "wrist_angle": "/wrist_angle",              # Angle -180 < x < 180
        "gripper_state": "/gripper_command",        # Bool, open/close
        "left_contact": "/left_contact_detected",   # Bool, true/false
        "right_contact": "/right_contact_detected", # Bool, true/false
    }

    t, obs, actions, rewards = extract_rosbag(args.bag, topics, hz=args.hz)
    save_dreamer_format(args.out, obs, actions, rewards)


    print("[DONE]")