#!/usr/bin/env python3
import os
import argparse
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.interpolate import interp1d
from tqdm import tqdm


"""
TODO FIX GRIPPER COMMAND PLOT IS NOT RIGHT
"""

def read_topic(reader, topic_name, msg_type):
    msgs = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, msg_type)
            msgs.append((t * 1e-9, msg))  # sec
    return msgs

def pose_to_numpy(msg):
    # Handles geometry_msgs/Pose and raw array-like fallback
    try:
        # Normal case (geometry_msgs/Pose)
        return np.array([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ])
    except AttributeError:
        # Fallback if msg is an array or list of floats
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
    for k,v in available.items(): print(f"  {k}: {v}")

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
        if not samples: continue
        ts = np.array([t for t,_ in samples])
        if key == "arm_joints":
            vals = np.array([msg.position for _, msg in samples])
        elif key in ["block_pose", "target_pose"]:
            vals = []
            for _, msg in samples:
                if hasattr(msg, "transform"):
                    # TransformStamped type
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
                    # Pose type
                    vals.append(pose_to_numpy(msg))
                else:
                    raise ValueError(f"Unexpected message type for {key}: {type(msg)}")
            vals = np.array(vals)
        elif key == "wrist_angle":
            vals = np.array([[msg.data] for _, msg in samples])
        elif key in ["gripper_state", "contact"]:
            vals = np.array([[float(msg.data)] for _, msg in samples])
        else:
            continue

        interp = interp1d(ts, vals, axis=0, bounds_error=False,
                          fill_value="extrapolate", kind="linear")
        obs[key] = interp(t_uniform)

    # --- Compute actions ---
    eff = np.hstack([
        obs.get("target_pose", np.zeros((len(t_uniform), 7)))[:, :3],
        obs.get("wrist_angle", np.zeros((len(t_uniform), 1))),
        obs.get("gripper_state", np.zeros((len(t_uniform), 1))),
    ])
    actions = np.diff(eff, axis=0, prepend=eff[0:1])
    max_abs = np.abs(actions).max(axis=0)
    max_abs[max_abs == 0] = 1.0
    actions /= max_abs

    rewards = obs.get("contact", np.zeros((len(t_uniform), 1)))

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
        "arm_joints": "/joint_states",
        "block_pose": "/block_pose",  # Pose type
        "target_pose": "/unity_target_pose",  # FIXED name
        "wrist_angle": "/wrist_angle",
        "gripper_state": "/gripper_command",
        "contact": "/contact_detected",
    }


    t, obs, actions, rewards = extract_rosbag(args.bag, topics, hz=args.hz)
    save_dreamer_format(args.out, obs, actions, rewards)
    print("[DONE]")
