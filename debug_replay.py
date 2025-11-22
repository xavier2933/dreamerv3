#!/usr/bin/env python3
import argparse
import os
import time
import json
import zmq
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Replay recorded actions to the Dreamer bridge.")
    parser.add_argument("--data", required=True, help="Path to directory containing actions.npy and obs.npz")
    parser.add_argument("--hz", type=float, default=10.0, help="Control frequency")
    args = parser.parse_args()

    # 1. Load Data
    actions_path = os.path.join(args.data, "actions.npy")
    obs_path = os.path.join(args.data, "obs.npz")

    if not os.path.exists(actions_path):
        print(f"[ERROR] Could not find {actions_path}")
        return

    print(f"[INFO] Loading data from {args.data}...")
    actions = np.load(actions_path)
    print(f"[INFO] Loaded {len(actions)} actions.")

    # 2. Setup ZMQ
    # Bridge binds PUB to 5556 (Obs) -> We SUB to 5556
    # Bridge binds SUB to 5557 (Actions) -> We PUB to 5557
    print("[INFO] Connecting to Bridge via ZMQ...")
    ctx = zmq.Context()
    
    # Socket to receive observations (to sync/verify)
    sub_obs = ctx.socket(zmq.SUB)
    sub_obs.connect("tcp://127.0.0.1:5556")
    sub_obs.setsockopt_string(zmq.SUBSCRIBE, "")

    # Socket to send actions
    pub_act = ctx.socket(zmq.PUB)
    pub_act.connect("tcp://127.0.0.1:5557")

    print("[INFO] Waiting for first observation from bridge to sync...")
    # Wait for a message to ensure bridge is running and we are connected
    msg = sub_obs.recv_string()
    print("[INFO] Bridge detected! Starting replay in 1 second...")
    time.sleep(1.0)

    # 3. Replay Loop
    dt = 1.0 / args.hz
    
    # Reset the environment first?
    # The bridge listens for "reset": true
    print("[INFO] Sending RESET command...")
    pub_act.send_string(json.dumps({"reset": True}))
    time.sleep(1.0) # Give it time to reset

    print(f"[INFO] Replaying {len(actions)} actions...")
    
    for i, action in enumerate(actions):
        start_time = time.time()

        # Action format expected by bridge: [dx, dy, dz, wrist, grip]
        # actions.npy from cleaner.py is: [dx, dy, dz, wrist, grip]
        # Ensure it's a list for JSON serialization
        action_list = action.tolist()
        
        # Send action
        payload = {"action": action_list}
        pub_act.send_string(json.dumps(payload))

        # Optional: Read observation to verify (non-blocking)
        try:
            while True:
                obs_msg = sub_obs.recv_string(flags=zmq.NOBLOCK)
                # Just drain the queue
        except zmq.Again:
            pass

        # Timing
        elapsed = time.time() - start_time
        sleep_time = max(0.0, dt - elapsed)
        
        if i % 10 == 0:
            print(f"Step {i}/{len(actions)}: Sent {action_list}")
            
        time.sleep(sleep_time)

    print("[INFO] Replay complete.")
    
    # Send zero action to stop
    pub_act.send_string(json.dumps({"action": [0.0, 0.0, 0.0, 0.0, 0.0]}))

if __name__ == "__main__":
    main()
