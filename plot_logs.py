import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_logs(log_dir):
    obs_file = os.path.join(log_dir, "observations.csv")
    act_file = os.path.join(log_dir, "actions.csv")
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(obs_file):
        print(f"No observations file found at {obs_file}")
        return

    print(f"Loading {obs_file}...")
    obs_df = pd.read_csv(obs_file)
    
    # Create time axis relative to start
    if not obs_df.empty:
        start_time = obs_df["time_sec"].iloc[0] + obs_df["time_ns"].iloc[0] * 1e-9
        obs_df["time"] = (obs_df["time_sec"] + obs_df["time_ns"] * 1e-9) - start_time
    
    # Plot 1: Target vs Block (Z-axis)
    plt.figure(figsize=(10, 6))
    plt.plot(obs_df["time"], obs_df["block_z"], label="Block Z")
    plt.plot(obs_df["time"], obs_df["target_z"], label="Target Z", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Block vs Target Z Position")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "z_position.png"))
    plt.close()

    # Plot 2: Joint Angles
    plt.figure(figsize=(12, 8))
    joint_cols = [c for c in obs_df.columns if "joint_" in c]
    for col in joint_cols:
        plt.plot(obs_df["time"], obs_df[col], label=col)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Joint Angles")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "joints.png"))
    plt.close()

    # Plot 3: Actions (if available)
    if os.path.exists(act_file):
        print(f"Loading {act_file}...")
        act_df = pd.read_csv(act_file)
        if not act_df.empty:
            # Align time
            act_df["time"] = (act_df["time_sec"] + act_df["time_ns"] * 1e-9) - start_time
            
            plt.figure(figsize=(10, 6))
            plt.plot(act_df["time"], act_df["dx"], label="dx")
            plt.plot(act_df["time"], act_df["dy"], label="dy")
            plt.plot(act_df["time"], act_df["dz"], label="dz")
            plt.xlabel("Time (s)")
            plt.ylabel("Action Value")
            plt.title("Actions over Time")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "actions.png"))
            plt.close()

    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="log_data", help="Directory containing csv logs")
    args = parser.parse_args()
    plot_logs(args.logdir)
