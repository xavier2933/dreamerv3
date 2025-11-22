import numpy as np
import os

# Check workspace bounds from demos
demo_dir = os.path.expanduser("~/dreamer/dreamerv3/log_data/online_training/replay")
files = sorted([f for f in os.listdir(demo_dir) if f.endswith('.npz')])

all_positions = []
all_wrists = []

print(f"Analyzing {len(files)} demo files...")

for f in files[:20]:  # Sample first 20 demos
    with np.load(os.path.join(demo_dir, f)) as data:
        target_pose = data['target_pose']
        wrist = data['wrist_angle']
        all_positions.append(target_pose[:, :3])
        all_wrists.append(wrist)

positions = np.vstack(all_positions)
wrists = np.concatenate(all_wrists)

print("\n" + "="*60)
print("WORKSPACE RANGES FROM DEMOS:")
print("="*60)
print(f"X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] (current bounds: [0.0, 0.6])")
print(f"Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] (current bounds: [-0.3, 0.3])")
print(f"Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] (current bounds: [0.0, 0.6])")
print(f"Wrist: [{wrists.min():.1f}°, {wrists.max():.1f}°] (current bounds: [-180, 180])")

print("\n" + "="*60)
print("RECOMMENDED BOUNDS (with 10% margin):")
print("="*60)
x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1
z_margin = (z_max - z_min) * 0.1

print(f"X: [{max(0, x_min - x_margin):.3f}, {x_max + x_margin:.3f}]")
print(f"Y: [{y_min - y_margin:.3f}, {y_max + y_margin:.3f}]")
print(f"Z: [{max(0, z_min - z_margin):.3f}, {z_max + z_margin:.3f}]")
