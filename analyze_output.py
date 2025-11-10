import matplotlib.pyplot as plt
import numpy as np

# Load data
gt = np.load("/home/xavie/dreamer/dreamerv3/data/demos/success/demo_rosbag2_2025_11_09-15_33_27/actions.npy")
pred = np.load("/home/xavie/dreamer/dreamerv3/data/demos/success/demo_rosbag2_2025_11_09-15_33_27/predicted_actions.npy")
T = min(len(gt), len(pred))

num_dims = gt.shape[1]
cols = 2
rows = int(np.ceil(num_dims / cols))

fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), sharex=True)
axes = axes.flatten()

for i in range(num_dims):
    ax = axes[i]
    ax.plot(gt[:T, i], label='Ground Truth', linewidth=2)
    ax.plot(pred[:T, i], label='Predicted', linestyle='--')
    ax.set_title(f'Action dim {i}')
    ax.set_ylabel('Value')
    ax.grid(True)

# Hide any unused subplots
for j in range(num_dims, len(axes)):
    fig.delaxes(axes[j])

axes[-1].set_xlabel('Timestep')
fig.suptitle("DreamerV3 Predicted vs Ground Truth Actions", fontsize=14)
fig.legend(loc='upper right')
plt.tight_layout()
plt.show()
