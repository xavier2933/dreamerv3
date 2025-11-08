from embodied.envs.arm import Arm

# Initialize your environment (point to your offline data directory)
env = Arm(task='debug', data_dir='/home/xavie/dreamer/dreamerv3/data/demos/')

print("\n[DEBUG] --- Episode boundaries ---")
for i, (s, e) in enumerate(zip(env.episode_starts, env.episode_ends)):
    print(f"Episode {i}: start={s}, end={e}, length={e - s}")
print()

# Step through a few steps manually
print("[DEBUG] --- Stepping through environment ---")
obs = env._format_obs(env.t, is_first=True)
print(f"Initial t={env.t}, done={env.done}")

for i in range(50):  # or len(env.actions) if you want full length
    obs = env.step({"reset": False, "action": env.actions[env.t]})
    print(f"Step {i:3d} | t={env.t:4d} | done={env.done} | reward={obs['reward']:.2f}")

    if env.done:
        print("[DEBUG] Episode ended.")
        break
