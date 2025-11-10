#!/usr/bin/env python3
import os
import subprocess

def run(cmd):
    print(f"\n[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}")
    else:
        print(f"✅ Done: {' '.join(cmd)}")

def main():
    bags_dir = "bags/success"
    output_root = "demos/success"

    # Ensure output directory exists
    os.makedirs(output_root, exist_ok=True)

    # Get all bag directories
    bag_dirs = sorted(
        [os.path.join(bags_dir, d) for d in os.listdir(bags_dir)
         if os.path.isdir(os.path.join(bags_dir, d))]
    )

    if not bag_dirs:
        print(f"No bag directories found in {bags_dir}/")
        return

    print(f"Found {len(bag_dirs)} bag(s):")
    for b in bag_dirs:
        print(" -", b)

    for bag_dir in bag_dirs:
        bag_name = os.path.basename(bag_dir.rstrip("/"))
        demo_out = os.path.join(output_root, f"demo_{bag_name}")

        # 1️⃣ Run cleaner.py
        run(["python3", "cleaner.py", "--bag", bag_dir, "--out", demo_out])

        # 2️⃣ Run analyze.py
        run(["python3", "analyze.py", "--data", demo_out])

    print("\n🎉 All bags processed successfully.")

if __name__ == "__main__":
    main()
