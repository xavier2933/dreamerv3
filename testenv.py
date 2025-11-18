#!/usr/bin/env python3
"""
Diagnostic script to identify issues with DreamerV3 training data and inference setup.
Checks:
1. Action normalization scales
2. Position distribution (X/Y/Z movement patterns)
3. Delta magnitudes
4. Workspace bounds
"""
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_demo(demo_path):
    """Analyze a single demo directory."""
    demo_path = Path(demo_path)
    
    # Load data
    obs = np.load(demo_path / "obs.npz")
    actions = np.load(demo_path / "actions.npy")
    
    # Extract relevant observations
    target_pose = obs["target_pose"][:, :3]  # x, y, z
    wrist = obs["wrist_angle"]
    gripper = obs["gripper_state"]
    
    # Compute raw deltas from observations
    eff = np.hstack([target_pose, wrist, gripper])
    raw_deltas = np.diff(eff, axis=0, prepend=eff[0:1])
    
    return {
        'target_pose': target_pose,
        'wrist': wrist,
        'gripper': gripper,
        'raw_deltas': raw_deltas,
        'stored_actions': actions,
        'eff': eff
    }


def diagnose_all_demos(data_dir):
    """Run comprehensive diagnostics on all demos."""
    data_dir = Path(data_dir)
    
    # Find all demo directories
    demo_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "obs.npz").exists()])
    
    if not demo_dirs:
        print(f"[ERROR] No demo directories found in {data_dir}")
        print("Looking for directories containing 'obs.npz' files")
        return
    
    print(f"\n{'='*80}")
    print(f"DREAMERV3 TRAINING DATA DIAGNOSTIC")
    print(f"{'='*80}")
    print(f"\nFound {len(demo_dirs)} demos in {data_dir}")
    
    # Collect all data
    all_raw_deltas = []
    all_stored_actions = []
    all_positions = []
    all_wrist = []
    all_gripper = []
    
    for demo_dir in demo_dirs:
        data = analyze_demo(demo_dir)
        all_raw_deltas.append(data['raw_deltas'])
        all_stored_actions.append(data['stored_actions'])
        all_positions.append(data['target_pose'])
        all_wrist.append(data['wrist'])
        all_gripper.append(data['gripper'])
    
    all_raw_deltas = np.concatenate(all_raw_deltas, axis=0)
    all_stored_actions = np.concatenate(all_stored_actions, axis=0)
    all_positions = np.concatenate(all_positions, axis=0)
    all_wrist = np.concatenate(all_wrist, axis=0)
    all_gripper = np.concatenate(all_gripper, axis=0)
    
    # === DIAGNOSTIC 1: Position Distribution ===
    print(f"\n{'='*80}")
    print("1. POSITION DISTRIBUTION (Workspace Analysis)")
    print(f"{'='*80}")
    
    print(f"\nX-axis (lateral):")
    print(f"  Range: [{all_positions[:, 0].min():.3f}, {all_positions[:, 0].max():.3f}]")
    print(f"  Span:  {all_positions[:, 0].ptp():.3f} m")
    print(f"  Mean:  {all_positions[:, 0].mean():.3f} ± {all_positions[:, 0].std():.3f} m")
    
    print(f"\nY-axis (lateral):")
    print(f"  Range: [{all_positions[:, 1].min():.3f}, {all_positions[:, 1].max():.3f}]")
    print(f"  Span:  {all_positions[:, 1].ptp():.3f} m")
    print(f"  Mean:  {all_positions[:, 1].mean():.3f} ± {all_positions[:, 1].std():.3f} m")
    
    print(f"\nZ-axis (vertical):")
    print(f"  Range: [{all_positions[:, 2].min():.3f}, {all_positions[:, 2].max():.3f}]")
    print(f"  Span:  {all_positions[:, 2].ptp():.3f} m")
    print(f"  Mean:  {all_positions[:, 2].mean():.3f} ± {all_positions[:, 2].std():.3f} m")
    
    # Check if Z-dominated
    x_span = all_positions[:, 0].ptp()
    y_span = all_positions[:, 1].ptp()
    z_span = all_positions[:, 2].ptp()
    
    if z_span > max(x_span, y_span) * 1.5:
        print(f"\n✓ Movement is Z-dominated (as expected)")
    else:
        print(f"\n⚠ WARNING: Movement is NOT Z-dominated!")
        print(f"  X/Y movements are comparable to Z movement.")
    
    # === DIAGNOSTIC 2: Raw Delta Analysis ===
    print(f"\n{'='*80}")
    print("2. RAW DELTA ANALYSIS (Before Normalization)")
    print(f"{'='*80}")
    
    print(f"\nRaw delta statistics [x, y, z, wrist, gripper]:")
    print(f"  Max abs:  {np.abs(all_raw_deltas).max(axis=0)}")
    print(f"  Mean abs: {np.abs(all_raw_deltas).mean(axis=0)}")
    print(f"  Std:      {all_raw_deltas.std(axis=0)}")
    
    # Check for zero movement dimensions
    for i, name in enumerate(['X', 'Y', 'Z', 'Wrist', 'Gripper']):
        if np.abs(all_raw_deltas[:, i]).max() < 1e-6:
            print(f"  ⚠ WARNING: {name} has essentially zero movement!")
    
    # === DIAGNOSTIC 3: Stored Action Normalization ===
    print(f"\n{'='*80}")
    print("3. STORED ACTION NORMALIZATION")
    print(f"{'='*80}")
    
    # Compute the normalization that was applied
    max_abs_deltas = np.abs(all_raw_deltas).max(axis=0)
    max_abs_deltas[max_abs_deltas == 0] = 1.0
    
    print(f"\nNormalization factors used in training:")
    print(f"  {max_abs_deltas}")
    
    print(f"\nStored action statistics (after normalization):")
    print(f"  Max abs:  {np.abs(all_stored_actions).max(axis=0)}")
    print(f"  Mean abs: {np.abs(all_stored_actions).mean(axis=0)}")
    print(f"  Std:      {all_stored_actions.std(axis=0)}")
    
    # Check if actions are properly normalized to [-1, 1]
    if np.abs(all_stored_actions).max() > 1.0 + 1e-6:
        print(f"  ⚠ WARNING: Actions exceed [-1, 1] range! Max: {np.abs(all_stored_actions).max()}")
    else:
        print(f"  ✓ Actions are properly normalized to [-1, 1]")
    
    # === DIAGNOSTIC 4: Action Scale for Inference ===
    print(f"\n{'='*80}")
    print("4. RECOMMENDED INFERENCE SETTINGS")
    print(f"{'='*80}")
    
    print(f"\n🔧 Copy this into your inference script:")
    print(f"self.action_scale = np.array([")
    for i, val in enumerate(max_abs_deltas):
        name = ['x', 'y', 'z', 'wrist', 'gripper'][i]
        print(f"    {val:.6f},  # {name}")
    print(f"])")
    
    # === DIAGNOSTIC 5: Movement Distribution ===
    print(f"\n{'='*80}")
    print("5. MOVEMENT PATTERN ANALYSIS")
    print(f"{'='*80}")
    
    # Compute per-axis movement magnitude
    movement_magnitude = np.abs(all_raw_deltas[:, :3]).sum(axis=0)
    total_movement = movement_magnitude.sum()
    
    print(f"\nTotal movement per axis:")
    print(f"  X: {movement_magnitude[0]:.3f} m ({100*movement_magnitude[0]/total_movement:.1f}%)")
    print(f"  Y: {movement_magnitude[1]:.3f} m ({100*movement_magnitude[1]/total_movement:.1f}%)")
    print(f"  Z: {movement_magnitude[2]:.3f} m ({100*movement_magnitude[2]/total_movement:.1f}%)")
    
    z_percentage = 100 * movement_magnitude[2] / total_movement
    if z_percentage < 50:
        print(f"\n⚠ WARNING: Only {z_percentage:.1f}% of movement is in Z!")
        print(f"  Your training data may not be 'up and down' dominated as expected.")
    
    # === DIAGNOSTIC 6: Workspace Bounds ===
    print(f"\n{'='*80}")
    print("6. RECOMMENDED WORKSPACE BOUNDS")
    print(f"{'='*80}")
    
    # Add 20% margin to observed bounds
    margin = 0.2
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    z_margin = (z_max - z_min) * margin
    
    print(f"\n🔧 Copy this into your inference script:")
    print(f"self.pos_min = np.array([{x_min - x_margin:.3f}, {y_min - y_margin:.3f}, {z_min - z_margin:.3f}])")
    print(f"self.pos_max = np.array([{x_max + x_margin:.3f}, {y_max + y_margin:.3f}, {z_max + z_margin:.3f}])")
    
    wrist_min, wrist_max = all_wrist.min(), all_wrist.max()
    print(f"self.wrist_min = {wrist_min - 10:.1f}")
    print(f"self.wrist_max = {wrist_max + 10:.1f}")
    
    # === DIAGNOSTIC 7: Visualization ===
    print(f"\n{'='*80}")
    print("7. GENERATING VISUALIZATION")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Data Analysis', fontsize=16)
    
    # Position distributions
    axes[0, 0].hist(all_positions[:, 0], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('X Position Distribution')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(all_positions[:, 1], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Y Position Distribution')
    axes[0, 1].set_xlabel('Y (m)')
    
    axes[0, 2].hist(all_positions[:, 2], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Z Position Distribution')
    axes[0, 2].set_xlabel('Z (m)')
    
    # Delta distributions
    axes[1, 0].hist(all_raw_deltas[:, 0], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('X Delta Distribution')
    axes[1, 0].set_xlabel('ΔX (m)')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist(all_raw_deltas[:, 1], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Y Delta Distribution')
    axes[1, 1].set_xlabel('ΔY (m)')
    
    axes[1, 2].hist(all_raw_deltas[:, 2], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Z Delta Distribution')
    axes[1, 2].set_xlabel('ΔZ (m)')
    
    plt.tight_layout()
    output_path = Path(data_dir) / "diagnostic_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    
    # === DIAGNOSTIC 8: Per-Demo Analysis ===
    print(f"\n{'='*80}")
    print("8. PER-DEMO ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n{'Demo':<20} {'Length':>8} {'X span':>10} {'Y span':>10} {'Z span':>10}")
    print(f"{'-'*60}")
    
    for demo_dir in demo_dirs:
        data = analyze_demo(demo_dir)
        demo_name = demo_dir.name
        length = len(data['target_pose'])
        x_span = data['target_pose'][:, 0].ptp()
        y_span = data['target_pose'][:, 1].ptp()
        z_span = data['target_pose'][:, 2].ptp()
        
        print(f"{demo_name:<20} {length:>8} {x_span:>10.3f} {y_span:>10.3f} {z_span:>10.3f}")
    
    # === FINAL SUMMARY ===
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    issues = []
    
    # Check for Z-dominated movement
    if z_span <= max(x_span, y_span) * 1.5:
        issues.append("❌ Training data is NOT Z-dominated (expected for up/down tasks)")
    
    # Check for proper normalization
    if np.abs(all_stored_actions).max() > 1.1:
        issues.append("❌ Actions are not properly normalized to [-1, 1]")
    
    # Check for zero movement
    for i, name in enumerate(['X', 'Y', 'Z', 'Wrist', 'Gripper']):
        if np.abs(all_raw_deltas[:, i]).max() < 1e-6:
            issues.append(f"❌ No movement in {name} axis")
    
    # Check workspace bounds
    current_bounds = np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0])
    if (all_positions.min(axis=0) < current_bounds[0]).any() or (all_positions.max(axis=0) > current_bounds[1]).any():
        issues.append("⚠ Some training positions are outside current workspace bounds")
    
    if issues:
        print(f"\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print(f"\nRecommendations:")
        print(f"  1. Update action_scale in inference script (see section 4)")
        print(f"  2. Update workspace bounds (see section 6)")
        print(f"  3. Verify training data collection process if Z-movement is wrong")
    else:
        print(f"\n✓ No major issues found!")
        print(f"  Make sure to use the action_scale from section 4 in your inference script.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose DreamerV3 training data issues")
    parser.add_argument("--data_dir", required=True, help="Path to directory containing demo folders")
    args = parser.parse_args()
    
    diagnose_all_demos(args.data_dir)