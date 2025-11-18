import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

# --- Configuration ---
# 1. CHANGE THIS TO YOUR ACTUAL CSV FILE PATH
CSV_FILE_PATH = 'logdata/actions.csv' 
# ---------------------

def load_data(file_path):
    """Loads the data from the CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print("Generating mock data for demonstration purposes.")
        return generate_mock_data()
    
    print(f"Loading data from: {file_path}...")
    try:
        # Assuming the CSV is well-formed. 
        # Using the first time column ('time_sec') as the primary time axis.
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None

def generate_mock_data():
    """Generates mock data matching the column structure for demonstration."""
    print("--- MOCK DATA GENERATION ---")
    N = 1000 # Number of data points for the mock data
    t = np.linspace(0, 10, N)
    
    data = {
        'time_sec': t,
        'time_ns': t * 1e9,
        
        # Joint Angles (simulated sinusoidal motion)
        **{f'joint_{i}': np.sin(t * (i/2 + 1)) * (0.1 * i + 0.5) for i in range(1, 8)},
        
        # Block Cartesian Pose (x, y, z)
        'block_x': 0.5 + 0.05 * np.cos(t),
        'block_y': -0.1 + 0.03 * np.sin(t * 2),
        'block_z': 0.2 + 0.01 * t,
        
        # Block Orientation (Quaternion) - smooth rotation simulation
        'block_qx': np.sin(t * 0.1),
        'block_qy': np.cos(t * 0.1),
        'block_qz': np.sin(t * 0.05),
        'block_qw': np.sqrt(1 - (np.sin(t * 0.1)**2 + np.cos(t * 0.1)**2 + np.sin(t * 0.05)**2)) / np.sqrt(3), # simplified normalization
        
        # Target Cartesian Pose (x, y, z) - constant or simple movement
        'target_x': 0.5 * np.ones_like(t),
        'target_y': -0.1 + 0.05 * t/10,
        'target_z': 0.25 * np.ones_like(t),

        # Target Orientation (Quaternion) - constant (Identity Quaternion)
        'target_qx': np.zeros_like(t),
        'target_qy': np.zeros_like(t),
        'target_qz': np.zeros_like(t),
        'target_qw': np.ones_like(t),
        
        # End-Effector State
        'wrist_angle': 0.1 * np.sin(t * 5) + 0.2,
        'gripper_state': np.where(t % 4 < 2, 1, 0), # Open/Close simulation
        'left_contact': np.where((t > 3) & (t < 5), 1, 0),
        'right_contact': np.where((t > 4) & (t < 6), 1, 0),
    }
    
    # Simple normalization for demonstration (ensure qw remains reasonable)
    data['block_qw'] = np.clip(data['block_qw'], 0.1, 1)

    return pd.DataFrame(data)

def plot_data(df):
    """Generates the set of six logical plots."""
    if df is None:
        return
    
    time_col = 'time_sec'

    # --- Plotting Groups ---
    
    # 1. Joint Angles (7 DOF)
    joint_cols = [f'joint_{i}' for i in range(1, 8)]
    plt.figure(figsize=(12, 6))
    for col in joint_cols:
        plt.plot(df[time_col], df[col], label=col, linewidth=1.5)
    plt.title('1. Robot Joint Positions Over Time (q1 to q7)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angle (rad or deg)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    # 2. Block Cartesian Position (x, y, z)
    block_pos_cols = ['block_x', 'block_y', 'block_z']
    plt.figure(figsize=(12, 6))
    for col in block_pos_cols:
        plt.plot(df[time_col], df[col], label=col.split('_')[-1], linewidth=1.5)
    plt.title('2. Block Cartesian Position (P)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # 3. Block Orientation (Quaternion)
    block_quat_cols = ['block_qx', 'block_qy', 'block_qz', 'block_qw']
    plt.figure(figsize=(12, 6))
    for col in block_quat_cols:
        plt.plot(df[time_col], df[col], label=col.split('_')[-1], linewidth=1.5)
    plt.title('3. Block Orientation (Quaternion Q)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Quaternion Component Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # 4. Target Cartesian Position (x, y, z)
    target_pos_cols = ['target_x', 'target_y', 'target_z']
    plt.figure(figsize=(12, 6))
    for col in target_pos_cols:
        plt.plot(df[time_col], df[col], label=col.split('_')[-1], linewidth=1.5)
    plt.title('4. Target Cartesian Position (P)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    # 5. Target Orientation (Quaternion)
    target_quat_cols = ['target_qx', 'target_qy', 'target_qz', 'target_qw']
    plt.figure(figsize=(12, 6))
    for col in target_quat_cols:
        plt.plot(df[time_col], df[col], label=col.split('_')[-1], linewidth=1.5)
    plt.title('5. Target Orientation (Quaternion Q)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Quaternion Component Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    # 6. End-Effector State (Wrist Angle, Gripper, Contacts)
    ee_state_cols = ['wrist_angle', 'gripper_state', 'left_contact', 'right_contact']
    plt.figure(figsize=(12, 6))
    
    # Plot continuous variable on the left axis
    ax1 = plt.gca()
    ax1.plot(df[time_col], df['wrist_angle'], label='Wrist Angle', color='blue', linewidth=2)
    ax1.set_ylabel('Wrist Angle (rad)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot binary/discrete variables on the right axis
    ax2 = ax1.twinx()
    ax2.plot(df[time_col], df['gripper_state'], label='Gripper State (1=Closed)', color='red', linestyle=':', linewidth=1)
    ax2.plot(df[time_col], df['left_contact'], label='Left Contact (1=Active)', color='green', linestyle='--', linewidth=1)
    ax2.plot(df[time_col], df['right_contact'], label='Right Contact (1=Active)', color='orange', linestyle='--', linewidth=1)
    ax2.set_ylabel('Gripper/Contact State (Binary)', color='black')
    
    # Set y-limits and ticks for the discrete axis to be clean
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    
    plt.title('6. End-Effector State and Contacts')
    ax1.set_xlabel('Time (seconds)')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show all plots
    plt.show()

if __name__ == '__main__':
    data_frame = load_data(CSV_FILE_PATH)
    if data_frame is not None:
        plot_data(data_frame)
    else:
        print("Could not process data. Please ensure the CSV file path is correct.")