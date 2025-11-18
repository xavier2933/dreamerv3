#!/usr/bin/env python3
import json
import zmq
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Bool
import math
import csv
import os
from rclpy.time import Time # Import for getting ROS time

# --- Helper Functions (unchanged) ---

def pose_to_array(msg: Pose):
    """Convert Pose to array [x, y, z, qx, qy, qz, qw]"""
    p = msg.position
    o = msg.orientation
    return [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

def quaternion_from_euler_z(yaw):
    """Convert yaw (z-axis rotation) to quaternion."""
    half_yaw = yaw / 2.0
    return np.array([
        0.0,
        0.0,
        math.sin(half_yaw),
        math.cos(half_yaw)
    ])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

# --- CSV Logger Class ---

class DreamerRosBridge(Node):
    def __init__(self):
        super().__init__("dreamer_ros_bridge")

        # === CSV LOGGING SETUP ===
        self.log_dir = os.path.join(os.getcwd(), "log_data")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.obs_file_name = os.path.join(self.log_dir, "observations.csv")
        self.action_file_name = os.path.join(self.log_dir, "actions.csv")
        
        # Observation CSV: Timestamps and values for all observations
        self.obs_csv_file = open(self.obs_file_name, 'w', newline='')
        self.obs_csv_writer = csv.writer(self.obs_csv_file)
        # 7 joints + block_pose (7) + target_pose (7) + wrist (1) + gripper (1) + 2 contacts (2) = 25
        obs_header = ["time_sec", "time_ns", 
                      "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", 
                      "block_x", "block_y", "block_z", "block_qx", "block_qy", "block_qz", "block_qw",
                      "target_x", "target_y", "target_z", "target_qx", "target_qy", "target_qz", "target_qw",
                      "wrist_angle", "gripper_state", "left_contact", "right_contact"]
        self.obs_csv_writer.writerow(obs_header)
        
        # Action CSV: Timestamps and values for all actions
        self.act_csv_file = open(self.action_file_name, 'w', newline='')
        self.act_csv_writer = csv.writer(self.act_csv_file)
        act_header = ["time_sec", "time_ns", "dx", "dy", "dz", "wrist", "grip"]
        self.act_csv_writer.writerow(act_header)
        
        # Dictionary to hold the latest *logged* observation values for the next log entry
        self.data_cache = {
            "arm_joints": [np.nan] * 7, # Assuming 7 joints
            "block_pose": [np.nan] * 7,
            "target_pose": [np.nan] * 7,
            "wrist_angle": [np.nan],
            "gripper_state": [np.nan],
            "left_contact": [np.nan],
            "right_contact": [np.nan],
            "action": [np.nan] * 5,
        }
        
        self.get_logger().info(f"Log files created in: {self.log_dir}")
        
        # === ROS Subscriptions (collecting observations) ===
        self.create_subscription(JointState, "/joint_states", self.cb_joint_states, 10)
        self.create_subscription(Pose, "/block_pose", self.cb_block_pose, 10)
        self.create_subscription(Pose, "/unity_target_pose", self.cb_target_pose, 10)
        self.create_subscription(Float32, "/wrist_angle", self.cb_wrist_angle, 10)
        self.create_subscription(Bool, "/gripper_command", self.cb_gripper_state, 10)
        self.create_subscription(Bool, "/left_contact_detected", self.cb_left_contact, 10)
        self.create_subscription(Bool, "/right_contact_detected", self.cb_right_contact, 10)

        # === ROS Publishers (sending Dreamer's actions) ===
        self.pub_target = self.create_publisher(Pose, "/bc_target_pose", 10)
        self.pub_gripper = self.create_publisher(Bool, "/gripper_cmd_aut", 10)

        # Cache for sending complete obs set to Dreamer via ZeroMQ (original purpose)
        self.obs_cache = {} 

        # === ZeroMQ sockets (unchanged) ===
        ctx = zmq.Context()
        self.pub = ctx.socket(zmq.PUB)
        self.pub.bind("tcp://127.0.0.1:5556")
        self.sub = ctx.socket(zmq.SUB)
        self.sub.bind("tcp://127.0.0.1:5557")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Debug tracking (unchanged)
        self.received_topics = set()
        
        self.get_logger().info("🤖 Dreamer ROS bridge ready.")

    def __del__(self):
        """Ensure file handles are closed when the node is destroyed."""
        if hasattr(self, 'obs_csv_file') and not self.obs_csv_file.closed:
            self.obs_csv_file.close()
            self.get_logger().info("Observation log closed.")
        if hasattr(self, 'act_csv_file') and not self.act_csv_file.closed:
            self.act_csv_file.close()
            self.get_logger().info("Action log closed.")

    def log_observation(self, current_time: Time):
        """Logs the latest state of all observations to the CSV."""
        # Note: You can also log only when a specific topic updates, 
        # but logging all fields ensures a comprehensive timeline.
        timestamp = [current_time.nanoseconds // 10**9, current_time.nanoseconds % 10**9]
        
        row = timestamp + \
              self.data_cache["arm_joints"] + \
              self.data_cache["block_pose"] + \
              self.data_cache["target_pose"] + \
              self.data_cache["wrist_angle"] + \
              self.data_cache["gripper_state"] + \
              self.data_cache["left_contact"] + \
              self.data_cache["right_contact"]
              
        self.obs_csv_writer.writerow(row)
        self.obs_csv_file.flush() # Ensure data is written to disk

    # === Observation Callbacks (Modified to log data) ===
    def cb_joint_states(self, msg: JointState):
        current_time = self.get_clock().now()
        data = list(msg.position)
        self.obs_cache["arm_joints"] = data # For ZeroMQ
        self.data_cache["arm_joints"] = data # For CSV
        self.log_observation(current_time)
        if "arm_joints" not in self.received_topics:
            self.received_topics.add("arm_joints")
            self.get_logger().info("✓ Receiving arm_joints")
    
    def cb_block_pose(self, msg: Pose):
        current_time = self.get_clock().now()
        data = pose_to_array(msg)
        self.obs_cache["block_pose"] = data # For ZeroMQ
        self.data_cache["block_pose"] = data # For CSV
        self.log_observation(current_time)
        if "block_pose" not in self.received_topics:
            self.received_topics.add("block_pose")
            self.get_logger().info("✓ Receiving block_pose")
    
    def cb_target_pose(self, msg: Pose):
        current_time = self.get_clock().now()
        data = pose_to_array(msg)
        self.obs_cache["target_pose"] = data # For ZeroMQ
        self.data_cache["target_pose"] = data # For CSV
        self.log_observation(current_time)
        if "target_pose" not in self.received_topics:
            self.received_topics.add("target_pose")
            self.get_logger().info("✓ Receiving target_pose")
    
    def cb_wrist_angle(self, msg: Float32):
        current_time = self.get_clock().now()
        data = [msg.data]
        self.obs_cache["wrist_angle"] = data # For ZeroMQ
        self.data_cache["wrist_angle"] = data # For CSV
        self.log_observation(current_time)
        if "wrist_angle" not in self.received_topics:
            self.received_topics.add("wrist_angle")
            self.get_logger().info("✓ Receiving wrist_angle")
    
    def cb_gripper_state(self, msg: Bool):
        current_time = self.get_clock().now()
        data = [float(msg.data)]
        self.obs_cache["gripper_state"] = data # For ZeroMQ
        self.data_cache["gripper_state"] = data # For CSV
        self.log_observation(current_time)
        if "gripper_state" not in self.received_topics:
            self.received_topics.add("gripper_state")
            self.get_logger().info("✓ Receiving gripper_state")
    
    def cb_left_contact(self, msg: Bool):
        current_time = self.get_clock().now()
        data = [float(msg.data)]
        self.obs_cache["left_contact"] = data # For ZeroMQ
        self.data_cache["left_contact"] = data # For CSV
        self.log_observation(current_time)
        if "left_contact" not in self.received_topics:
            self.received_topics.add("left_contact")
            self.get_logger().info("✓ Receiving left_contact")
    
    def cb_right_contact(self, msg: Bool):
        current_time = self.get_clock().now()
        data = [float(msg.data)]
        self.obs_cache["right_contact"] = data # For ZeroMQ
        self.data_cache["right_contact"] = data # For CSV
        self.log_observation(current_time)
        if "right_contact" not in self.received_topics:
            self.received_topics.add("right_contact")
            self.get_logger().info("✓ Receiving right_contact")

    # --- Timer and Action Publishing (Modified) ---
    
    def timer_callback(self):
        # Send observations to Dreamer process (unchanged)
        if self.obs_cache:
            msg = json.dumps(self.obs_cache)
            self.pub.send_string(msg)
            
            required = ["arm_joints", "block_pose", "target_pose", "wrist_angle", 
                        "gripper_state", "left_contact", "right_contact"]
            if all(k in self.obs_cache for k in required) and "_logged_complete" not in self.received_topics:
                self.get_logger().info("✓ Sending complete observation set to Dreamer")
                self.received_topics.add("_logged_complete")

        # Receive action updates from Dreamer if available (unchanged logic)
        try:
            while self.sub.poll(timeout=0):
                msg_str = self.sub.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(msg_str)
                action = np.array(data["action"], dtype=np.float32)
                self.get_logger().info(f"📥 Received action from Dreamer: {action}")
                self.publish_actions(action)
        except zmq.Again:
            pass

    def publish_actions(self, act):
        """
        Publish Dreamer's actions to ROS and log them to CSV.
        Actions are [dx, dy, dz, wrist, grip]
        """
        # --- ROS Publishing Logic (unchanged) ---
        dx, dy, dz, wrist, grip = act[:5]

        # Construct orientation from wrist angle
        base_down_q = np.array([0.7071068, -0.7071068, 0.0, 0.0])
        q = quaternion_from_euler_z(math.radians(wrist))
        final_q = quaternion_multiply(q, base_down_q)

        # Create pose message
        pose_msg = Pose()
        pose_msg.position.x = float(dx)
        pose_msg.position.y = float(dy)
        pose_msg.position.z = float(dz)
        pose_msg.orientation.x = final_q[0]
        pose_msg.orientation.y = final_q[1]
        pose_msg.orientation.z = final_q[2]
        pose_msg.orientation.w = final_q[3]
        
        self.pub_target.publish(pose_msg)

        # Publish gripper command
        grip_msg = Bool()
        grip_msg.data = bool(grip > 0)
        self.pub_gripper.publish(grip_msg)
        
        self.get_logger().info(
            f"📤 Published Dreamer actions: "
            f"pos=[{dx:.3f}, {dy:.3f}, {dz:.3f}], "
            f"wrist={wrist:.1f}°, grip={grip > 0}"
        )
        
        # --- CSV Logging Logic (New) ---
        current_time = self.get_clock().now()
        timestamp = [current_time.nanoseconds // 10**9, current_time.nanoseconds % 10**9]
        
        # Action log row: time + [dx, dy, dz, wrist, grip]
        self.act_csv_writer.writerow(timestamp + list(act[:5]))
        self.act_csv_file.flush()
        
        # Update the latest action data in the cache for plotting comparison
        self.data_cache["action"] = list(act[:5])


def main():
    rclpy.init()
    node = DreamerRosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destructor __del__ is not guaranteed to be called, so explicitly close the node
        # The node's destroy_node calls the destructor implicitly for resource cleanup.
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()