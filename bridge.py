#!/usr/bin/env python3
import json
import zmq
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Bool
import math



def transform_to_array(msg: TransformStamped):
    """Convert TransformStamped to array [x, y, z, qx, qy, qz, qw]"""
    t = msg.transform.translation
    r = msg.transform.rotation
    return [t.x, t.y, t.z, r.x, r.y, r.z, r.w]


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

class DreamerRosBridge(Node):
    def __init__(self):
        super().__init__("dreamer_ros_bridge")

        # === ROS pubs/subs ===
        self.create_subscription(JointState, "/joint_states", self.cb_joint_states, 10)
        
        # Subscribe to Pose instead of TransformStamped
        self.create_subscription(Pose, "/block_pose", self.cb_block_pose, 10)
        self.create_subscription(Pose, "/unity_target_pose", self.cb_target_pose, 10)
        
        self.create_subscription(Float32, "/wrist_angle", self.cb_wrist_angle, 10)
        self.create_subscription(Bool, "/gripper_command", self.cb_gripper_state, 10)
        self.create_subscription(Bool, "/left_contact_detected", self.cb_left_contact, 10)
        self.create_subscription(Bool, "/right_contact_detected", self.cb_right_contact, 10)

        self.pub_target = self.create_publisher(Pose, "/target_pose", 10)
        self.pub_wrist = self.create_publisher(Float32, "/wrist_angle", 10)
        self.pub_gripper = self.create_publisher(Bool, "/gripper_command", 10)

        self.obs_cache = {}

        # === ZeroMQ sockets ===
        ctx = zmq.Context()
        self.pub = ctx.socket(zmq.PUB)
        self.pub.bind("tcp://127.0.0.1:5556")

        self.sub = ctx.socket(zmq.SUB)
        self.sub.bind("tcp://127.0.0.1:5557")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Debug: track what we've received
        self.received_topics = set()
        
        self.get_logger().info("ROS bridge ready.")

    # === Callbacks ===
    def cb_joint_states(self, msg):
        self.obs_cache["arm_joints"] = list(msg.position)
        if "arm_joints" not in self.received_topics:
            self.received_topics.add("arm_joints")
            self.get_logger().info("✓ Received arm_joints")
    
    def cb_block_pose(self, msg):
        self.obs_cache["block_pose"] = pose_to_array(msg)
        if "block_pose" not in self.received_topics:
            self.received_topics.add("block_pose")
            self.get_logger().info("✓ Received block_pose")
    
    def cb_target_pose(self, msg):
        self.obs_cache["target_pose"] = pose_to_array(msg)
        if "target_pose" not in self.received_topics:
            self.received_topics.add("target_pose")
            self.get_logger().info("✓ Received target_pose")
    
    def cb_wrist_angle(self, msg):
        self.obs_cache["wrist_angle"] = [msg.data]
        if "wrist_angle" not in self.received_topics:
            self.received_topics.add("wrist_angle")
            self.get_logger().info("✓ Received wrist_angle")
    
    def cb_gripper_state(self, msg):
        self.obs_cache["gripper_state"] = [float(msg.data)]
        if "gripper_state" not in self.received_topics:
            self.received_topics.add("gripper_state")
            self.get_logger().info("✓ Received gripper_state")
    
    def cb_left_contact(self, msg):
        self.obs_cache["left_contact"] = [float(msg.data)]
        if "left_contact" not in self.received_topics:
            self.received_topics.add("left_contact")
            self.get_logger().info("✓ Received left_contact")
    
    def cb_right_contact(self, msg):
        self.obs_cache["right_contact"] = [float(msg.data)]
        if "right_contact" not in self.received_topics:
            self.received_topics.add("right_contact")
            self.get_logger().info("✓ Received right_contact")

    def timer_callback(self):
        # Send observations to Dreamer process
        if self.obs_cache:
            msg = json.dumps(self.obs_cache)
            self.pub.send_string(msg)
            
            # Debug: show what we're sending (only first time all are available)
            required = ["arm_joints", "block_pose", "target_pose", "wrist_angle", 
                       "gripper_state", "left_contact", "right_contact"]
            if all(k in self.obs_cache for k in required) and len(self.received_topics) == 7:
                self.get_logger().info(f"✓ Sending complete observation set to Dreamer")
                # Only log once
                self.received_topics.add("_logged_complete")

        # Receive action updates if available
        try:
            while self.sub.poll(timeout=0):
                msg_str = self.sub.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(msg_str)
                action = np.array(data["action"], dtype=np.float32)
                self.get_logger().info(f"Received action from Dreamer: {action}")
                self.publish_actions(action)
        except zmq.Again:
            pass

    def publish_actions(self, act):
        dx, dy, dz, wrist, grip = act[:5]

        '''
                q = quaternion_from_euler(0, 0, math.radians(self.wristAngle))
        final_q = quaternion_multiply(q, base_down_q)

        safe_pose = Pose()
        safe_pose.position.x = ros_x
        safe_pose.position.y = ros_y
        safe_pose.position.z = ros_z
        safe_pose.orientation.x = final_q[0]
        safe_pose.orientation.y = final_q[1]
        safe_pose.orientation.z = final_q[2]
        safe_pose.orientation.w = final_q[3]
        '''
        base_down_q = np.array([0.7071068, -0.7071068, 0.0, 0.0])

        q = quaternion_from_euler_z(math.radians(wrist))
        final_q = quaternion_multiply(q, base_down_q)

        pose_msg = Pose()
        pose_msg.position.x = float(dx)
        pose_msg.position.y = float(dy)
        pose_msg.position.z = float(dz)
        pose_msg.orientation.x = final_q[0]
        pose_msg.orientation.y = final_q[1]
        pose_msg.orientation.z = final_q[2]
        pose_msg.orientation.w = final_q[3]
        self.pub_target.publish(pose_msg)

        grip_msg = Bool()
        grip_msg.data = bool(grip > 0)
        self.pub_gripper.publish(grip_msg)
        
        self.get_logger().info(f"Published actions: pos=[{dx:.3f}, {dy:.3f}, {dz:.3f}], wrist={wrist:.3f}, grip={grip > 0}")


def main():
    rclpy.init()
    node = DreamerRosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()