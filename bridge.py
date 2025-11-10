#!/usr/bin/env python3
import json
import zmq
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Bool


def tf_to_array(msg: TransformStamped):
    t = msg.transform.translation
    r = msg.transform.rotation
    return [t.x, t.y, t.z, r.x, r.y, r.z, r.w]


class DreamerRosBridge(Node):
    def __init__(self):
        super().__init__("dreamer_ros_bridge")

        # === ROS pubs/subs ===
        self.create_subscription(JointState, "/joint_states", self.cb_joint_states, 10)
        self.create_subscription(TransformStamped, "/block_pose", self.cb_block_pose, 10)
        self.create_subscription(TransformStamped, "/unity_target_pose", self.cb_target_pose, 10)
        self.create_subscription(Float32, "/wrist_angle", self.cb_wrist_angle, 10)
        self.create_subscription(Bool, "/gripper_command", self.cb_gripper_state, 10)
        self.create_subscription(Bool, "/left_contact_detected", self.cb_left_contact, 10)
        self.create_subscription(Bool, "/right_contact_detected", self.cb_right_contact, 10)

        self.pub_target = self.create_publisher(PoseStamped, "/target_pose_cmd", 10)
        self.pub_wrist = self.create_publisher(Float32, "/wrist_angle_cmd", 10)
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
        self.get_logger().info("ROS bridge ready.")

    # === Callbacks ===
    def cb_joint_states(self, msg): self.obs_cache["arm_joints"] = list(msg.position)
    def cb_block_pose(self, msg): self.obs_cache["block_pose"] = tf_to_array(msg)
    def cb_target_pose(self, msg): self.obs_cache["target_pose"] = tf_to_array(msg)
    def cb_wrist_angle(self, msg): self.obs_cache["wrist_angle"] = [msg.data]
    def cb_gripper_state(self, msg): self.obs_cache["gripper_state"] = [float(msg.data)]
    def cb_left_contact(self, msg): self.obs_cache["left_contact"] = [float(msg.data)]
    def cb_right_contact(self, msg): self.obs_cache["right_contact"] = [float(msg.data)]

    def timer_callback(self):
        # Send observations to Dreamer process
        if self.obs_cache:
            msg = json.dumps(self.obs_cache)
            self.pub.send_string(msg)

        # Receive action updates if available
        try:
            while self.sub.poll(timeout=0):
                msg = self.sub.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(msg)
                self.publish_actions(np.array(data["action"], dtype=np.float32))
        except zmq.Again:
            pass

    def publish_actions(self, act):
        dx, dy, dz, wrist, grip = act[:5]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = dx
        pose_msg.pose.position.y = dy
        pose_msg.pose.position.z = dz
        self.pub_target.publish(pose_msg)

        wrist_msg = Float32()
        wrist_msg.data = wrist
        self.pub_wrist.publish(wrist_msg)

        grip_msg = Bool()
        grip_msg.data = grip > 0
        self.pub_gripper.publish(grip_msg)


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
