#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import PyKDL
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import tf2_ros
from tf2_ros import TransformException
import transformations

from geometry_msgs.msg import Twist, WrenchStamped, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class AdmittanceController(Node):
    """
    ROS2 Admittance Controller for compliant control of a robotic arm.
    
    This controller implements the admittance equation:
    M·ẍ + D·ẋ + K·(x-x_d) = F_ext
    
    where:
    - M: Virtual mass matrix
    - D: Damping matrix
    - K: Stiffness matrix
    - x: Current position
    - x_d: Desired position
    - F_ext: External forces/torques
    
    The controller calculates the appropriate Cartesian velocity response
    to external forces and sends it to a Cartesian velocity controller.
    """

    def __init__(self):
        super().__init__('admittance_controller')
        
        # Parameters
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("end_link", "tool0")
        self.declare_parameter("force_torque_topic", "wrench")
        self.declare_parameter("cartesian_velocity_topic", "command_cart_vel")
        self.declare_parameter("ee_pose_topic", "ee_pose")
        self.declare_parameter("update_rate", 100.0)  # Hz
        
        # Mass, damping, stiffness matrices (diagonal)
        self.declare_parameter("mass_matrix", [10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        self.declare_parameter("damping_matrix", [70.0, 70.0, 70.0, 5.0, 5.0, 5.0])
        self.declare_parameter("stiffness_matrix", [300.0, 300.0, 300.0, 20.0, 20.0, 20.0])
        
        # Force/torque thresholds
        self.declare_parameter("force_dead_zone", [2.0, 2.0, 2.0])
        self.declare_parameter("torque_dead_zone", [0.5, 0.5, 0.5])
        
        # Get parameters
        self.base_link = self.get_parameter("base_link").value
        self.end_link = self.get_parameter("end_link").value
        self.ft_topic = self.get_parameter("force_torque_topic").value
        self.cart_vel_topic = self.get_parameter("cartesian_velocity_topic").value
        self.ee_pose_topic = self.get_parameter("ee_pose_topic").value
        self.update_rate = self.get_parameter("update_rate").value
        
        # Get control parameters
        self.mass_matrix = np.diag(self.get_parameter("mass_matrix").value)
        self.damping_matrix = np.diag(self.get_parameter("damping_matrix").value)
        self.stiffness_matrix = np.diag(self.get_parameter("stiffness_matrix").value)
        self.force_dead_zone = np.array(self.get_parameter("force_dead_zone").value)
        self.torque_dead_zone = np.array(self.get_parameter("torque_dead_zone").value)
        
        # Initialize state variables
        self.ee_pose = None  # Current end-effector pose
        self.ee_twist = np.zeros(6)  # Current end-effector twist
        self.wrench_ext = np.zeros(6)  # External wrench
        self.desired_pose = None  # Desired equilibrium pose
        self.desired_twist = np.zeros(6)  # Desired admittance-generated twist
        
        # Setup TF listener for force transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Set up callback group for concurrent callbacks
        callback_group = ReentrantCallbackGroup()
        
        # Create subscribers
        self.ft_sub = self.create_subscription(
            WrenchStamped,
            self.ft_topic, 
            self.wrench_callback,
            10,
            callback_group=callback_group
        )
        
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            self.ee_pose_topic,
            self.ee_pose_callback,
            10,
            callback_group=callback_group
        )
        
        # Create publishers
        self.cart_vel_pub = self.create_publisher(
            Twist,
            self.cart_vel_topic,
            10
        )
        
        # Create timer for admittance control loop
        self.control_timer = self.create_timer(
            1.0/self.update_rate,
            self.admittance_control_loop,
            callback_group=callback_group
        )
        
        self.last_time = self.get_clock().now()
        self.get_logger().info("Admittance controller initialized. Waiting for first pose measurements...")
    
    def ee_pose_callback(self, msg):
        """Process end-effector pose from Cartesian velocity controller"""
        if self.desired_pose is None:
            # Initialize desired pose to first received pose
            self.desired_pose = self.pose_stamped_to_array(msg)
            self.get_logger().info("Desired pose initialized to current pose")
        
        # Update current end-effector pose
        self.ee_pose = self.pose_stamped_to_array(msg)
    
    def wrench_callback(self, msg):
        """Process force/torque sensor readings"""
        # Convert force/torque message to numpy array
        wrench_ft_frame = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])
        
        # Apply deadzone to raw forces/torques
        deadzone = np.concatenate([self.force_dead_zone, self.torque_dead_zone])
        wrench_ft_frame_filtered = np.copy(wrench_ft_frame)
        
        # Zero out forces/torques below threshold
        for i in range(6):
            if abs(wrench_ft_frame[i]) < deadzone[i]:
                wrench_ft_frame_filtered[i] = 0.0
        
        # Transform wrench from sensor frame to base frame
        try:
            # Get transform from FT sensor frame to base frame
            transform = self.tf_buffer.lookup_transform(
                self.base_link,
                msg.header.frame_id,
                rclpy.time.Time())
            
            # Create rotation matrix
            quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            rot_matrix = transformations.quaternion_matrix(quat)[:3, :3]
            
            # Apply rotation to force and torque
            force_base = rot_matrix @ wrench_ft_frame_filtered[:3]
            torque_base = rot_matrix @ wrench_ft_frame_filtered[3:]
            
            # Combine into wrench in base frame
            self.wrench_ext = np.concatenate([force_base, torque_base])
            
        except TransformException as ex:
            self.get_logger().warning(f"Could not transform wrench: {ex}")
            # Use untransformed wrench if transform fails
            self.wrench_ext = wrench_ft_frame_filtered
    
    def admittance_control_loop(self):
        """Main admittance control loop"""
        # Skip if we don't have pose data yet
        if self.ee_pose is None or self.desired_pose is None:
            return
        
        # Calculate time since last update
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        if dt <= 0.0 or dt > 0.5:  # Skip if dt is invalid
            return
        
        # Calculate pose error (position and orientation)
        error = np.zeros(6)
        
        # Position error (simple subtraction)
        error[:3] = self.ee_pose[:3] - self.desired_pose[:3]
        
        # Orientation error (using quaternion difference)
        # This is simplified - a full implementation would use quaternion math
        # to find the angular error between current and desired orientation
        # For now, we'll use a simple difference of the quaternion vector parts
        error[3:] = self.ee_pose[3:6] - self.desired_pose[3:6]
        
        # Calculate virtual coupling wrench (spring-damper forces)
        coupling_wrench = self.damping_matrix @ self.ee_twist + self.stiffness_matrix @ error
        
        # Compute desired acceleration from admittance equation
        # M·ẍ = F_ext - D·ẋ - K·(x-x_d)
        acceleration = np.linalg.inv(self.mass_matrix) @ (self.wrench_ext - coupling_wrench)
        
        # Integrate acceleration to get velocity
        self.desired_twist += acceleration * dt
        
        # Apply velocity limits
        max_lin_vel = 0.5  # m/s
        max_ang_vel = 0.5  # rad/s
        self.desired_twist[:3] = np.clip(self.desired_twist[:3], -max_lin_vel, max_lin_vel)
        self.desired_twist[3:] = np.clip(self.desired_twist[3:], -max_ang_vel, max_ang_vel)
        
        # Send velocity commands to the Cartesian velocity controller
        twist_msg = Twist()
        
        # Scale velocities for safety (0.3 factor, as in the original C++ code)
        scale = 0.3
        twist_msg.linear.x = self.desired_twist[0] * scale
        twist_msg.linear.y = self.desired_twist[1] * scale
        twist_msg.linear.z = self.desired_twist[2] * scale
        twist_msg.angular.x = self.desired_twist[3] * scale
        twist_msg.angular.y = self.desired_twist[4] * scale
        twist_msg.angular.z = self.desired_twist[5] * scale
        
        self.cart_vel_pub.publish(twist_msg)
        
        # Debug output
        if np.linalg.norm(self.wrench_ext) > 0.1:
            self.get_logger().debug(f"Force: {self.wrench_ext[:3]}, Velocity: {self.desired_twist[:3]}")
    
    @staticmethod
    def pose_stamped_to_array(msg):
        """Convert PoseStamped message to numpy array [x, y, z, qx, qy, qz, qw]"""
        return np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])


def main(args=None):
    rclpy.init(args=args)
    
    controller = AdmittanceController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()