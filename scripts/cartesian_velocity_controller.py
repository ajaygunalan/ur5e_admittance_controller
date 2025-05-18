#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import time
import PyKDL
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import os
from ament_index_python.packages import get_package_share_directory

from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance


class CartesianVelocityController(Node):
    """
    A ROS2 Python implementation of a Cartesian velocity controller.
    This controller:
    1. Subscribes to Cartesian velocity commands (Twist)
    2. Uses KDL to convert Cartesian velocities to joint velocities
    3. Integrates velocities to positions
    4. Sends position commands via FollowJointTrajectory action
    """

    def __init__(self):
        super().__init__('cartesian_velocity_controller')
        
        # -------------------------------------------------------------------
        # Parameters 
        # -------------------------------------------------------------------
        self.declare_parameter("controller_name", "scaled_joint_trajectory_controller")
        self.declare_parameter("robot_description_param", "robot_description")
        self.declare_parameter("update_rate", 100.0)  # Hz
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("end_link", "tool0")
        self.declare_parameter("joints", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])
        
        # Get parameters
        self.controller_name = self.get_parameter("controller_name").value
        self.robot_description_param = self.get_parameter("robot_description_param").value
        self.update_rate = self.get_parameter("update_rate").value
        self.base_link = self.get_parameter("base_link").value
        self.end_link = self.get_parameter("end_link").value
        self.joint_names = self.get_parameter("joints").value
        
        # Define joint limits for safety
        self.joint_limits = [
            [-3.14159, 3.14159],  # shoulder_pan_joint
            [-3.14159, 0.0],      # shoulder_lift_joint
            [-3.14159, 3.14159],  # elbow_joint
            [-3.14159, 0.0],      # wrist_1_joint
            [-3.14159, 3.14159],  # wrist_2_joint
            [-3.14159, 3.14159],  # wrist_3_joint
        ]
        
        # -------------------------------------------------------------------
        # Initialize state variables
        # -------------------------------------------------------------------
        self.joint_positions = np.zeros(len(self.joint_names))
        self.joint_velocities = np.zeros(len(self.joint_names))
        self.cartesian_vel_cmd = PyKDL.Twist()
        self.last_update_time = self.get_clock().now()
        self.executing_goal = False  # Flag to prevent goal conflicts
        self.last_command_time = self.get_clock().now()
        
        # -------------------------------------------------------------------
        # Initialize KDL kinematics
        # -------------------------------------------------------------------
        try:
            self.init_kinematics()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize kinematics: {e}")
            raise
        
        # -------------------------------------------------------------------
        # Set up ROS infrastructure 
        # -------------------------------------------------------------------
        callback_group = ReentrantCallbackGroup()
        
        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10,
            callback_group=callback_group
        )
        
        self.cart_vel_sub = self.create_subscription(
            Twist,
            'command_cart_vel',
            self.cart_vel_callback,
            10,
            callback_group=callback_group
        )
        
        # Set up action client
        controller_action = f'{self.controller_name}/follow_joint_trajectory'
        self.trajectory_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            controller_action,
            callback_group=callback_group
        )
        
        # Wait for action server
        self.get_logger().info(f"Waiting for action server on {controller_action}")
        if not self.trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"Timeout waiting for action server {controller_action}")
            raise RuntimeError(f"Action server {controller_action} not available")
        else:
            self.get_logger().info("Action server found!")
        
        # Publisher for current end-effector pose
        self.ee_pose_pub = self.create_publisher(
            PoseStamped,
            'ee_pose',
            10
        )
        
        # Create timer for control loop
        self.control_timer = self.create_timer(
            1.0/self.update_rate,
            self.control_loop,
            callback_group=callback_group
        )
        
        self.get_logger().info("Cartesian velocity controller initialized and ready!")
        self.get_logger().info("Listening for commands on topic: /command_cart_vel")
    
    def init_kinematics(self):
        """Initialize KDL kinematics from URDF"""
        # First try to get robot description directly from parameter
        robot_description = None
        
        try:
            robot_description = self.get_parameter(self.robot_description_param).value
            self.get_logger().info("Got robot description from parameter")
        except Exception:
            self.get_logger().info("Robot description not found as local parameter")
            
        # If not available, try to get it from parameter service
        if not robot_description:
            try:
                from rcl_interfaces.srv import GetParameters
                self.get_logger().info("Trying to get robot description from parameter service")
                
                cli = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
                if not cli.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn("Parameter service not available, trying alternative method")
                else:
                    req = GetParameters.Request()
                    req.names = [self.robot_description_param]
                    future = cli.call_async(req)
                    rclpy.spin_until_future_complete(self, future)
                    robot_description = future.result().values[0].string_value
                    self.get_logger().info("Got robot description from parameter service")
            except Exception as e:
                self.get_logger().warn(f"Failed to get robot description from service: {e}")
                
        # If still not available, try to load from file
        if not robot_description:
            try:
                self.get_logger().info("Trying to load robot description from file")
                # Try to find the URDF file
                urdf_path = os.path.join(
                    get_package_share_directory('ur_description'),
                    'urdf',
                    'ur5e.urdf.xacro'  # Assuming ur5e is the default robot
                )
                
                if os.path.exists(urdf_path):
                    # This is just an alternative approach - in practice you'd need to process the xacro
                    self.get_logger().error(f"Found URDF at {urdf_path}, but xacro processing required")
                    self.get_logger().error("Please ensure robot_description parameter is available")
                    raise RuntimeError("Cannot load robot description from file directly")
            except Exception as e:
                self.get_logger().error(f"Failed to load robot description from file: {e}")
        
        if not robot_description:
            raise RuntimeError("Could not obtain robot description")
        
        # Parse URDF and create KDL tree
        urdf_model = URDF.from_xml_string(robot_description)
        kdl_tree = treeFromUrdfModel(urdf_model)
        
        # Extract the chain from base to end-effector
        self.kdl_chain = kdl_tree.getChain(self.base_link, self.end_link)
        
        # Create KDL solvers
        self.fk_pos_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)
        self.fk_vel_solver = PyKDL.ChainFkSolverVel_recursive(self.kdl_chain)
        self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)
        
        self.get_logger().info(f"Initialized kinematics with {self.kdl_chain.getNrOfJoints()} joints")
    
    def joint_state_callback(self, msg):
        """Update current joint state from robot"""
        # Match joint names with the ones we're interested in
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
                if msg.velocity and len(msg.velocity) > idx:
                    self.joint_velocities[i] = msg.velocity[idx]
                else:
                    self.joint_velocities[i] = 0.0
        
        # Publish current end-effector pose
        self.publish_ee_pose()
    
    def cart_vel_callback(self, msg):
        """Handle new Cartesian velocity commands"""
        self.cartesian_vel_cmd = PyKDL.Twist(
            PyKDL.Vector(msg.linear.x, msg.linear.y, msg.linear.z),
            PyKDL.Vector(msg.angular.x, msg.angular.y, msg.angular.z)
        )
        
        # For debugging
        self.get_logger().debug(f"Received vel cmd: lin=[{msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}], " + 
                               f"ang=[{msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f}]")
        
        # Update time of last command
        self.last_command_time = self.get_clock().now()
    
    def publish_ee_pose(self):
        """Publish current end-effector pose"""
        # Create KDL joint array
        kdl_jnt_pos = PyKDL.JntArray(len(self.joint_positions))
        for i in range(len(self.joint_positions)):
            kdl_jnt_pos[i] = self.joint_positions[i]
        
        # Calculate forward kinematics
        ee_frame = PyKDL.Frame()
        self.fk_pos_solver.JntToCart(kdl_jnt_pos, ee_frame)
        
        # Convert to PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.base_link
        
        # Position
        pose_msg.pose.position.x = ee_frame.p.x()
        pose_msg.pose.position.y = ee_frame.p.y()
        pose_msg.pose.position.z = ee_frame.p.z()
        
        # Orientation (converted to quaternion)
        quat = ee_frame.M.GetQuaternion()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.ee_pose_pub.publish(pose_msg)
    
    def control_loop(self):
        """Main control loop to convert Cartesian velocities to joint positions"""
        # Calculate time since last update
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time
        
        # Skip if dt is invalid or if we're executing a goal
        if dt <= 0.0 or dt > 0.5:
            return
        
        # Skip if we're already executing a trajectory
        if self.executing_goal:
            # Avoid warning spamming by only logging occasionally
            if (current_time - self.last_command_time).nanoseconds / 1e9 < 1.0:
                self.get_logger().debug("Skipping control loop - already executing trajectory")
            return
        
        # Check for command timeout - if no recent commands, skip processing
        command_age = (current_time - self.last_command_time).nanoseconds / 1e9
        if command_age > 0.5:  # 500ms timeout
            # Zero velocity if no recent commands (normal operation)
            return
        
        # Create KDL joint array for current position
        kdl_jnt_pos = PyKDL.JntArray(len(self.joint_positions))
        for i in range(len(self.joint_positions)):
            kdl_jnt_pos[i] = self.joint_positions[i]
        
        # Solve inverse velocity kinematics to get joint velocities from Cartesian velocities
        kdl_jnt_vel = PyKDL.JntArray(len(self.joint_positions))
        result = self.ik_vel_solver.CartToJnt(kdl_jnt_pos, self.cartesian_vel_cmd, kdl_jnt_vel)
        
        if result < 0:
            self.get_logger().warn(f"IK solver failed with error {result}")
            return
        
        # Convert to numpy array for easier manipulation
        joint_vel_cmd = np.zeros(len(self.joint_positions))
        for i in range(len(self.joint_positions)):
            joint_vel_cmd[i] = kdl_jnt_vel[i]
        
        # Apply velocity limits
        max_vel = 1.0  # rad/s
        joint_vel_cmd = np.clip(joint_vel_cmd, -max_vel, max_vel)
        
        # Integrate velocity to get new position
        new_positions = self.joint_positions + joint_vel_cmd * dt
        
        # Apply joint limits
        for i in range(len(new_positions)):
            if i < len(self.joint_limits):
                new_positions[i] = np.clip(new_positions[i], 
                                          self.joint_limits[i][0], 
                                          self.joint_limits[i][1])
        
        # Send joint trajectory with the new positions
        self.send_joint_trajectory(new_positions)
    
    def send_joint_trajectory(self, positions):
        """Send a trajectory point to the robot"""
        # Set executing flag to prevent overlapping commands
        self.executing_goal = True
        
        # Create goal message
        goal = FollowJointTrajectory.Goal()
        
        # Create trajectory
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names
        
        # Create trajectory point - use a consistent time horizon for stability
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.velocities = [0.0] * len(positions)  # Zero velocities at the target
        
        # Use a fixed time from start for more stability (100ms)
        point.time_from_start = Duration(sec=0, nanosec=100000000)
        
        goal.trajectory.points.append(point)
        
        # Set goal tolerance
        goal.goal_time_tolerance = Duration(sec=0, nanosec=200000000)  # 200ms tolerance
        goal.goal_tolerance = [
            JointTolerance(name=self.joint_names[i], position=0.02, velocity=0.02)
            for i in range(len(self.joint_names))
        ]
        
        # Send goal
        self.trajectory_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        ).add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Handle response from action server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Trajectory goal rejected")
            self.executing_goal = False  # Reset flag on rejection
            return
        
        # Request result asynchronously
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """Handle trajectory execution result"""
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f"Goal failed with status: {status}")
        
        # Reset executing flag to allow new commands
        self.executing_goal = False
    
    def feedback_callback(self, feedback_msg):
        """Handle feedback from action server"""
        # Could be used for monitoring trajectory progress
        pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create and run the controller
        controller = CartesianVelocityController()
        
        # Log instructions for users
        controller.get_logger().info("\n" + "-"*80 + 
            "\nCartesian Velocity Controller is running!" +
            "\n\nControl the robot using:" +
            "\n  - From terminal: ros2 topic pub /command_cart_vel geometry_msgs/msg/Twist ..." +
            "\n  - With keyboard: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/command_cart_vel" +
            "\n\nPress Ctrl+C to exit" +
            "\n" + "-"*80)
        
        # Spin the node to process callbacks
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Shutting down due to keyboard interrupt")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Clean shutdown
        rclpy.shutdown()


if __name__ == '__main__':
    main()