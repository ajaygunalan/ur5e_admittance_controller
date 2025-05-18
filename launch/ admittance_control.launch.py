from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type/series of used UR robot."
        )
    )
    
    # Initialize arguments
    ur_type = LaunchConfiguration("ur_type")
    
    # Include the Gazebo simulation launch file
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "launch", "ur_sim_control.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
        }.items(),
    )
    
    # Launch the Cartesian velocity controller
    cartesian_velocity_controller_node = Node(
        package="ur5e_admittance_controller",
        executable="cartesian_velocity_controller.py",
        name="cartesian_velocity_controller",
        output="screen",
        parameters=[{
            "controller_name": "scaled_joint_trajectory_controller",
            "update_rate": 100.0,
        }],
    )
    
    # Launch the admittance controller
    admittance_controller_node = Node(
        package="ur5e_admittance_controller",
        executable="admittance_controller.py",
        name="admittance_controller",
        output="screen",
        parameters=[{
            "mass_matrix": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0],
            "damping_matrix": [70.0, 70.0, 70.0, 5.0, 5.0, 5.0],
            "stiffness_matrix": [300.0, 300.0, 300.0, 20.0, 20.0, 20.0],
            "force_dead_zone": [2.0, 2.0, 2.0],
            "torque_dead_zone": [0.5, 0.5, 0.5],
            "force_torque_topic": "/wrench",
            "cartesian_velocity_topic": "/command_cart_vel",
            "ee_pose_topic": "/ee_pose",
            "update_rate": 100.0,
        }],
    )
    
    # Create and return launch description
    return LaunchDescription(
        declared_arguments + 
        [gazebo_launch, cartesian_velocity_controller_node, admittance_controller_node]
    )