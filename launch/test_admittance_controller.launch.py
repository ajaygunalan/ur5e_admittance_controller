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
    
    # Launch our custom controller
    controller_node = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ur5e_admittance_controller"],
        output="screen",
    )
    
    # Create and return launch description
    return LaunchDescription(
        declared_arguments + [gazebo_launch, controller_node]
    )