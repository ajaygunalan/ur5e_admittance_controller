# cartesian_velocity_controller.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur5e_admittance_controller',
            executable='cartesian_velocity_controller.py',
            name='cartesian_velocity_controller',
            output='screen',
            parameters=[
                {'controller_name': 'scaled_joint_trajectory_controller'},
                {'update_rate': 100.0},
            ],
        ),
    ])