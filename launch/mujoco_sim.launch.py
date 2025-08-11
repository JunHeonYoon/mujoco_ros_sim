from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='franka_fr3',
        description='Name of the robot model to be used in MuJoCo'
    )

    robot_name = LaunchConfiguration('robot_name')

    sim_node = Node(
        package='mujoco_ros_sim',      
        executable='mujoco_sim',   
        name='mujoco_sim_node',
        output='screen',
        parameters=[{
            'robot_name': robot_name
        }]
    )

    return LaunchDescription([
        robot_name_arg,
        sim_node
    ])