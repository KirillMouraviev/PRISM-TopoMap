from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    output_dir = Path.home() / 'prism_topomap_output' / 'scout_rosbag'
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument('path_to_gt_map', default_value=''),
        DeclareLaunchArgument('path_to_save_graph', default_value=str(output_dir / 'graph')),
        DeclareLaunchArgument('path_to_save_logs', default_value=str(output_dir / 'logs')),
        DeclareLaunchArgument('config_file', default_value='scout_rosbag.yaml'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the /clock topic from ros2 bag play --clock',
        ),
        Node(
            package='prism_topomap',
            executable='prism_topomap_node',
            output='screen',
            parameters=[{
                'path_to_gt_map': ParameterValue(
                    LaunchConfiguration('path_to_gt_map'), value_type=str
                ),
                'path_to_save_graph': LaunchConfiguration('path_to_save_graph'),
                'path_to_save_logs': LaunchConfiguration('path_to_save_logs'),
                'config_file': LaunchConfiguration('config_file'),
                'use_sim_time': use_sim_time,
            }],
        ),
        Node(
            package='prism_topomap',
            executable='tf_manager',
            name='odometry_publisher',
            output='screen',
            parameters=[{
                'publish_odom_from_tf': True,
                'source_frame': 'map',
                'target_frame': 'base_footprint',
                'use_sim_time': use_sim_time,
            }],
        ),
    ])
