from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'track_name',
            description='Name of the track'
        ),
        DeclareLaunchArgument(
            'path_to_save_graph',
            default_value='/data/maps/topo_graph_for_tracks_with_curbs_auto',
            description='Path to save the graph'
        ),
        DeclareLaunchArgument(
            'path_to_save_logs',
            default_value='/data/prism_topomap_logs',
            description='Path to save logs'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='itlp_campus_mapping.yaml',
            description='Configuration file name'
        ),

        # prism_topomap_node
        Node(
            package='prism_topomap',
            executable='prism_topomap_node',
            name='prism_topomap_node',
            output='screen',
            parameters=[{
                'path_to_save_graph': LaunchConfiguration('path_to_save_graph'),
                'path_to_save_logs': LaunchConfiguration('path_to_save_logs'),
                'config_file': LaunchConfiguration('config_file'),
            }]
        ),

        # transform_manager
        Node(
            package='prism_topomap',
            executable='tf_manager',
            name='transform_manager',
            output='screen',
            parameters=[{
                'source_frame': 'map',
                'target_frame': 'base_link',
                'publish_tf_from_odom': True,
                'odometry_topic': '/glim_ros/odom',
                'tf_from_odom_target_frame': 'base_link',
            }]
        ),

        # gt_tf_publisher
        Node(
            package='prism_topomap',
            executable='gt_for_itlp',
            name='gt_tf_publisher',
            output='screen',
            parameters=[{
                'path_to_gt': PathJoinSubstitution([
                    '/media/kirill/7CM/ITLP_campus_bags/dataset/outdoor',
                    LaunchConfiguration('track_name'),
                    'track.csv'
                ]),
                'odom_topic': '/glim_ros/odom',
                'source_frame': 'map',
                'target_frame': 'base_link',
            }]
        ),
    ])