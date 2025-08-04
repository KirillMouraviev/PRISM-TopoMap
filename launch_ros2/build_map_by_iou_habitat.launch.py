from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'scene_name',
            default_value='2n8kARJN3HM',
            description='Name of the scene'
        ),
        DeclareLaunchArgument(
            'path_to_gt_map',
            default_value=PathJoinSubstitution([
                '/home/kirill/TopoSLAM/GT',
                LaunchConfiguration('scene_name')
            ]),
            description='Path to ground truth map'
        ),
        DeclareLaunchArgument(
            'path_to_save_graph',
            default_value=PathJoinSubstitution([
                '/home/kirill/TopoSLAM/toposlam_ws/src/simple_toposlam_model/graph_jsons_new',
                LaunchConfiguration('scene_name')
            ]),
            description='Path to save graph'
        ),
        DeclareLaunchArgument(
            'path_to_save_logs',
            default_value='/home/kirill/TopoSLAM/toposlam_ws/data/prism_topomap_logs',
            description='Path to save logs'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='habitat_mp3d_noised_odom.yaml',
            description='Configuration file name'
        ),
        
        # Nodes
        Node(
            package='prism_topomap',
            executable='prism_topomap_node.py',
            name='prism_topomap_node',
            output='screen',
            parameters=[{
                'path_to_gt_map': LaunchConfiguration('path_to_gt_map'),
                'path_to_save_graph': LaunchConfiguration('path_to_save_graph'),
                'config_file': LaunchConfiguration('config_file'),
            }]
        ),
        
        Node(
            package='prism_topomap',
            executable='tf_to_odom.py',
            name='odom_from_tf',
            parameters=[{
                'source_frame': 'map',
                'target_frame': 'base_link',
            }]
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_map_to_odom',
            arguments=['0', '0', '0', '2.51', '0', '0', 'map', 'odom']
        )
    ])