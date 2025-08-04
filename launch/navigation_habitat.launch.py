from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'path_to_gt_map',
            default_value='/home/kirill/TopoSLAM/GT/2n8kARJN3HM/map_cropped_0_600_300_900.png',
            description='Path to the ground truth map image'
        ),
        DeclareLaunchArgument(
            'path_to_save_json',
            default_value='/home/kirill/TopoSLAM/toposlam_ws/src/simple_toposlam_model/graph_jsons_new/2n8kARJN3HM',
            description='Path to save the JSON output'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='habitat_mp3d.yaml',
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
                'path_to_save_json': LaunchConfiguration('path_to_save_json'),
                'config_file': LaunchConfiguration('config_file'),
            }]
        ),

        Node(
            package='prism_topomap',
            executable='navigation.py',
            name='navigation_server',
            output='screen',
            parameters=[{
                'rate': 2,
                'tolerance': 0.5,
                'timeout': 1000,
            }]
        )
    ])