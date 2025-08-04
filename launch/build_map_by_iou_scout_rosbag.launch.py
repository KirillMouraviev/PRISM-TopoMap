from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'path_to_gt_map',
            default_value='',
            description='Path to ground truth map'
        ),
        DeclareLaunchArgument(
            'path_to_save_graph',
            default_value='/home/kirill/TopoSLAM/toposlam_ws/src/simple_toposlam_model/graph_jsons_new/scout_rosbag',
            description='Path to save graph'
        ),
        DeclareLaunchArgument(
            'path_to_save_logs',
            default_value='/home/kirill/TopoSLAM/toposlam_ws/data/prism_topomap_logs',
            description='Path to save logs'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='scout_rosbag.yaml',
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
            name='odometry_publisher',
            output='screen'
        ),

        # Commented out static transform publisher
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='tf_base_to_lidar',
        #     arguments=['-0.300', '0.014', '0.883', '-0.008', '0.004', '-0.008', '1.000', 'base_link', 'velodyne']
        # ),

        # Commented out image republisher for zed camera
        # Node(
        #     package='image_transport',
        #     executable='republish',
        #     name='decompress_front_image',
        #     arguments=['compressed', 'in:=/zed_node/left/image_rect_color', 'raw', 'out:=/zed_node/left/image_rect_color']
        # ),

        # Image republisher for realsense camera
        Node(
            package='image_transport',
            executable='republish',
            name='decompress_realsense_image',
            arguments=['compressed', 'in:=/camera/color/image_raw', 'raw', 'out:=/camera/color/image_raw'],
            output='screen'
        )
    ])