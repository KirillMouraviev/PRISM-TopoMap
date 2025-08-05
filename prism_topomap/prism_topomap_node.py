#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
np.float = np.float64
import ros2_numpy
import os
import cv2
import sys
import transformations as tf
import time
import yaml
import copy
from prism_topomap.utils import *
from prism_topomap.gt_map import GTMap
from prism_topomap.prism_topomap import TopoSLAMModel
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32, Int32MultiArray
# from toposlam_msgs.msg import TopologicalPath
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool
from collections import deque
from cv_bridge import CvBridge

class ResultsPublisher(Node):
    def __init__(self, map_frame):
        super().__init__('results_publisher')
        # QoS profile for publishers (similar to latch=True in ROS1)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100
        )
        # Localization
        self.cand_cloud_publisher = self.create_publisher(PointCloud2, '/candidate_cloud', qos_profile)
        self.matched_points_publisher = self.create_publisher(Marker, '/matched_points', qos_profile)
        self.unmatched_points_publisher = self.create_publisher(Marker, '/unmatched_points', qos_profile)
        self.transforms_publisher = self.create_publisher(Marker, '/localization_transforms', qos_profile)
        self.first_pr_publisher = self.create_publisher(Marker, '/first_point', qos_profile)
        self.first_pr_image_publisher = self.create_publisher(Image, '/place_recognition/image', qos_profile)
        self.freeze_publisher = self.create_publisher(Bool, '/freeze', qos_profile)
        self.ref_cloud_pub = self.create_publisher(PointCloud2, '/ref_cloud', qos_profile)
        # Visualization
        self.gt_map_publisher = self.create_publisher(OccupancyGrid, '/habitat/gt_map', qos_profile)
        self.last_vertex_publisher = self.create_publisher(Marker, '/last_vertex', qos_profile)
        self.last_vertex_id_publisher = self.create_publisher(Int32, '/last_vertex_id', qos_profile)
        self.loop_closure_results_publisher = self.create_publisher(MarkerArray, '/loop_closure_results', qos_profile)
        self.rel_pose_of_vcur_publisher = self.create_publisher(PoseStamped, '/rel_pose_of_vcur', qos_profile)
        self.local_grid_publisher = self.create_publisher(OccupancyGrid, '/local_grid', qos_profile)
        self.cur_grid_publisher = self.create_publisher(OccupancyGrid, '/current_grid', qos_profile)
        self.graph_viz_pub = self.create_publisher(MarkerArray, 'topological_map', qos_profile)
        # Navigation
        self.path_publisher = self.create_publisher(TopologicalPath, '/topological_path', qos_profile)
        self.path_marker_publisher = self.create_publisher(Marker, '/topological_path_marker', qos_profile)
        self.pointgoal_publisher = self.create_publisher(PoseStamped, '/pointgoal', qos_profile)
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.map_frame = map_frame
        self.path_to_gt_map = None
        # Current timestamp
        self.current_stamp = None

    def init_gt_map(self, path_to_gt_map):
        gt_map_filename = [fn for fn in os.listdir(path_to_gt_map) if fn.startswith('map_cropped_')][0]
        self.gt_map = GTMap(os.path.join(path_to_gt_map, gt_map_filename))

    def publish_localization_results(self, graph, vertex_ids_matched, rel_poses, vertex_ids_unmatched):
        print('vertex_ids_unmatched:', vertex_ids_unmatched)
        if len(vertex_ids_matched) > 0:
            vertex_id_first = vertex_ids_matched[0]
            # Publish top-1 PlaceRecognition vertex
            vertices_marker = Marker()
            #vertices_marker = ns = 'points_and_lines'
            vertices_marker.type = Marker.POINTS
            vertices_marker.id = 0
            vertices_marker.header.frame_id = self.map_frame
            vertices_marker.header.stamp = rclpy.time.Time()
            vertices_marker.scale.x = self.match_marker_size * 1.5
            vertices_marker.scale.y = self.match_marker_size * 1.5
            vertices_marker.scale.z = self.match_marker_size * 1.5
            vertices_marker.color.r = 0
            vertices_marker.color.g = 1
            vertices_marker.color.b = 0
            vertices_marker.color.a = 1
            x, y, _ = graph.vertices[vertex_id_first]['pose_for_visualization']
            img_front = graph.vertices[vertex_id_first].get('img_front', None)
            vertices_marker.points.append(Point(x, y, 0.1))
            self.first_pr_publisher.publish(vertices_marker)

            # Publish matched vertices
            vertices_marker = Marker()
            #vertices_marker = ns = 'points_and_lines'
            vertices_marker.type = Marker.POINTS
            vertices_marker.id = 0
            vertices_marker.header.frame_id = self.map_frame
            vertices_marker.header.stamp = self.get_clock().now().to_msg()
            vertices_marker.scale.x = self.match_marker_size
            vertices_marker.scale.y = self.match_marker_size
            vertices_marker.scale.z = self.match_marker_size
            vertices_marker.color.r = 0
            vertices_marker.color.g = 1
            vertices_marker.color.b = 0
            vertices_marker.color.a = 1
            localized_vertices = [graph.vertices[i] for i in vertex_ids_matched]
            for vertex_dict in localized_vertices:
                x, y, _ = vertex_dict['pose_for_visualization']
                vertices_marker.points.append(Point(x, y, 0.1))
            self.matched_points_publisher.publish(vertices_marker)

            transforms_marker = Marker()
            transforms_marker.type = Marker.LINE_LIST
            transforms_marker.header.frame_id = self.map_frame
            transforms_marker.header.stamp = self.get_clock().now().to_msg()
            transforms_marker.scale.x = 0.1
            transforms_marker.color.r = 1
            transforms_marker.color.g = 0
            transforms_marker.color.b = 0
            transforms_marker.color.a = 0.5
            transforms_marker.pose.orientation.w = 1
            for vertex_dict, rel_pose in zip(localized_vertices, rel_poses):
                x, y, theta = vertex_dict['pose_for_visualization']
                transforms_marker.points.append(Point(x, y, 0.1))
                x, y, theta = apply_pose_shift([x, y, theta], *rel_pose)
                transforms_marker.points.append(Point(x, y, 0.1))
            self.transforms_publisher.publish(transforms_marker)

        # Publish unmatched vertices
        vertices_marker = Marker()
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = self.get_clock().now().to_msg()
        vertices_marker.scale.x = self.match_marker_size
        vertices_marker.scale.y = self.match_marker_size
        vertices_marker.scale.z = self.match_marker_size
        vertices_marker.color.r = 1
        vertices_marker.color.g = 1
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        localized_vertices = [graph.vertices[i] for i in vertex_ids_unmatched]
        for vertex_dict in localized_vertices:
            x, y, _ = vertex_dict['pose_for_visualization']
            vertices_marker.points.append(Point(x, y, 0.1))
        self.unmatched_points_publisher.publish(vertices_marker)

    def publish_ref_cloud(self, cloud, stamp):
        cloud_with_fields = np.zeros((cloud.shape[0]), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),#])
            ('r', np.uint8),
            ('g', np.uint8),
            ('b', np.uint8)])
        cloud_with_fields['x'] = cloud[:, 0]
        cloud_with_fields['y'] = cloud[:, 1]
        cloud_with_fields['z'] = cloud[:, 2]
        #cloud_with_fields['r'] = cloud[:, 3]
        #cloud_with_fields['g'] = cloud[:, 4]
        #cloud_with_fields['b'] = cloud[:, 5]
        #cloud_with_fields = ros2_numpy.point_cloud2.merge_rgb_fields(cloud_with_fields)
        cloud_msg = ros2_numpy.point_cloud2.array_to_pointcloud2(cloud_with_fields)
        if stamp is not None:
            cloud_msg.header.stamp = stamp
        else:
            cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = 'base_link'
        self.ref_cloud_pub.publish(cloud_msg)

    def publish_gt_map(self):
        gt_map_msg = OccupancyGrid()
        gt_map_msg.header.stamp = self.get_clock().now().to_msg()
        gt_map_msg.header.frame_id = self.map_frame
        gt_map_msg.info.resolution = 0.05
        gt_map_msg.info.width = self.gt_map.gt_map.shape[1]
        gt_map_msg.info.height = self.gt_map.gt_map.shape[0]
        # gt_map_msg.info.origin.position.x = -100 + self.gt_map.start_j / 20
        # gt_map_msg.info.origin.position.y = -100 + self.gt_map.start_i / 20
        gt_map_msg.info.origin.position.x = -24 + self.gt_map.start_j / 20
        gt_map_msg.info.origin.position.y = -24 + self.gt_map.start_i / 20
        gt_map_msg.info.origin.orientation.x = 0
        gt_map_msg.info.origin.orientation.y = 0
        gt_map_msg.info.origin.orientation.z = 0
        gt_map_msg.info.origin.orientation.w = 1
        gt_map_ravel = self.gt_map.gt_map.ravel()
        gt_map_data = self.gt_map.gt_map.ravel().astype(np.int8)
        gt_map_data[gt_map_ravel == 0] = 100
        gt_map_data[gt_map_ravel == 127] = -1
        gt_map_data[gt_map_ravel == 255] = 0
        gt_map_msg.data = list(gt_map_data)
        self.gt_map_publisher.publish(gt_map_msg)

    def publish_graph(self, graph):
        # Publish graph for visualization
        graph_msg = MarkerArray()
        vertices_marker = Marker()
        #vertices_marker = ns = 'points_and_lines'
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = self.get_clock().now().to_msg()
        vertices_marker.scale.x = self.vertex_marker_size
        vertices_marker.scale.y = self.vertex_marker_size
        vertices_marker.scale.z = self.vertex_marker_size
        vertices_marker.color.r = 1
        vertices_marker.color.g = 0
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        for vertex_dict in graph.vertices:
            x, y, _ = vertex_dict['pose_for_visualization']
            vertices_marker.points.append(Point(x, y, 0.05))
        graph_msg.markers.append(vertices_marker)

        edges_marker = Marker()
        edges_marker.id = 1
        edges_marker.type = Marker.LINE_LIST
        edges_marker.header = vertices_marker.header
        edges_marker.scale.x = self.edge_marker_size
        edges_marker.color.r = 0
        edges_marker.color.g = 0
        edges_marker.color.b = 1
        edges_marker.color.a = 0.5
        edges_marker.pose.orientation.w = 1
        for u in range(len(graph.vertices)):
            for v, pose in graph.adj_lists[u]:
                ux, uy, _ = graph.vertices[u]['pose_for_visualization']
                vx, vy, _ = graph.vertices[v]['pose_for_visualization']
                edges_marker.points.append(Point(ux, uy, 0.05))
                edges_marker.points.append(Point(vx, vy, 0.05))
        graph_msg.markers.append(edges_marker)

        vertex_orientation_marker = Marker()
        vertex_orientation_marker.id = 2
        vertex_orientation_marker.type = Marker.LINE_LIST
        vertex_orientation_marker.header = vertices_marker.header
        vertex_orientation_marker.scale.x = self.vertex_orientation_marker_size
        vertex_orientation_marker.color.r = 1
        vertex_orientation_marker.color.g = 0
        vertex_orientation_marker.color.b = 0
        vertex_orientation_marker.color.a = 1
        vertex_orientation_marker.pose.orientation.w = 1
        for vertex_dict in graph.vertices:
            x, y, theta = vertex_dict['pose_for_visualization']
            vertex_orientation_marker.points.append(Point(x, y, 0.1))
            vertex_orientation_marker.points.append(Point(x + np.cos(theta) * 1, y + np.sin(theta) * 1, 0.05))
        # graph_msg.markers.append(vertex_orientation_marker)

        cnt = 3
        text_marker = Marker()
        text_marker.header = vertices_marker.header
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.scale.z = self.text_marker_size
        text_marker.color.r = 1
        text_marker.color.g = 0.5
        text_marker.color.b = 0
        text_marker.color.a = 1
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.w = 1
        for i, vertex_dict in enumerate(graph.vertices):
            x, y, theta = vertex_dict['pose_for_visualization']
            text_marker.id = cnt
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.text = '{}: ({}, {}, {})'.format(i, round(x, 1), round(y, 1), round(theta, 2))
            graph_msg.markers.append(copy.deepcopy(text_marker))
            cnt += 1
        text_marker.scale.z = self.text_marker_size
        text_marker.color.r = 0
        text_marker.color.g = 1
        text_marker.color.b = 1
        for u in range(len(graph.vertices)):
            for v, pose in graph.adj_lists[u]:
                if u >= v:
                    continue
                ux, uy, _ = graph.vertices[u]['pose_for_visualization']
                vx, vy, _ = graph.vertices[v]['pose_for_visualization']
                text_marker.id = cnt
                text_marker.pose.position.x = (ux + vx) / 2
                text_marker.pose.position.y = (uy + vy) / 2
                text_marker.text = '({}, {}, {})'.format(round(pose[0], 1), round(pose[1], 1), round(pose[2], 2))
                graph_msg.markers.append(copy.deepcopy(text_marker))
                cnt += 1

        self.graph_viz_pub.publish(graph_msg)

    def publish_last_vertex(self, last_vertex, last_vertex_id):
        marker_msg = Marker()
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.header.frame_id = self.map_frame
        marker_msg.type = Marker.SPHERE
        last_x, last_y, last_theta = last_vertex['pose_for_visualization']
        marker_msg.pose.position.x = last_x
        marker_msg.pose.position.y = last_y
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.w = 1.0
        marker_msg.color.r = 0
        marker_msg.color.g = 1
        marker_msg.color.b = 0
        marker_msg.color.a = 1
        marker_msg.scale.x = self.vcur_marker_size
        marker_msg.scale.y = self.vcur_marker_size
        marker_msg.scale.z = self.vcur_marker_size
        self.last_vertex_publisher.publish(marker_msg)
        vertex_id_msg = Int32()
        vertex_id_msg.data = last_vertex_id
        self.last_vertex_id_publisher.publish(vertex_id_msg)
        self.tfbr.sendTransform((last_x, last_y, 0),
                                    tf.quaternion_from_euler(0, 0, last_theta),
                                    self.current_stamp,
                                    "vcur",
                                    self.map_frame)

    def publish_loop_closure_results(self, graph, path, global_pose_for_visualization):
        assert len(path) >= 2
        loop_closure_msg = MarkerArray()
        vertices_marker = Marker()
        x, y, _ = global_pose_for_visualization
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = self.get_clock().now().to_msg()
        vertices_marker.scale.x = self.loop_closure_marker_size
        vertices_marker.scale.y = self.loop_closure_marker_size
        vertices_marker.scale.z = self.loop_closure_marker_size
        vertices_marker.color.r = 1
        vertices_marker.color.g = 0
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        u = path[0]
        v = path[-1]
        ux, uy, _ = graph.get_vertex(u)['pose_for_visualization']
        vx, vy, _ = graph.get_vertex(v)['pose_for_visualization']
        vertices_marker.points.append(Point(ux, uy, 0.05))
        vertices_marker.points.append(Point(vx, vy, 0.05))
        vertices_marker.points.append(Point(x, y, 0.05))
        loop_closure_msg.markers.append(vertices_marker)

        edges_marker = Marker()
        edges_marker.id = 1
        edges_marker.type = Marker.LINE_LIST
        edges_marker.header.frame_id = self.map_frame
        edges_marker.header.stamp = self.get_clock().now().to_msg()
        edges_marker.scale.x = self.loop_closure_edge_marker_size
        edges_marker.color.r = 0
        edges_marker.color.g = 1
        edges_marker.color.b = 1
        edges_marker.color.a = 0.5
        edges_marker.pose.orientation.w = 1
        for i in range(1, len(path)):
            ux, uy, _ = graph.get_vertex(path[i - 1])['pose_for_visualization']
            vx, vy, _ = graph.get_vertex(path[i])['pose_for_visualization']
            edges_marker.points.append(Point(ux, uy, 0.05))
            edges_marker.points.append(Point(vx, vy, 0.05))
        ux, uy, _ = graph.get_vertex(u)['pose_for_visualization']
        edges_marker.points.append(Point(ux, uy, 0.05))
        edges_marker.points.append(Point(x, y, 0.05))
        vx, vy, _ = graph.get_vertex(v)['pose_for_visualization']
        edges_marker.points.append(Point(vx, vy, 0.05))
        edges_marker.points.append(Point(x, y, 0.05))
        loop_closure_msg.markers.append(edges_marker)
        self.loop_closure_results_publisher.publish(loop_closure_msg)

    def publish_local_grid(self, local_grid):
        local_grid_msg = OccupancyGrid()
        local_grid_msg.header.stamp = self.current_stamp
        local_grid_msg.header.frame_id = 'vcur'
        resolution = local_grid.resolution
        local_grid_msg.info.resolution = resolution
        local_grid_msg.info.width = local_grid.grid_size
        local_grid_msg.info.height = local_grid.grid_size
        local_grid_msg.info.origin.position.x = -local_grid.grid_size * resolution / 2
        local_grid_msg.info.origin.position.y = -local_grid.grid_size * resolution / 2
        local_grid_msg.info.origin.orientation.x = 0
        local_grid_msg.info.origin.orientation.y = 0
        local_grid_msg.info.origin.orientation.z = 0
        local_grid_msg.info.origin.orientation.w = 1
        local_map = local_grid.layers['occupancy'].T.ravel().astype(np.int8)
        local_map[local_map >= 2] = 100
        local_map[local_map == 0] = -1
        local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.local_grid_publisher.publish(local_grid_msg)

    def publish_cur_grid(self, cur_grid):
        local_grid_msg = OccupancyGrid()
        local_grid_msg.header.stamp = self.current_stamp
        local_grid_msg.header.frame_id = 'current_state'
        resolution = cur_grid.resolution
        local_grid_msg.info.resolution = resolution
        local_grid_msg.info.width = cur_grid.grid_size
        local_grid_msg.info.height = cur_grid.grid_size
        local_grid_msg.info.origin.position.x = -cur_grid.grid_size * resolution / 2
        local_grid_msg.info.origin.position.y = -cur_grid.grid_size * resolution / 2
        local_grid_msg.info.origin.orientation.x = 0
        local_grid_msg.info.origin.orientation.y = 0
        local_grid_msg.info.origin.orientation.z = 0
        local_grid_msg.info.origin.orientation.w = 1
        local_map = cur_grid.layers['density_map'].T.ravel().astype(np.int8)
        threshold = 7
        local_map[local_map >= threshold] = 100
        local_map[local_map < threshold] = 0
        #local_map[local_map == 0] = -1
        #local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.cur_grid_publisher.publish(local_grid_msg)

    def publish_rel_pose(self, rel_pose_of_vcur):
        rel_pose_msg = PoseStamped()
        rel_pose_msg.header.stamp = self.get_clock().now().to_msg()
        rel_pose_msg.header.frame_id = 'last_vertex'
        rel_x, rel_y, rel_theta = rel_pose_of_vcur
        rel_pose_msg.pose.position.x = rel_x
        rel_pose_msg.pose.position.y = rel_y
        rel_pose_msg.pose.position.z = 0
        orientation = Quaternion()
        orientation.w, orientation.x, orientation.y, orientation.z = tf.quaternion_from_euler(0, 0, rel_theta)
        rel_pose_msg.pose.orientation = orientation
        self.rel_pose_of_vcur_publisher.publish(rel_pose_msg)
        self.tfbr.sendTransform((rel_x, rel_y, 0), 
                                 tf.quaternion_from_euler(0, 0, rel_theta),
                                 self.current_stamp,
                                 "current_state",
                                 "vcur")

    def publish_topological_path(self, graph, path):
        # Publish topological path
        path_msg = TopologicalPath()
        path_msg.header.stamp = self.current_stamp
        for v_id in path:
            node_msg = Int32()
            node_msg.data = v_id
            path_msg.nodes.append(v_id)
        for i in range(len(path) - 1):
            rel_x, rel_y, rel_theta = graph.get_edge(path[i], path[i + 1])
            point_msg = Point()
            point_msg.x = rel_x
            point_msg.y = rel_y
            point_msg.z = 0
            path_msg.rel_poses.append(point_msg)
        self.path_publisher.publish(path_msg)
        # Publish path marker msg
        path_marker_msg = Marker()
        path_marker_msg.header.stamp = self.current_stamp
        path_marker_msg.header.frame_id = self.map_frame
        path_marker_msg.ns = 'points_and_lines'
        path_marker_msg.action = Marker.ADD
        path_marker_msg.pose.orientation.w = 1.0
        path_marker_msg.type = 4
        path_marker_msg.scale.x = self.path_marker_size
        path_marker_msg.scale.y = self.path_marker_size
        path_marker_msg.color.a = 0.8
        path_marker_msg.color.r = 1.0
        path_marker_msg.color.g = 1.0
        path_marker_msg.color.b = 0
        for v_id in path:
            pos = graph.get_vertex(v_id)['pose_for_visualization']
            pt = Point(pos[0], pos[1], 0.2)
            path_marker_msg.points.append(pt)
        self.path_marker_publisher.publish(path_marker_msg)

    def publish_subgoal(self, subgoal):
        x, y, theta = subgoal
        self.tfbr.sendTransform((x, y, 0), 
                                 tf.quaternion_from_euler(0, 0, theta),
                                 self.current_stamp,
                                 "next_state",
                                 "current_state")
        pointgoal_msg = PoseStamped()
        pointgoal_msg.header.stamp = self.current_stamp
        pointgoal_msg.header.frame_id = 'current_state'
        pointgoal_msg.pose.position.x = x
        pointgoal_msg.pose.position.y = y
        pointgoal_msg.pose.position.z = 1
        self.pointgoal_publisher.publish(pointgoal_msg)

    def publish_tf_from_odom(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, __, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.tfbr.sendTransform((x, y, 0),
                                tf.quaternion_from_euler(0, 0, theta),
                                msg.header.stamp,
                                msg.child_frame_id,
                                msg.header.frame_id)

    def freeze(self):
        freeze_msg = Bool()
        freeze_msg.data = True
        self.freeze_publisher.publish(freeze_msg)

    def unfreeze(self):
        freeze_msg = Bool()
        freeze_msg.data = False
        self.freeze_publisher.publish(freeze_msg)

class PRISMTopomapNode(Node):
    def __init__(self):
        super().__init__('prism_topomap_node')
        self.get_logger().info(f'File: {__file__}')
        # Parameters
        self.declare_parameter('path_to_gt_map', None)
        self.declare_parameter('path_to_load_graph', None)
        self.declare_parameter('path_to_save_graph', None)
        self.declare_parameter('path_to_save_logs', None)
        self.declare_parameter('config_file', 'config.yaml')  # Default config file name
        self.path_to_gt_map = self.get_parameter('path_to_gt_map').value
        self.path_to_load_graph = self.get_parameter('path_to_load_graph').value
        self.path_to_save_graph = self.get_parameter('path_to_save_graph').value
        self.path_to_save_logs = self.get_parameter('path_to_save_logs').value
        config_file_name = self.get_parameter('config_file').value
        # Load config
        package_share_dir = get_package_share_directory('prism_topomap')
        config_file = os.path.join(package_share_dir, 'config', config_file_name)
        with open(config_file, 'r') as fin:
            self.config = yaml.safe_load(fin)
        self.toposlam_model = TopoSLAMModel(
            self.config,
            path_to_load_graph=self.path_to_load_graph,
            path_to_save_graph=self.path_to_save_graph,
            path_to_save_logs=self.path_to_save_logs,
        )
        self.gt_poses = []
        self.odom_poses = []
        self.curb_clouds = []
        self.rgb_buffer_front = deque(maxlen=100)
        self.rgb_buffer_back = deque(maxlen=100)
        self.cv_bridge = CvBridge()
        self.cur_stamp = None
        self.metric_goal = None
        self.path_to_goal = None
        self.init_publishers_and_subscribers(self.config)
        if self.publish_gt_map_flag and self.path_to_gt_map is None:
            self.get_logger().error('Path to gt map is not set but publish_gt_map is set true')
            raise RuntimeError('Path to gt map is not set but publish_gt_map is set true')
        self.current_stamp = None
        # Create timers
        self.localization_timer = self.create_timer(
            1.0 / self.localization_frequency,
            self.localize
        )
        
        # Uncomment if needed:
        # self.rel_pose_correction_timer = self.create_timer(
        #     1.0 / self.rel_pose_correction_frequency,
        #     self.toposlam_model.correct_rel_pose
        # )

    def init_publishers_and_subscribers(self, config):
        # Visualization
        visualization_config = config['visualization']
        self.results_publisher = ResultsPublisher(visualization_config['map_frame'])
        self.results_publisher.vertex_marker_size = visualization_config['vertex_marker_size']
        self.results_publisher.edge_marker_size = visualization_config['edge_marker_size']
        self.results_publisher.match_marker_size = visualization_config['match_marker_size']
        self.results_publisher.text_marker_size = visualization_config['text_marker_size']
        self.results_publisher.vertex_orientation_marker_size = visualization_config['vertex_orientation_marker_size']
        self.results_publisher.vcur_marker_size = visualization_config['vcur_marker_size']
        self.results_publisher.loop_closure_marker_size = visualization_config['loop_closure_marker_size']
        self.results_publisher.loop_closure_edge_marker_size = visualization_config['loop_closure_edge_marker_size']
        self.results_publisher.path_marker_size = visualization_config['path_marker_size']
        self.publish_gt_map_flag = visualization_config['publish_gt_map']
        if self.publish_gt_map_flag:
            self.results_publisher.init_gt_map(self.path_to_gt_map)
        input_config = config['input']
        self.publish_tf_from_odom = visualization_config['publish_tf_from_odom']
        # Point cloud
        pointcloud_config = input_config['pointcloud']
        pointcloud_topic = pointcloud_config['topic']
        self.pcd_fields = pointcloud_config['fields']
        self.pcd_rotation = np.array(pointcloud_config['rotation_matrix'])
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pcd_subscriber = self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.pcd_callback,
            qos_profile
        )
        self.use_curb_detection = pointcloud_config['subscribe_to_curbs']
        if self.use_curb_detection:
            curb_detection_topic = pointcloud_config['curb_detection_topic']
            self.curb_detection_subscriber = self.create_subscription(
                PointCloud2,
                curb_detection_topic,
                self.curb_detection_callback,
                qos_profile
            )
        # Odometry
        odometry_config = input_config['odometry']
        odometry_topic = odometry_config['topic']
        self.odom_subscriber = self.create_subscription(
            Odometry,
            odometry_topic,
            self.odom_callback,
            10
        )
        # GT pose for visualization and evaluation
        self.use_gt_pose = input_config['subscribe_to_gt_pose']
        if self.use_gt_pose:
            gt_pose_config = input_config['gt_pose']
            pose_topic = gt_pose_config['topic']
            topic_type = gt_pose_config['type']
            if topic_type == 'PoseStamped':
                self.pose_subscriber = self.create_subscription(
                    PoseStamped,
                    pose_topic,
                    self.gt_pose_callback,
                    10
                )
            else:
                self.pose_subscriber = self.create_subscription(
                    Odometry,
                    pose_topic,
                    self.gt_odom_pose_callback,
                    10
                )
        # Images
        self.use_images = input_config['subscribe_to_images']
        if self.use_images:
            front_image_config = input_config['image_front']
            front_image_topic = front_image_config['topic']
            self.front_image_subscriber = self.create_subscription(
                Image,
                front_image_topic,
                self.front_image_callback,
                10
            )
            back_image_config = input_config['image_back']
            back_image_topic = back_image_config['topic']
            self.back_image_subscriber = self.create_subscription(
                Image,
                back_image_topic,
                self.back_image_callback,
                10
            )
        # Navigation
        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        # Localization and correction frequency
        topomap_config = config['topomap']
        self.localization_frequency = topomap_config['localization_frequency']
        self.rel_pose_correction_frequency = topomap_config['rel_pose_correction_frequency']

    def localize(self):
        self.results_publisher.freeze()
        self.toposlam_model.localizer.localize()
        localized_state = self.toposlam_model.localizer.get_localized_state()
        vertex_ids = localized_state['vertex_ids_matched']
        if vertex_ids is None:
            return
        rel_poses = localized_state['rel_poses']
        vertex_ids_unmatched = localized_state['vertex_ids_unmatched']
        localized_time = localized_state['timestamp']
        
        if self.toposlam_model.found_loop_closure:
            self.results_publisher.publish_loop_closure_results(
                self.toposlam_model.graph,
                self.toposlam_model.path,
                self.toposlam_model.last_vertex['pose_for_visualization']
            )
        self.results_publisher.publish_localization_results(
            self.toposlam_model.graph, 
            vertex_ids, 
            rel_poses, 
            vertex_ids_unmatched
        )
        self.results_publisher.unfreeze()
        timestamp = self.get_clock().now().to_msg()
        timestamp.sec = int(localized_time)
        timestamp.nanosec = int((localized_time - int(localized_time)) * 1e9)
        self.results_publisher.publish_ref_cloud(self.cur_cloud, timestamp)

    def gt_pose_callback(self, msg):
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        orientation = msg.pose.orientation
        _, __, theta = tf_transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        self.gt_poses.append([float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9, [x, y, theta]])

    def gt_odom_pose_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, __, theta = tf_transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        self.gt_poses.append([float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9, [x, y, theta]])

    def odom_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, __, theta = tf_transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        if self.publish_tf_from_odom:
            self.results_publisher.publish_tf_from_odom(msg)
        self.odom_poses.append([float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9, [x, y, theta]])

    def front_image_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg)[:, :, :3]
        self.rgb_buffer_front.append([float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9, image])

    def back_image_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg)[:, :, :3]
        self.rgb_buffer_back.append([float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9, image])

    def goal_callback(self, msg):
        self.metric_goal = (msg.pose.position.x, msg.pose.position.y)
        self.path_to_goal = self.toposlam_model.get_path_to_metric_goal(*self.metric_goal)

    def interpolate_pose(self, pose_left_stamped, pose_right_stamped, timestamp):
        alpha = (timestamp - pose_left_stamped[0]) / (pose_right_stamped[0] - pose_left_stamped[0])
        pose_left = np.array(pose_left_stamped[1])
        pose_right = np.array(pose_right_stamped[1])
        pose_left[-1] = normalize(pose_left[-1])
        pose_right[-1] = normalize(pose_right[-1])
        result = alpha * pose_right + (1 - alpha) * pose_left
        if abs(pose_left[2] - pose_right[2]) > np.pi:
            if pose_left[2] < 0:
                pose_left[2] += 2 * np.pi
            if pose_right[2] < 0:
                pose_right[2] += 2 * np.pi
            result = alpha * pose_right + (1 - alpha) * pose_left
            self.get_logger().info(f'Interpolation: {pose_left}, {pose_right}, {result}')
        return result

    def get_sync_pose_and_images(self, timestamp, arrays, is_pose_array):
        result = []
        delta = 0.05
        for array, is_pose in zip(arrays, is_pose_array):
            if len(array) == 0:
                result.append(None)
            else:
                i = 0
                while i < len(array) and array[i][0] < timestamp - delta:
                    i += 1
                if i == len(array):
                    result.append(None)
                elif not is_pose:
                    result.append(array[i][1])
                else:
                    if i == 0:
                        if array[0][0] - timestamp > 0.2:
                            result.append(None)
                        else:
                            result.append(array[0][1])
                    else:
                        if is_pose:
                            result.append(self.interpolate_pose(array[i - 1], array[i], timestamp))
                        else:
                            result.append(array[i][1])
        return result

    def curb_detection_callback(self, msg):
        start_time = time.time()
        cloud = get_xyz_coords_from_msg(msg, "xyz", self.pcd_rotation)
        self.curb_clouds.append([float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9, cloud])

    def get_navigation_subgoal(self):
        vcur = self.toposlam_model.last_vertex_id
        if self.path_to_goal is None:
            self.get_logger().warning('NO PATH TO GOAL!')
            return
        if vcur in self.path_to_goal:
            self.path_to_goal = self.path_to_goal[self.path_to_goal.index(vcur):]
        else:
            self.path_to_goal = self.toposlam_model.get_path_to_metric_goal(*self.metric_goal)
        self.results_publisher.publish_topological_path(self.toposlam_model.graph, self.path_to_goal)
        # Publish navigation subgoal
        if len(self.path_to_goal) <= 2:
            cur_global_pose = apply_pose_shift(
                self.toposlam_model.last_vertex['pose_for_visualization'],
                *self.toposlam_model.rel_pose_of_vcur
            )
            subgoal = get_rel_pose(*cur_global_pose, self.metric_goal[0], self.metric_goal[1], 0)
        else:
            pose_on_edge = self.toposlam_model.graph.get_edge(self.path_to_goal[0], self.path_to_goal[1])
            subgoal_in_vcur_coords = pose_on_edge
            for i in range(1, len(self.path_to_goal) - 1):
                pose_on_edge = self.toposlam_model.graph.get_edge(self.path_to_goal[i], self.path_to_goal[i + 1])
                subgoal_in_vcur_coords_next = apply_pose_shift(subgoal_in_vcur_coords, *pose_on_edge)
                if self.toposlam_model.last_vertex['grid'].is_inside(*subgoal_in_vcur_coords_next):
                    subgoal_in_vcur_coords = subgoal_in_vcur_coords_next
                else:
                    break
            subgoal = get_rel_pose(*self.toposlam_model.rel_pose_of_vcur, *subgoal_in_vcur_coords)
        return subgoal

    def pcd_callback(self, msg):
        if self.publish_gt_map_flag:
            self.results_publisher.publish_gt_map()
        current_time = self.get_clock().now().nanoseconds / 1e9
        msg_time = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9
        dt = current_time - msg_time
        if dt > 10000:
            self.get_logger().warning('Too big message time difference. Probably you have wrong use_sim_time ROS param value')
        if dt < -1000:
            self.get_logger().warning('Too big negative message time difference. Is your clock published?')
        if dt > 0.5 or dt < -0.5:
            return
        cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back, cur_curbs = self.get_sync_pose_and_images(
            msg_time,
            [self.gt_poses, self.odom_poses, self.rgb_buffer_front, self.rgb_buffer_back, self.curb_clouds],
            [True, True, False, False, False]
        )
        start_time = time.time()
        while cur_odom_pose is None or (self.use_images and cur_img_front is None) or (self.use_images and cur_img_back is None):
            cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back, cur_curbs = self.get_sync_pose_and_images(
                msg_time,
                [self.gt_poses, self.odom_poses, self.rgb_buffer_front, self.rgb_buffer_back, self.curb_clouds],
                [True, True, False, False, False]
            )
            time.sleep(0.01)
            if time.time() - start_time > 0.5:
                self.get_logger().warning('Waiting for sync pose and images timed out!')
                return
        self.cur_global_pose = cur_global_pose
        if self.cur_global_pose is None:
            self.get_logger().warning('No global pose!')
            return
        self.current_stamp = msg.header.stamp
        self.results_publisher.current_stamp = msg.header.stamp
        self.toposlam_model.current_stamp = msg_time
        start_time = time.time()
        self.cur_cloud = get_xyz_coords_from_msg(msg, self.pcd_fields, self.pcd_rotation)
        self.toposlam_model.update(cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back, self.cur_cloud, cur_curbs)
        # Publish graph and all the other visualization info
        self.results_publisher.publish_graph(self.toposlam_model.graph)
        self.results_publisher.publish_cur_grid(self.toposlam_model.cur_grid)
        self.results_publisher.publish_local_grid(self.toposlam_model.last_vertex['grid'])
        self.results_publisher.publish_last_vertex(self.toposlam_model.last_vertex, self.toposlam_model.last_vertex_id)
        self.results_publisher.publish_rel_pose(self.toposlam_model.rel_pose_of_vcur)

    def save_graph(self):
        self.toposlam_model.save_graph()

    def run(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    node = PRISMTopomapNode()
    node.run()
    node.save_graph()

if __name__ == '__main__':
    main()
