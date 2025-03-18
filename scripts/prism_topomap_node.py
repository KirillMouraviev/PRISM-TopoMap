#!/usr/bin/env python

import rospy
import rospkg
import numpy as np
np.float = np.float64
import ros_numpy
import os
import cv2
import sys
import tf
import time
import yaml
import copy
from utils import *
from gt_map import GTMap
from prism_topomap import TopoSLAMModel
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32, Int32MultiArray
from toposlam_msgs.msg import TopologicalPath
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque
from cv_bridge import CvBridge

rospy.init_node('prism_topomap_node')

class PRISMTopomapNode():
    def __init__(self):
        print('File:', __file__)
        self.path_to_gt_map = rospy.get_param('~path_to_gt_map', None)
        self.path_to_load_graph = rospy.get_param('~path_to_load_graph', None)
        self.path_to_save_graph = rospy.get_param('~path_to_save_graph', None)
        rospack = rospkg.RosPack()
        config_file = os.path.join(rospack.get_path('prism_topomap'), 'config', rospy.get_param('~config_file'))
        fin = open(config_file, 'r')
        self.config = yaml.safe_load(fin)
        fin.close()

        self.toposlam_model = TopoSLAMModel(self.config,
                                            path_to_load_graph=self.path_to_load_graph,
                                            path_to_save_graph=self.path_to_save_graph,
                                            )

        self.gt_poses = []
        self.odom_poses = []
        self.curb_clouds = []
        self.rgb_buffer_front = deque(maxlen=100)
        self.rgb_buffer_back = deque(maxlen=100)
        self.cv_bridge = CvBridge()
        self.cur_stamp = None
        self.tfbr = tf.TransformBroadcaster()
        self.init_publishers_and_subscribers(self.config)

        if self.publish_gt_map_flag and self.path_to_gt_map is None:
            print('ERROR! Path to gt map is not set but publish_gt_map is set true')
            exit(1)

        self.current_stamp = None
        rospy.Timer(rospy.Duration(self.localization_frequency), self.toposlam_model.localizer.localize)
        # rospy.Timer(rospy.Duration(self.rel_pose_correction_frequency), self.toposlam_model.correct_rel_pose)

    def init_publishers_and_subscribers(self, config):
        # GT map
        visualization_config = config['visualization']
        self.publish_gt_map_flag = visualization_config['publish_gt_map']
        if self.publish_gt_map_flag:
            gt_map_filename = [fn for fn in os.listdir(self.path_to_gt_map) if fn.startswith('map_cropped_')][0]
            self.gt_map = GTMap(os.path.join(self.path_to_gt_map, gt_map_filename))
        else:
            self.gt_map = None
        input_config = config['input']
        # Extra params
        self.map_frame = visualization_config['map_frame']
        self.publish_tf_from_odom = visualization_config['publish_tf_from_odom']
        # Point cloud
        pointcloud_config = input_config['pointcloud']
        pointcloud_topic = pointcloud_config['topic']
        self.pcd_fields = pointcloud_config['fields']
        self.pcd_rotation = np.array(pointcloud_config['rotation_matrix'])
        self.pcd_subscriber = rospy.Subscriber(pointcloud_topic, PointCloud2, self.pcd_callback)
        self.use_curb_detection = pointcloud_config['subscribe_to_curbs']
        if self.use_curb_detection:
            curb_detection_topic = pointcloud_config['curb_detection_topic']
            self.curb_detection_subscriber = rospy.Subscriber(curb_detection_topic, PointCloud2, self.curb_detection_callback)
        # Odometry
        odometry_config = input_config['odometry']
        odometry_topic = odometry_config['topic']
        self.odom_subscriber = rospy.Subscriber(odometry_topic, Odometry, self.odom_callback)
        # GT pose for visualization and evaluation
        self.use_gt_pose = input_config['subscribe_to_gt_pose']
        if self.use_gt_pose:
            gt_pose_config = input_config['gt_pose']
            pose_topic = gt_pose_config['topic']
            topic_type = gt_pose_config['type']
            if topic_type == 'PoseStamped':
                self.pose_subscriber = rospy.Subscriber(pose_topic, PoseStamped, self.gt_pose_callback)
            else:
                self.pose_subscriber = rospy.Subscriber(pose_topic, Odometry, self.gt_odom_pose_callback)
        # Images
        self.use_images = input_config['subscribe_to_images']
        if self.use_images:
            front_image_config = input_config['image_front']
            front_image_topic = front_image_config['topic']
            self.front_image_subscriber = rospy.Subscriber(front_image_topic, Image, self.front_image_callback)
            back_image_config = input_config['image_back']
            back_image_topic = back_image_config['topic']
            self.back_image_subscriber = rospy.Subscriber(back_image_topic, Image, self.back_image_callback)
        # Localization
        self.localization_subscriber = rospy.Subscriber('/localized_nodes', Float32MultiArray, self.localization_callback)
        # Visualization
        self.gt_map_publisher = rospy.Publisher('/habitat/gt_map', OccupancyGrid, latch=True, queue_size=100)
        self.last_vertex_publisher = rospy.Publisher('/last_vertex', Marker, latch=True, queue_size=100)
        self.last_vertex_id_publisher = rospy.Publisher('/last_vertex_id', Int32, latch=True, queue_size=100)
        self.loop_closure_results_publisher = rospy.Publisher('/loop_closure_results', MarkerArray, latch=True, queue_size=100)
        self.rel_pose_of_vcur_publisher = rospy.Publisher('/rel_pose_of_vcur', PoseStamped, latch=True, queue_size=100)
        self.local_grid_publisher = rospy.Publisher('/local_grid', OccupancyGrid, latch=True, queue_size=100)
        self.cur_grid_publisher = rospy.Publisher('/current_grid', OccupancyGrid, latch=True, queue_size=100)
        self.cur_grid_transformed_publisher = rospy.Publisher('/current_grid_transformed', OccupancyGrid, latch=True, queue_size=100)
        self.graph_viz_pub = rospy.Publisher('topological_map', MarkerArray, latch=True, queue_size=100)
        # Localization and correction frequency
        topomap_config = config['topomap']
        self.localization_frequency = topomap_config['localization_frequency']
        self.rel_pose_correction_frequency = topomap_config['rel_pose_correction_frequency']

    def publish_gt_map(self):
        gt_map_msg = OccupancyGrid()
        gt_map_msg.header.stamp = rospy.Time.now()
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

    def gt_pose_callback(self, msg):
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        orientation = msg.pose.orientation
        _, __, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.gt_poses.append([msg.header.stamp.to_sec(), [x, y, theta]])

    def gt_odom_pose_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, __, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.gt_poses.append([msg.header.stamp.to_sec(), [x, y, theta]])

    def odom_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, __, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        if self.publish_tf_from_odom:
            self.tfbr.sendTransform((x, y, 0),
                                    tf.transformations.quaternion_from_euler(0, 0, theta),
                                    msg.header.stamp,
                                    msg.child_frame_id,
                                    msg.header.frame_id)
        self.odom_poses.append([msg.header.stamp.to_sec(), [x, y, theta]])

    def front_image_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg)[:, :, :3]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.rgb_buffer_front.append([msg.header.stamp.to_sec(), image])

    def back_image_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg)[:, :, :3]
        self.rgb_buffer_back.append([msg.header.stamp.to_sec(), image])

    def publish_graph(self):
        # Publish graph for visualization
        graph = self.toposlam_model.graph
        graph_msg = MarkerArray()
        vertices_marker = Marker()
        #vertices_marker = ns = 'points_and_lines'
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = graph.map_frame
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.5
        vertices_marker.scale.y = 0.5
        vertices_marker.scale.z = 0.5
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
        edges_marker.scale.x = 0.2
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
        vertex_orientation_marker.scale.x = 0.05
        vertex_orientation_marker.color.r = 1
        vertex_orientation_marker.color.g = 0
        vertex_orientation_marker.color.b = 0
        vertex_orientation_marker.color.a = 1
        vertex_orientation_marker.pose.orientation.w = 1
        for vertex_dict in graph.vertices:
            x, y, theta = vertex_dict['pose_for_visualization']
            vertex_orientation_marker.points.append(Point(x, y, 0.1))
            vertex_orientation_marker.points.append(Point(x + np.cos(theta) * 0.5, y + np.sin(theta) * 0.5, 0.05))
        # graph_msg.markers.append(vertex_orientation_marker)

        cnt = 3
        text_marker = Marker()
        text_marker.header = vertices_marker.header
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.scale.z = 0.4
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
        text_marker.scale.z = 0.3
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

    def publish_loop_closure_results(self, path, global_pose_for_visualization):
        assert len(path) >= 2
        loop_closure_msg = MarkerArray()
        vertices_marker = Marker()
        x, y, _ = global_pose_for_visualization
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.2
        vertices_marker.scale.y = 0.2
        vertices_marker.scale.z = 0.2
        vertices_marker.color.r = 1
        vertices_marker.color.g = 0
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        u = path[0]
        v = path[-1]
        ux, uy, _ = self.toposlam_model.graph.get_vertex(u)['pose_for_visualization']
        vx, vy, _ = self.toposlam_model.graph.get_vertex(v)['pose_for_visualization']
        vertices_marker.points.append(Point(ux, uy, 0.05))
        vertices_marker.points.append(Point(vx, vy, 0.05))
        vertices_marker.points.append(Point(x, y, 0.05))
        loop_closure_msg.markers.append(vertices_marker)

        edges_marker = Marker()
        edges_marker.id = 1
        edges_marker.type = Marker.LINE_LIST
        edges_marker.header.frame_id = self.map_frame
        edges_marker.header.stamp = rospy.Time.now()
        edges_marker.scale.x = 0.15
        edges_marker.color.r = 0
        edges_marker.color.g = 1
        edges_marker.color.b = 1
        edges_marker.color.a = 0.5
        edges_marker.pose.orientation.w = 1
        for i in range(1, len(path)):
            ux, uy, _ = self.toposlam_model.graph.get_vertex(path[i - 1])['pose_for_visualization']
            vx, vy, _ = self.toposlam_model.graph.get_vertex(path[i])['pose_for_visualization']
            edges_marker.points.append(Point(ux, uy, 0.05))
            edges_marker.points.append(Point(vx, vy, 0.05))
        ux, uy, _ = self.toposlam_model.graph.get_vertex(u)['pose_for_visualization']
        edges_marker.points.append(Point(ux, uy, 0.05))
        edges_marker.points.append(Point(x, y, 0.05))
        vx, vy, _ = self.toposlam_model.graph.get_vertex(v)['pose_for_visualization']
        edges_marker.points.append(Point(vx, vy, 0.05))
        edges_marker.points.append(Point(x, y, 0.05))
        loop_closure_msg.markers.append(edges_marker)
        self.loop_closure_results_publisher.publish(loop_closure_msg)

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
            print('Interpolation:', pose_left, pose_right, result)
        return result

    def get_sync_pose_and_images(self, timestamp, arrays, is_pose_array):
        result = []
        for array, is_pose in zip(arrays, is_pose_array):
            if len(array) == 0:
                result.append(None)
            else:
                i = 0
                while i < len(array) and array[i][0] < timestamp:
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

    def publish_last_vertex(self):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = self.map_frame
        marker_msg.type = Marker.SPHERE
        last_x, last_y, last_theta = self.toposlam_model.last_vertex['pose_for_visualization']
        marker_msg.pose.position.x = last_x
        marker_msg.pose.position.y = last_y
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.w = 1.0
        marker_msg.color.r = 0
        marker_msg.color.g = 1
        marker_msg.color.b = 0
        marker_msg.color.a = 1
        marker_msg.scale.x = 0.8
        marker_msg.scale.y = 0.8
        marker_msg.scale.z = 0.8
        self.last_vertex_publisher.publish(marker_msg)
        vertex_id_msg = Int32()
        vertex_id_msg.data = self.toposlam_model.last_vertex_id
        self.last_vertex_id_publisher.publish(vertex_id_msg)
        self.tfbr.sendTransform((last_x, last_y, 0),
                                    tf.transformations.quaternion_from_euler(0, 0, last_theta),
                                    self.current_stamp,
                                    "vcur",
                                    self.map_frame)

    def publish_local_grid(self):
        local_grid = self.toposlam_model.last_vertex['grid'].grid
        local_grid_msg = OccupancyGrid()
        local_grid_msg.header.stamp = self.current_stamp
        local_grid_msg.header.frame_id = 'vcur'
        resolution = self.toposlam_model.grid_resolution
        local_grid_msg.info.resolution = resolution
        local_grid_msg.info.width = local_grid.shape[1]
        local_grid_msg.info.height = local_grid.shape[0]
        local_grid_msg.info.origin.position.x = -local_grid.shape[1] * resolution / 2
        local_grid_msg.info.origin.position.y = -local_grid.shape[0] * resolution / 2
        local_grid_msg.info.origin.orientation.x = 0
        local_grid_msg.info.origin.orientation.y = 0
        local_grid_msg.info.origin.orientation.z = 0
        local_grid_msg.info.origin.orientation.w = 1
        local_map = local_grid.T.ravel().astype(np.int8)
        local_map[local_map >= 2] = 100
        local_map[local_map == 0] = -1
        local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.local_grid_publisher.publish(local_grid_msg)

    def publish_cur_grid(self):
        local_grid_msg = OccupancyGrid()
        local_grid_msg.header.stamp = self.current_stamp
        local_grid_msg.header.frame_id = 'current_state'
        resolution = self.toposlam_model.grid_resolution
        cur_grid = self.toposlam_model.cur_grid
        local_grid_msg.info.resolution = resolution
        local_grid_msg.info.width = cur_grid.grid.shape[1]
        local_grid_msg.info.height = cur_grid.grid.shape[0]
        local_grid_msg.info.origin.position.x = -cur_grid.grid.shape[1] * resolution / 2
        local_grid_msg.info.origin.position.y = -cur_grid.grid.shape[0] * resolution / 2
        local_grid_msg.info.origin.orientation.x = 0
        local_grid_msg.info.origin.orientation.y = 0
        local_grid_msg.info.origin.orientation.z = 0
        local_grid_msg.info.origin.orientation.w = 1
        local_map = cur_grid.grid.T.ravel().astype(np.int8)
        local_map[local_map > 2] = 100
        local_map[local_map == 0] = -1
        local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.cur_grid_publisher.publish(local_grid_msg)
        
        rel_x_old, rel_y_old, rel_theta_old = self.toposlam_model.get_rel_pose_from_stamp(self.toposlam_model.cur_stamp)[0]
        cur_grid_transformed = cur_grid.get_transformed_grid(rel_x_old, rel_y_old, -rel_theta_old)
        local_grid_msg.header.frame_id = 'vcur'
        resolution = 0.1
        local_grid_msg.info.resolution = resolution
        local_grid_msg.info.width = cur_grid_transformed.shape[1]
        local_grid_msg.info.height = cur_grid_transformed.shape[0]
        local_grid_msg.info.origin.position.x = -cur_grid_transformed.shape[1] * resolution / 2
        local_grid_msg.info.origin.position.y = -cur_grid_transformed.shape[0] * resolution / 2
        local_grid_msg.info.origin.orientation.x = 0
        local_grid_msg.info.origin.orientation.y = 0
        local_grid_msg.info.origin.orientation.z = 0
        local_grid_msg.info.origin.orientation.w = 1
        local_map = cur_grid_transformed.T.ravel().astype(np.int8)
        local_map[local_map >= 2] = 100
        local_map[local_map == 0] = -1
        local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.cur_grid_transformed_publisher.publish(local_grid_msg)

    def publish_rel_pose(self):
        rel_pose_msg = PoseStamped()
        rel_pose_msg.header.stamp = rospy.Time.now()
        rel_pose_msg.header.frame_id = 'last_vertex'
        rel_x, rel_y, rel_theta = self.toposlam_model.rel_pose_of_vcur
        rel_pose_msg.pose.position.x = rel_x
        rel_pose_msg.pose.position.y = rel_y
        rel_pose_msg.pose.position.z = 0
        orientation = Quaternion()
        orientation.w, orientation.x, orientation.y, orientation.z = tf.transformations.quaternion_from_euler(0, 0, rel_theta)
        rel_pose_msg.pose.orientation = orientation
        self.rel_pose_of_vcur_publisher.publish(rel_pose_msg)
        self.tfbr.sendTransform((rel_x, rel_y, 0), 
                                 tf.transformations.quaternion_from_euler(0, 0, rel_theta),
                                 self.current_stamp,
                                 "current_state",
                                 "vcur")

    def localization_callback(self, msg):
        self.toposlam_model.update_from_localization(msg)
        if self.toposlam_model.found_loop_closure:
            self.publish_loop_closure_results(self.toposlam_model.path, self.toposlam_model.last_vertex['global_pose_for_visualization'])

    def curb_detection_callback(self, msg):
        cloud = get_xyz_coords_from_msg(msg, "xyz", self.pcd_rotation)
        self.curb_clouds.append([msg.header.stamp.to_sec(), cloud])

    def pcd_callback(self, msg):
        if self.publish_gt_map_flag:
            self.publish_gt_map()
        dt = (rospy.Time.now() - msg.header.stamp).to_sec()
        # print('Msg lag is {} seconds'.format(dt))
        if dt > 10000:
            print('Too big message time difference. Probably you have wrong use_sim_time ROS param value')
        if dt < -1000:
            print('Too big negative message time difference. Is your clock published?')
        if dt > 0.5 or dt < -0.5:
            return
        cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back, cur_curbs = self.get_sync_pose_and_images(msg.header.stamp.to_sec(),
            [self.gt_poses, self.odom_poses, self.rgb_buffer_front, self.rgb_buffer_back, self.curb_clouds],
            [True, True, False, False, False])
        start_time = rospy.Time.now().to_sec()
        while cur_odom_pose is None:
            cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back, cur_curbs = self.get_sync_pose_and_images(msg.header.stamp.to_sec(),
                [self.gt_poses, self.odom_poses, self.rgb_buffer_front, self.rgb_buffer_back, self.curb_clouds],
                [True, True, False, False, False])
            rospy.sleep(1e-2)
            if rospy.Time.now().to_sec() - start_time > 0.5:
                print('Waiting for sync pose and images timed out!')
                return
        self.cur_global_pose = cur_global_pose
        if self.cur_global_pose is None:
            print('No global pose!')
            return
        
        # print('Cur odom pose:', cur_odom_pose)
        self.current_stamp = msg.header.stamp
        self.toposlam_model.current_stamp = self.current_stamp

        cur_cloud = get_xyz_coords_from_msg(msg, self.pcd_fields, self.pcd_rotation)
        start_time = time.time()
        self.toposlam_model.update_by_iou(cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back, cur_cloud, cur_curbs, msg.header.stamp)

        # Publish graph and all the other visualization info
        self.publish_graph()
        self.publish_cur_grid()
        self.publish_local_grid()
        self.publish_last_vertex()
        self.publish_rel_pose()
        # print('Update by iou time:', time.time() - start_time)

    def save_graph(self):
        self.toposlam_model.save_graph()

    def run(self):
        rospy.spin()


node = PRISMTopomapNode()
node.run()
node.save_graph()