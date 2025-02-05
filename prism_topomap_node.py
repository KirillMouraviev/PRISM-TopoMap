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
from localization import Localizer
from utils import *
from gt_map import GTMap
from topo_graph import TopologicalGraph
from local_grid import LocalGrid
from models import get_place_recognition_model, get_registration_model
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32, Int32MultiArray
from toposlam_msgs.msg import TopologicalPath
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque
from cv_bridge import CvBridge
from skimage.io import imsave
from threading import Lock

from memory_profiler import profile

rospy.init_node('prism_topomap_node')

class TopoSLAMModel():
    def __init__(self):
        print('File:', __file__)
        self.path_to_gt_map = rospy.get_param('~path_to_gt_map')
        self.path_to_save_json = rospy.get_param('~path_to_save_json')
        rospack = rospkg.RosPack()
        config_file = os.path.join(rospack.get_path('prism_topomap'), 'config', rospy.get_param('~config_file'))
        fin = open(config_file, 'r')
        self.config = yaml.safe_load(fin)
        fin.close()
        self.init_params_from_config(self.config)

        self.last_vertex = None
        self.last_vertex_id = None
        self.prev_img_front = None
        self.prev_img_back = None
        self.prev_cloud = None
        self.prev_pose_for_visualization = None
        self.prev_rel_pose = None
        self.in_sight_response = None
        self.cur_global_pose = None
        self.gt_poses = []
        self.odom_poses = []
        self.rgb_buffer_front = deque(maxlen=100)
        self.rgb_buffer_back = deque(maxlen=100)
        self.cv_bridge = CvBridge()
        self.pose_pairs = []
        self.cur_grid = None
        self.cur_stamp = None
        self.rel_poses_stamped = []
        self.odom_pose = None
        self.rel_pose_of_vcur = None
        self.rel_pose_vcur_to_loc = None
        self.tfbr = tf.TransformBroadcaster()

        self.path = []

        self.graph = TopologicalGraph(place_recognition_model=self.place_recognition_model,
                                      place_recognition_index=self.place_recognition_index,
                                      registration_model=self.registration_model,
                                      inline_registration_model=self.inline_registration_model,
                                      map_frame=self.map_frame,
                                      registration_score_threshold=self.registration_score_threshold,
                                      inline_registration_score_threshold=self.inline_registration_score_threshold,
                                      grid_resolution=self.grid_resolution,
                                      grid_radius=self.grid_radius,
                                      max_grid_range=self.max_grid_range,
                                      floor_height=self.floor_height, ceil_height=self.ceil_height)
        # self.graph.load_from_json(self.path_to_save_json)
        self.localizer = Localizer(self.graph, self.gt_map, map_frame=self.map_frame, top_k=self.top_k)
        self.init_publishers_and_subscribers(self.config)

        self.localization_time = 0
        self.cur_iou = 0
        self.localization_results = ([], [])
        self.edge_reattach_cnt = 0
        self.rel_pose_cnt = 0
        self.iou_cnt = 0
        self.local_grid_cnt = 0
        self.current_stamp = None
        rospy.Timer(rospy.Duration(self.localization_frequency), self.localizer.localize)
        rospy.Timer(rospy.Duration(self.rel_pose_correction_frequency), self.correct_rel_pose)
        self.mutex = Lock()

    def init_params_from_config(self, config):
        # TopoMap
        topomap_config = config['topomap']
        self.iou_threshold = topomap_config['iou_threshold']
        self.localization_frequency = topomap_config['localization_frequency']
        self.rel_pose_correction_frequency = topomap_config['rel_pose_correction_frequency']
        self.max_edge_length = topomap_config['max_edge_length']
        # Input
        pointcloud_config = config['input']['pointcloud']
        self.floor_height = pointcloud_config['floor_height']
        self.ceil_height = pointcloud_config['ceiling_height']
        # Place recognition
        place_recognition_config = config['place_recognition']
        self.place_recognition_model, self.place_recognition_index = get_place_recognition_model(place_recognition_config)
        self.top_k = place_recognition_config['top_k']
        # Local grid
        grid_config = config['local_occupancy_grid']
        self.grid_resolution = grid_config['resolution']
        self.grid_radius = grid_config['radius']
        self.max_grid_range = grid_config['max_range']
        # Registration
        registration_config = config['scan_matching']
        self.registration_model = get_registration_model(registration_config)
        self.registration_score_threshold = registration_config['score_threshold']
        inline_registration_config = config['scan_matching_along_edge']
        self.inline_registration_model = get_registration_model(inline_registration_config)
        self.inline_registration_score_threshold = inline_registration_config['score_threshold']
        # Visualization
        visualization_config = config['visualization']
        self.publish_gt_map_flag = visualization_config['publish_gt_map']
        if self.publish_gt_map_flag:
            gt_map_filename = [fn for fn in os.listdir(self.path_to_gt_map) if fn.startswith('map_cropped_')][0]
            self.gt_map = GTMap(os.path.join(self.path_to_gt_map, gt_map_filename))
        else:
            self.gt_map = None
        self.map_frame = visualization_config['map_frame']
        self.publish_tf_from_odom = visualization_config['publish_tf_from_odom']

    def init_publishers_and_subscribers(self, config):
        input_config = config['input']
        # Point cloud
        pointcloud_config = input_config['pointcloud']
        pointcloud_topic = pointcloud_config['topic']
        self.pcd_fields = pointcloud_config['fields']
        self.pcd_rotation = np.array(pointcloud_config['rotation_matrix'])
        self.pcd_subscriber = rospy.Subscriber(pointcloud_topic, PointCloud2, self.pcd_callback)
        # Odometry
        odometry_config = input_config['odometry']
        odometry_topic = odometry_config['topic']
        self.odom_subscriber = rospy.Subscriber(odometry_topic, Odometry, self.odom_callback)
        # GT pose for visualization and evaluation
        gt_pose_config = input_config['gt_pose']
        pose_topic = gt_pose_config['topic']
        topic_type = gt_pose_config['type']
        if topic_type == 'PoseStamped':
            self.pose_subscriber = rospy.Subscriber(pose_topic, PoseStamped, self.gt_pose_callback)
        else:
            self.pose_subscriber = rospy.Subscriber(pose_topic, Odometry, self.gt_odom_pose_callback)
        # Images
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
        self.gt_poses.append([msg.header.stamp.to_sec(), x, y, theta])

    def gt_odom_pose_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, __, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.gt_poses.append([msg.header.stamp.to_sec(), x, y, theta])

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
        self.odom_poses.append([msg.header.stamp.to_sec(), x, y, theta])

    def front_image_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg)[:, :, :3]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.rgb_buffer_front.append([msg.header.stamp.to_sec(), image])

    def back_image_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg)[:, :, :3]
        self.rgb_buffer_back.append([msg.header.stamp.to_sec(), image])

    def correct_rel_pose(self, event=None):
        if self.last_vertex_id is None:
            return
        if self.cur_grid is None:
            return
        self.mutex.acquire()
        rel_x_old, rel_y_old, rel_theta_old = self.get_rel_pose_from_stamp(self.cur_stamp)[0]
        cur_grid_transformed = self.cur_grid.copy()
        cur_grid_transformed.transform(rel_x_old, rel_y_old, -rel_theta_old)
        x, y, theta = self.graph.get_transform_to_vertex(self.last_vertex_id, cur_grid_transformed)
        true_rel_pose = get_rel_pose(*self.last_vertex['pose_for_visualization'], *self.cur_global_pose)
        # print('True rel pose:', true_rel_pose)
        # print('Old rel pose:', rel_x_old, rel_y_old, rel_theta_old)
        # print('Rel pose of vcur:', self.rel_pose_of_vcur)
        # print('True correction:', get_rel_pose(*self.rel_pose_of_vcur, *true_rel_pose))
        # print('Found pose:', x, y, theta)
        if x is not None and np.sqrt(x ** 2 + y ** 2) < 0.5:
            a_inv_v_inv_n = apply_pose_shift((x, y, theta), *self.rel_pose_of_vcur)
            corrected_pose = a_inv_v_inv_n#apply_pose_shift(self.rel_pose_of_vcur, est_x, est_y, theta)
            # print('Corrected pose:', corrected_pose)
            self.rel_pose_of_vcur = corrected_pose
        # else:
            # print('Failed to correct rel pose!')
        self.mutex.release()

    def check_path_condition(self, u, v, estimated_transform=None):
        print('Checking path condition between {} and {}'.format(u, v))
        path, path_length = self.graph.get_path_with_length(u, v)
        if path is None:
            return True
        # print('Path:', path)
        # print('Path length:', path_length)
        # print('Node positions:')
        #for v in path:
        #    print(self.graph.get_vertex(v)['pose_for_visualization'])
        rel_pose_along_path = [0, 0, 0]
        for i in range(1, len(path)):
            rel_pose_along_path = apply_pose_shift(rel_pose_along_path, *self.graph.get_edge(path[i - 1], path[i]))
        # if path_length < 8:
        #     return True
        print('Path length to vertex {} is {}'.format(v, path_length))

        straight_length = np.sqrt(rel_pose_along_path[0] ** 2 + rel_pose_along_path[1] ** 2)
        print('Straight length:', straight_length)
        if path_length > 3 * straight_length or straight_length < 5:
            # if estimated_transform is not None:
            #     print('Rel pose along path:', rel_pose_along_path)
            #     print('Estimated transform:', estimated_transform)
            #     diff = np.array(estimated_transform) - np.array(rel_pose_along_path)
            #     error = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            #     print('Abs diff:', error)
            #     print('Rel diff:', error / straight_length)
            #     return error < 3 or error / straight_length < 0.6 + 0.5 - 1 / len(path)
            return True
        return False

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
        ux, uy, _ = self.graph.get_vertex(u)['pose_for_visualization']
        vx, vy, _ = self.graph.get_vertex(v)['pose_for_visualization']
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
            ux, uy, _ = self.graph.get_vertex(path[i - 1])['pose_for_visualization']
            vx, vy, _ = self.graph.get_vertex(path[i])['pose_for_visualization']
            edges_marker.points.append(Point(ux, uy, 0.05))
            edges_marker.points.append(Point(vx, vy, 0.05))
        ux, uy, _ = self.graph.get_vertex(u)['pose_for_visualization']
        edges_marker.points.append(Point(ux, uy, 0.05))
        edges_marker.points.append(Point(x, y, 0.05))
        vx, vy, _ = self.graph.get_vertex(v)['pose_for_visualization']
        edges_marker.points.append(Point(vx, vy, 0.05))
        edges_marker.points.append(Point(x, y, 0.05))
        loop_closure_msg.markers.append(edges_marker)
        self.loop_closure_results_publisher.publish(loop_closure_msg)

    def find_loop_closure(self, vertex_ids, dists, global_pose_for_visualization):
        found_loop_closure = False
        for i in range(len(vertex_ids)):
            for j in range(len(vertex_ids)):
                u = vertex_ids[i]
                v = vertex_ids[j]
                if u < 0 or v < 0:
                    continue
                path, path_len = self.graph.get_path_with_length(u, v)
                if path is None:
                    continue
                dst_through_cur = dists[i] + dists[j]
                print('Path len: {}, dst through cur: {}'.format(path_len, dst_through_cur))
                if path_len > 5 and path_len > 2 * dst_through_cur and self.check_path_condition(u, v):
                    #ux, uy, _ = self.graph.get_vertex(u)['pose_for_visualization']
                    #vx, vy, _ = self.graph.get_vertex(v)['pose_for_visualization']
                    #print('u:', ux, uy)
                    #print('v:', vx, vy)
                    #print('Path in graph:', path_len)
                    #print('Path through cur:', dst_through_cur)
                    found_loop_closure = True
                    self.publish_loop_closure_results(path, global_pose_for_visualization)
                    break
            if found_loop_closure:
                break
        return found_loop_closure

    def interpolate_pose(self, pose_left_stamped, pose_right_stamped, timestamp):
        alpha = (timestamp - pose_left_stamped[0]) / (pose_right_stamped[0] - pose_left_stamped[0])
        pose_left_stamped[-1] = normalize(pose_left_stamped[-1])
        pose_right_stamped[-1] = normalize(pose_right_stamped[-1])
        pose_left = np.array(pose_left_stamped[1:])
        pose_right = np.array(pose_right_stamped[1:])
        result = alpha * pose_right + (1 - alpha) * pose_left
        if abs(pose_left[2] - pose_right[2]) > np.pi:
            if pose_left[2] < 0:
                pose_left[2] += 2 * np.pi
            if pose_right[2] < 0:
                pose_right[2] += 2 * np.pi
            result = alpha * pose_right + (1 - alpha) * pose_left
            print('Interpolation:', pose_left, pose_right, result)
        return result

    def get_sync_pose_and_images(self, timestamp):
        if len(self.odom_poses) == 0:
            print('No odometry available!')
            return None, None, None, None
        if len(self.gt_poses) == 0:
            print('No pose available!')
            return None, None, None, None
        if len(self.rgb_buffer_front) == 0:
            print('No front image available!')
            return None, None, None, None
        if len(self.rgb_buffer_back) == 0:
            print('No back image available!')
            return None, None, None, None
        eps = 0.1
        i = 0
        while i < len(self.gt_poses) and self.gt_poses[i][0] < timestamp:
            i += 1
        j = 0
        while j < len(self.rgb_buffer_front) and self.rgb_buffer_front[j][0] < timestamp - eps:
            j += 1
        k = 0
        while k < len(self.rgb_buffer_back) and self.rgb_buffer_back[k][0] < timestamp - eps:
            k += 1
        l = 0
        while l< len(self.odom_poses) and self.odom_poses[l][0] < timestamp:
            l += 1
        if i == len(self.gt_poses):
            #print('No sync pose available!')
            return None, None, None, None
        if j == len(self.rgb_buffer_front):
            #print('No sync front image available!')
            return None, None, None, None
        if k == len(self.rgb_buffer_back):
            #print('No sync back image available!')
            return None, None, None, None
        if l == len(self.odom_poses):
            print('No odometry available!')
            return None, None, None, None
        if i == 0:
            if self.gt_poses[0][0] - timestamp > 0.2 or len(self.rgb_buffer_front) == 0 or len(self.rgb_buffer_back) == 0:
                #print('No sync pose available!')
                return None, None, None, None
            gt_pose_sync = self.gt_poses[0][1:]
        else:
            alpha = (timestamp - self.gt_poses[i - 1][0]) / (self.gt_poses[i][0] - self.gt_poses[i - 1][0])
            pose_left = np.array(self.gt_poses[i - 1][1:])
            pose_right = np.array(self.gt_poses[i][1:])
            gt_pose_sync = self.interpolate_pose(self.gt_poses[i - 1], self.gt_poses[i], timestamp)
        if l == 0:
            if self.odom_poses[0][0] - timestamp > 0.2 or len(self.rgb_buffer_front) == 0 or len(self.rgb_buffer_back) == 0:
                #print('No sync pose available!')
                return None, None, None, None
            odom_pose_sync = self.odom_poses[0][1:]
        else:
            alpha = (timestamp - self.odom_poses[l - 1][0]) / (self.odom_poses[l][0] - self.odom_poses[l - 1][0])
            pose_left = np.array(self.odom_poses[l - 1][1:])
            pose_right = np.array(self.odom_poses[l][1:])
            odom_pose_sync = self.interpolate_pose(self.odom_poses[l - 1], self.odom_poses[l], timestamp)
        return gt_pose_sync, odom_pose_sync,\
               self.rgb_buffer_front[j][1], self.rgb_buffer_back[k][1]

    def is_inside_vcur(self):
        return self.last_vertex['grid'].is_inside(*self.rel_pose_of_vcur)

    def publish_last_vertex(self):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = self.map_frame
        marker_msg.type = Marker.SPHERE
        last_x, last_y, last_theta = self.last_vertex['pose_for_visualization']
        marker_msg.pose.position.x = last_x
        marker_msg.pose.position.y = last_y
        marker_msg.pose.position.z = 0.0
        marker_msg.color.r = 0
        marker_msg.color.g = 1
        marker_msg.color.b = 0
        marker_msg.color.a = 1
        marker_msg.scale.x = 0.5
        marker_msg.scale.y = 0.5
        marker_msg.scale.z = 0.5
        self.last_vertex_publisher.publish(marker_msg)
        vertex_id_msg = Int32()
        vertex_id_msg.data = self.last_vertex_id
        self.last_vertex_id_publisher.publish(vertex_id_msg)
        self.tfbr.sendTransform((last_x, last_y, 0),
                                    tf.transformations.quaternion_from_euler(0, 0, last_theta),
                                    self.current_stamp,
                                    "vcur",
                                    self.map_frame)

    def publish_local_grid(self):
        local_grid = self.last_vertex['grid'].grid
        local_grid_msg = OccupancyGrid()
        local_grid_msg.header.stamp = self.current_stamp
        local_grid_msg.header.frame_id = 'vcur'
        resolution = 0.1
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
        local_map[local_map == 2] = 100
        local_map[local_map == 0] = -1
        local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.local_grid_publisher.publish(local_grid_msg)

    def publish_cur_grid(self):
        local_grid_msg = OccupancyGrid()
        local_grid_msg.header.stamp = self.current_stamp
        local_grid_msg.header.frame_id = 'current_state'
        resolution = 0.1
        local_grid_msg.info.resolution = resolution
        local_grid_msg.info.width = self.cur_grid.grid.shape[1]
        local_grid_msg.info.height = self.cur_grid.grid.shape[0]
        local_grid_msg.info.origin.position.x = -self.cur_grid.grid.shape[1] * resolution / 2
        local_grid_msg.info.origin.position.y = -self.cur_grid.grid.shape[0] * resolution / 2
        local_grid_msg.info.origin.orientation.x = 0
        local_grid_msg.info.origin.orientation.y = 0
        local_grid_msg.info.origin.orientation.z = 0
        local_grid_msg.info.origin.orientation.w = 1
        local_map = self.cur_grid.grid.T.ravel().astype(np.int8)
        local_map[local_map == 2] = 100
        local_map[local_map == 0] = -1
        local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.cur_grid_publisher.publish(local_grid_msg)
        
        rel_x_old, rel_y_old, rel_theta_old = self.get_rel_pose_from_stamp(self.cur_stamp)[0]
        cur_grid_transformed = self.cur_grid.get_transformed_grid(rel_x_old, rel_y_old, -rel_theta_old)
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
        local_map[local_map == 2] = 100
        local_map[local_map == 0] = -1
        local_map[local_map == 1] = 0
        local_grid_msg.data = list(local_map)
        self.cur_grid_transformed_publisher.publish(local_grid_msg)

    def publish_rel_pose(self):
        rel_pose_msg = PoseStamped()
        rel_pose_msg.header.stamp = rospy.Time.now()
        rel_pose_msg.header.frame_id = 'last_vertex'
        rel_x, rel_y, rel_theta = self.rel_pose_of_vcur
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

    # def update_local_grid(self):
    #     save_dir = os.path.join('/home/kirill/test_local_grid', str(self.local_grid_cnt))
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     local_grid = self.last_vertex['grid']
    #     np.savez(os.path.join(save_dir, 'central_grid.npz'), local_grid)
    #     np.savetxt(os.path.join(save_dir, 'central_position.txt'), self.last_vertex['pose_for_visualization'])
    #     edges_from_vcur = self.graph.get_edges_from(self.last_vertex_id)
    #     for v, rel_pose in edges_from_vcur:
    #         vgrid = self.graph.get_vertex(v)['grid']
    #         rel_x, rel_y, rel_theta = rel_pose
    #         rel_x_rotated = -rel_x * np.cos(rel_theta) - rel_y * np.sin(rel_theta)
    #         rel_y_rotated = rel_x * np.sin(rel_theta) - rel_y * np.cos(rel_theta)
    #         vgrid_transformed = transform_grid(vgrid, rel_x, rel_y, -rel_theta)
    #         local_grid = np.maximum(local_grid, vgrid_transformed)
    #         np.savez(os.path.join(save_dir, '{}_grid.npz'.format(v)), vgrid)
    #         np.savez(os.path.join(save_dir, '{}_grid_transformed.npz').format(v), vgrid_transformed)
    #         np.savetxt(os.path.join(save_dir, '{}_rel_pose.txt'.format(v)), np.array(rel_pose))
    #         np.savetxt(os.path.join(save_dir, '{}_position.txt'.format(v)), np.array(self.graph.get_vertex(v)['pose_for_visualization']))
    #     self.local_grid = local_grid
    #     self.local_grid_cnt += 1

    def get_rel_pose_from_stamp(self, timestamp, verbose=False):
        if len(self.rel_poses_stamped) == 0:
            self.rel_poses_stamped.append([timestamp] + self.rel_pose_of_vcur)
        j = 0
        while j < len(self.rel_poses_stamped) and self.rel_poses_stamped[j][0] < timestamp:
            j += 1
        if j == len(self.rel_poses_stamped):
            j -= 1
        return self.rel_poses_stamped[j][1:], get_rel_pose(*self.rel_poses_stamped[j][1:], *self.rel_pose_of_vcur)

    def get_rel_pose_from_localization(self, rel_pose_v_to_vcur, timestamp):
        if len(self.rel_poses_stamped) == 0:
            self.rel_poses_stamped.append([timestamp] + self.rel_pose_of_vcur)
        j = 0
        while j < len(self.rel_poses_stamped) and self.rel_poses_stamped[j][0] < self.localizer.localized_stamp:
            j += 1
        if j == len(self.rel_poses_stamped):
            j -= 1
        #print('Rel pose stamped:', self.rel_poses_stamped[j])
        #print(self.rel_poses_stamped[j][1:])
        #print(self.rel_pose_of_vcur)
        rel_pose_after_localization = get_rel_pose(*self.rel_poses_stamped[j][1:], *self.rel_pose_of_vcur)
        #print('Rel pose after localization:', rel_pose_after_localization)
        rel_pose_of_vcur = apply_pose_shift(rel_pose_v_to_vcur, *rel_pose_after_localization)
        return rel_pose_of_vcur

    def reattach_by_edge(self, cur_grid, timestamp):
        pose_diffs = []
        edge_poses = []
        neighbours = []
        target_pose = None
        target_pose_diff = None
        for vertex_id, pose_to_vertex in self.graph.adj_lists[self.last_vertex_id]:
            edge_poses.append(pose_to_vertex)
            pose_diff = np.sqrt((pose_to_vertex[0] - self.rel_pose_of_vcur[0]) ** 2 + (pose_to_vertex[1] - self.rel_pose_of_vcur[1]) ** 2)
            pose_diffs.append(pose_diff)
            neighbours.append(vertex_id)
            if vertex_id == target_vertex_id:
                target_pose = pose_to_vertex
                target_pose_diff = pose_diff
        dist_to_vcur = np.sqrt(self.rel_pose_of_vcur[0] ** 2 + self.rel_pose_of_vcur[1] ** 2)
        changed = False
        if target_vertex_id is not None and target_pose_diff is not None and target_pose_diff < 5 and target_pose_diff < dist_to_vcur:
            nearest_vertex_id = target_vertex_id
            pose_on_edge = target_pose
        elif len(pose_diffs) > 0 and min(pose_diffs) < 3 and min(pose_diffs) < dist_to_vcur:
            nearest_vertex_id = neighbours[np.argmin(pose_diffs)]
            pose_on_edge = edge_poses[np.argmin(pose_diffs)]
        else:
            return False
        print('\n\n\n                    Pose on edge:', pose_on_edge)
        old_rel_pose_of_vcur = self.rel_pose_of_vcur
        rel_pose_to_vertex = get_rel_pose(*self.rel_pose_of_vcur, *pose_on_edge)
        cur_grid_transformed = cur_grid.copy()
        cur_grid_transformed.transform(*rel_pose_to_vertex)
        x, y, theta = self.graph.get_transform_to_vertex(nearest_vertex_id, cur_grid_transformed)
        if x is not None:
            changed = True
            print('\n\n\n Change to vertex {} by edge \n\n\n'.format(nearest_vertex_id))
            print('Old transform:', rel_pose_to_vertex)
            print('Scan matching transform:', x, y, theta)
            self.last_vertex_id = nearest_vertex_id
            self.last_vertex = self.graph.get_vertex(self.last_vertex_id)
            #self.rel_pose_of_vcur = self.graph.inverse_transform(x, y, theta)
            self.rel_pose_of_vcur = self.graph.inverse_transform(*rel_pose_to_vertex)
            if self.rel_pose_vcur_to_loc is not None:
                self.rel_pose_vcur_to_loc = apply_pose_shift(self.graph.inverse_transform(*pose_on_edge), *self.rel_pose_vcur_to_loc)
            self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        else:
            print('Failed to match current cloud to vertex {}!'.format(nearest_vertex_id))

        return changed

    def reattach_by_localization(self, iou_threshold, cur_grid, timestamp):
        vertex_ids, rel_poses = self.localization_results
        if len(self.rel_poses_stamped) > 0 and self.localizer.localized_stamp < self.rel_poses_stamped[0][0]:
            print('Old localization 1! Ignore it')
            #print((self.rel_poses_stamped[0][0] - self.localizer.localized_stamp).to_sec())
            return
        found_proper_vertex = False
        # First try to pass the nearest edge
        for i, v in enumerate(vertex_ids):
            if v == self.last_vertex_id:
                continue
            pred_rel_pose_vcur_to_v = apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_poses[i]))
            iou = cur_grid.get_iou(self.graph.get_vertex(v)['grid'], *pred_rel_pose_vcur_to_v, save=False)
            vx, vy, vtheta = self.graph.get_vertex(v)['pose_for_visualization']
            #print('IoU between ({}, {}) and ({}, {}) is {}'.format(x, y, vx, vy, iou))
            if iou > iou_threshold:
                found_proper_vertex = True
                print('\n\n\n Change to vertex {} with coords ({}, {})\n\n\n'.format(v, vx, vy))
                #last_x, last_y, last_theta = self.last_vertex['pose_for_visualization']
                self.graph.add_edge(self.last_vertex_id, v, *pred_rel_pose_vcur_to_v)
                self.last_vertex_id = v
                self.last_vertex = self.graph.get_vertex(v)
                _, rel_pose_after_localization = self.get_rel_pose_from_stamp(self.localizer.localized_stamp, verbose=True)
                pred_rel_pose = apply_pose_shift(rel_poses[i], *rel_pose_after_localization)
                save_dir = '/home/kirill/test_rel_pose/{}'.format(self.rel_pose_cnt)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                np.savetxt(os.path.join(save_dir, 'predicted_rel_pose.txt'), pred_rel_pose)
                self.rel_pose_cnt += 1
                self.rel_pose_of_vcur = pred_rel_pose
                self.rel_pose_vcur_to_loc = apply_pose_shift(self.graph.inverse_transform(*pred_rel_pose_vcur_to_v), *self.rel_pose_vcur_to_loc)
                self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
                #self.localization_time = 0
                return True
        return False

    def add_new_vertex(self, timestamp, global_pose_for_visualization, img_front, img_back, cur_cloud, vertex_ids, rel_poses):
        new_vertex_id = self.graph.add_vertex(global_pose_for_visualization,
                                                img_front, img_back,
                                                cur_cloud)
        new_vertex = self.graph.get_vertex(new_vertex_id)
        pose_stamped, new_rel_pose_of_vcur = self.get_rel_pose_from_stamp(timestamp)
        if self.last_vertex is not None:
            #true_rel_pose = get_rel_pose(*self.last_vertex['pose_for_visualization'], *new_vertex['pose_for_visualization'])
            self.graph.add_edge(self.last_vertex_id, new_vertex_id, *pose_stamped)

        self.rel_pose_of_vcur = new_rel_pose_of_vcur
        if self.rel_pose_vcur_to_loc is not None:
            self.rel_pose_vcur_to_loc = get_rel_pose(*pose_stamped, *self.rel_pose_vcur_to_loc)
        if len(self.rel_poses_stamped) == 0 or self.localizer.localized_stamp is None or self.localizer.localized_stamp >= self.rel_poses_stamped[0][0]:
            for v, rel_pose in zip(vertex_ids, rel_poses):
                pred_rel_pose = apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_pose))
                if np.sqrt(pred_rel_pose[0] ** 2 + pred_rel_pose[1] ** 2) < 5:
                    self.graph.add_edge(new_vertex_id, v, *pred_rel_pose)
        else:
            print('Old localization 2! Ignore it')
            #print((self.rel_poses_stamped[0][0] - self.localizer.localized_stamp).to_sec())
        self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        self.graph.save_vertex(new_vertex_id)
        self.last_vertex_id = new_vertex_id
        self.last_vertex = new_vertex
        self.graph.save_vertex(new_vertex_id)

    def localization_callback(self, msg):
        #self.publish_cur_cloud()
        self.mutex.acquire()
        if len(self.rel_poses_stamped) > 0 and self.localizer.localized_stamp < self.rel_poses_stamped[0][0]:
            print('Old localization 3! Ignore it')
            #print((self.rel_poses_stamped[0][0] - self.localizer.localized_stamp).to_sec())
            self.mutex.release()
            return
        n = msg.layout.dim[0].size // 4
        vertex_ids = [int(x) for x in msg.data[:n]]
        rel_poses = np.zeros((n, 3))
        rel_poses[:, 0] = msg.data[n:2 * n]
        rel_poses[:, 1] = msg.data[2 * n:3 * n]
        rel_poses[:, 2] = msg.data[3 * n:4 * n]
        dists = np.sqrt(rel_poses[:, 0] ** 2 + rel_poses[:, 1] ** 2)
        vertex_ids_refined = []
        self.rel_pose_vcur_to_loc, _ = self.get_rel_pose_from_stamp(self.localizer.localized_stamp)
        for vertex_id, rel_pose in zip(vertex_ids, rel_poses):
            #x, y, theta = get_rel_pose(*self.last_vertex['pose_for_visualization'], *self.graph.get_vertex(vertex_id)['pose_for_visualization'])
            #dst = np.sqrt(x ** 2 + y ** 2)
            if self.check_path_condition(self.last_vertex_id, vertex_id, \
                                         apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_pose))) and \
                                         vertex_id != self.last_vertex_id and not self.graph.has_edge(self.last_vertex_id, vertex_id):
            #if dst < 8:
                vertex_ids_refined.append(vertex_id)
            else:
                print('Remove vertex {} from localization, it is too far'.format(vertex_id))
        vertex_ids = vertex_ids_refined
        self.localization_results = (vertex_ids, rel_poses)
        print('\n\nLocalized in vertices', vertex_ids, '\n\n')
        self.localization_pose = [self.localizer.localized_x, self.localizer.localized_y, self.localizer.localized_theta]
        x = self.localizer.localized_x
        y = self.localizer.localized_y
        theta = self.localizer.localized_theta
        stamp = self.localizer.localized_stamp
        global_pose_for_visualization = [x, y, theta]
        img_front = self.localizer.localized_img_front
        img_back = self.localizer.localized_img_back
        cloud = self.localizer.localized_cloud
        # grid = get_occupancy_grid(cloud)
        grid = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
        grid.load_from_cloud(cloud)
        if len(vertex_ids) == 0:
            self.mutex.release()
            return
        self.localization_time = rospy.Time.now().to_sec()
        if self.last_vertex is None:
            print('Initially attached to vertex {} from localization'.format(vertex_ids[0]))
            self.last_vertex_id = vertex_ids[0]
            self.last_vertex = self.graph.get_vertex(vertex_ids[0])
            self.rel_pose_of_vcur = self.graph.inverse_transform(*rel_poses[0])
            self.rel_pose_vcur_to_loc = self.rel_pose_of_vcur
            self.current_stamp = self.localizer.localized_stamp
            self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        else:
            if self.find_loop_closure(vertex_ids, dists, global_pose_for_visualization):
                print('Found loop closure. Add new vertex to close loop')
                self.add_new_vertex(stamp, global_pose_for_visualization, 
                                    img_front, img_back, cloud,
                                    vertex_ids, rel_poses)
            # else:
            #     changed = self.reattach_by_localization(self.cur_iou, 
            #                                             grid, self.localizer.localized_stamp)
            #     if not changed:
            #         for i in range(len(vertex_ids)):
            #             #true_rel_pose_vcur_to_v = get_rel_pose(*self.last_vertex['pose_for_visualization'], *self.graph.get_vertex(i)['pose_for_visualization'])
            #             pred_rel_pose_vcur_to_v = apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_poses[i]))
            #             if np.sqrt(pred_rel_pose_vcur_to_v[0] ** 2 + pred_rel_pose_vcur_to_v[1] ** 2) < 5:
            #                 print('\nAdd edge from {} to {} with rel pose ({}, {}, {})\n'.format(self.last_vertex_id, vertex_ids[i], *pred_rel_pose_vcur_to_v))
            #                 self.graph.add_edge(self.last_vertex_id, vertex_ids[i], *pred_rel_pose_vcur_to_v)
        self.mutex.release()

    def update_by_iou(self, global_pose_for_visualization, img_front, img_back, cur_cloud, timestamp):
        if self.publish_gt_map_flag:
            self.publish_gt_map()
        t1 = rospy.Time.now().to_sec()
        if cur_cloud is None:
            print('No point cloud received!')
            return
        # cur_grid = get_occupancy_grid(cur_cloud)
        cur_grid = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
        cur_grid.load_from_cloud(cur_cloud)
        self.cur_grid = cur_grid
        self.cur_stamp = timestamp
        if self.last_vertex is None:
            rospy.sleep(2.0) # wait for localization
            if self.last_vertex is None: # If localization failed, init with new vertex
                print('Add new vertex at start')
                self.add_new_vertex(timestamp, global_pose_for_visualization, 
                                    img_front, img_back, cur_cloud,
                                    [], [])
        changed = self.reattach_by_edge(cur_grid, timestamp)
        #last_x, last_y, _ = self.last_vertex['pose_for_visualization']
        inside_vcur = self.is_inside_vcur()
        # print('Rel pose of vcur:', self.rel_pose_of_vcur)
        iou = cur_grid.get_iou(self.last_vertex['grid'], *self.graph.inverse_transform(*self.rel_pose_of_vcur), \
                               save=True, cnt=self.iou_cnt)
        self.iou_cnt += 1
        self.cur_iou = iou
        # print('IoU:', iou)
        # print('Vcur:', self.last_vertex_id)
        dst = np.sqrt(self.rel_pose_of_vcur[0] ** 2 + self.rel_pose_of_vcur[1] ** 2)
        if not inside_vcur or iou < self.iou_threshold or dst > self.max_edge_length:
            self.mutex.acquire()
            if not inside_vcur:
                print('Moved outside vcur {}'.format(self.last_vertex_id))
            elif iou < self.iou_threshold:
                print('Low IoU {}'.format(iou))
            else:
                print('Too far from location center')
            #print('Last localization {} seconds ago'.format(rospy.Time.now().to_sec() - self.localization_time))
            #print(self.path[0], self.last_vertex_id)
            if not changed:
                if rospy.Time.now().to_sec() - self.localization_time < 5:
                    #print('Localized stamp:', self.localizer.localized_stamp)
                    changed = self.reattach_by_localization(0, cur_grid, self.localizer.localized_stamp)
                    if not changed:
                        print('No proper vertex to change. Add new vertex')
                        vertex_ids, rel_poses = self.localization_results
                        self.add_new_vertex(timestamp, global_pose_for_visualization,
                                            img_front, img_back, cur_cloud,
                                            vertex_ids, rel_poses)
                else:
                    print('No recent localization. Add new vertex')
                    self.add_new_vertex(timestamp, global_pose_for_visualization,
                                        img_front, img_back, cur_cloud,
                                        [], [])
            self.mutex.release()
        self.graph.publish_graph()
        self.publish_last_vertex()
        self.publish_local_grid()
        self.publish_cur_grid()
        self.publish_rel_pose()

    def pcd_callback(self, msg):
        dt = (rospy.Time.now() - msg.header.stamp).to_sec()
        #print('Msg lag is {} seconds'.format(dt))
        if dt > 0.5:
            return
        cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back = self.get_sync_pose_and_images(msg.header.stamp.to_sec())
        start_time = rospy.Time.now().to_sec()
        while cur_global_pose is None:
            cur_global_pose, cur_odom_pose, cur_img_front, cur_img_back = self.get_sync_pose_and_images(msg.header.stamp.to_sec())
            rospy.sleep(1e-2)
            if rospy.Time.now().to_sec() - start_time > 0.5:
                print('Waiting for sync pose and images timed out!')
                return
        self.cur_global_pose = cur_global_pose
        
        # print('Cur odom pose:', cur_odom_pose)
        x, y, theta = cur_odom_pose
        if self.odom_pose is not None:
            rel_x, rel_y, rel_theta = get_rel_pose(*self.odom_pose, x, y, theta)
        else:
            rel_x, rel_y, rel_theta = x, y, theta
        self.odom_pose = [x, y, theta]
        if self.rel_pose_of_vcur is None:
            print('Rel pose of vcur is None, initialize it as ({}, {}, {})'.format(rel_x, rel_y, rel_theta))
            self.rel_pose_of_vcur = [rel_x, rel_y, rel_theta - 0.1]
        else:
            # print('Apply pose shift:', rel_x, rel_y, rel_theta)
            self.rel_pose_of_vcur = apply_pose_shift(self.rel_pose_of_vcur, rel_x, rel_y, rel_theta)

        self.current_stamp = msg.header.stamp
        self.rel_poses_stamped.append([msg.header.stamp] + self.rel_pose_of_vcur)

        cur_cloud = get_xyz_coords_from_msg(msg, self.pcd_fields, self.pcd_rotation)
        self.localizer.global_pose_for_visualization = cur_global_pose
        self.localizer.img_front = cur_img_front
        self.localizer.img_back = cur_img_back
        self.localizer.cloud = cur_cloud
        self.localizer.stamp = msg.header.stamp
        start_time = time.time()
        self.update_by_iou(cur_global_pose, cur_img_front, cur_img_back, cur_cloud, msg.header.stamp)
        # print('Update by iou time:', time.time() - start_time)

    def save_graph(self):
        self.graph.save_to_json(self.path_to_save_json)
        print('N of localizer calls:', self.localizer.cnt)
        print('N of localization fails:', self.localizer.n_loc_fails)

    def run(self):
        rospy.spin()


toposlam_model = TopoSLAMModel()
toposlam_model.run()
#toposlam_model.save_graph()