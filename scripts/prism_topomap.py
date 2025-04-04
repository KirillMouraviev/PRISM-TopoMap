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
import torch
from localization import Localizer
from utils import *
from topo_graph import TopologicalGraph
from local_grid import LocalGrid
from models import get_place_recognition_model, get_registration_model
from skimage.io import imsave
from threading import Lock

from memory_profiler import profile

rospy.init_node('prism_topomap_node')

class TopoSLAMModel():
    def __init__(self, config,
                 path_to_load_graph=None,
                 path_to_save_graph=None):
        print('Intializing...')
        print('File:', __file__)
        self.path_to_load_graph = path_to_load_graph
        self.path_to_save_graph = path_to_save_graph
        self.init_params_from_config(config)

        self.last_vertex = None
        self.last_vertex_id = None
        self.prev_img_front = None
        self.prev_img_back = None
        self.prev_cloud = None
        self.prev_pose_for_visualization = None
        self.prev_rel_pose = None
        self.odom_pose = None
        self.in_sight_response = None
        self.cur_global_pose = None
        self.pose_pairs = []
        self.cur_grid = None
        self.cur_stamp = None
        self.rel_poses_stamped = []
        self.accumulated_curbs = None
        self.rel_pose_of_vcur = None
        self.rel_pose_vcur_to_loc = None
        self.found_loop_closure = False
        self.need_to_change_vcur = False
        self.path = []

        self.graph = TopologicalGraph(place_recognition_model=self.place_recognition_model,
                                      place_recognition_index=self.place_recognition_index,
                                      registration_model=self.registration_model,
                                      inline_registration_model=self.inline_registration_model,
                                      map_frame=self.map_frame,
                                      pointcloud_quantization_size=self.pointcloud_quantization_size,
                                      registration_score_threshold=self.registration_score_threshold,
                                      inline_registration_score_threshold=self.inline_registration_score_threshold,
                                      grid_resolution=self.grid_resolution,
                                      grid_radius=self.grid_radius,
                                      max_grid_range=self.max_grid_range,
                                      floor_height=self.floor_height, ceil_height=self.ceil_height)
        if self.path_to_load_graph is not None:
            print('Loading graph from {}'.format(self.path_to_load_graph))
            self.graph.load_from_json(self.path_to_load_graph)
            print('Done!')
            # print('Grid max of graph vertex 0:', self.graph.vertices[0]['grid'].grid.max())
        self.localizer = Localizer(self.graph, map_frame=self.map_frame, top_k=self.top_k)

        self.localization_time = 0
        self.last_successful_match_time = 0
        self.cur_iou = 0
        self.localization_results = ([], [])
        self.edge_reattach_cnt = 0
        self.rel_pose_cnt = 0
        self.iou_cnt = 0
        self.local_grid_cnt = 0
        self.current_stamp = None
        self.mutex = Lock()
        print('Initializing done!')

    def init_params_from_config(self, config):
        # TopoMap
        topomap_config = config['topomap']
        self.mode = topomap_config['mode']
        if self.mode not in ['localization', 'mapping']:
            print('Invalid mode {}. Mode can be only "mapping" or "localization"'.format(self.mode))
            exit(1)
        if 'start_location' in topomap_config:
            self.start_location = topomap_config['start_location']
            if 'start_local_pose' in topomap_config:
                self.start_local_pose = topomap_config['start_local_pose']
        else:
            self.start_location = None
        self.iou_threshold = topomap_config['iou_threshold']
        self.localization_frequency = topomap_config['localization_frequency']
        self.rel_pose_correction_frequency = topomap_config['rel_pose_correction_frequency']
        self.max_edge_length = topomap_config['max_edge_length']
        self.drift_coef = topomap_config['drift_coef']
        # Input
        pointcloud_config = config['input']['pointcloud']
        self.floor_height = pointcloud_config['floor_height']
        self.ceil_height = pointcloud_config['ceiling_height']
        # Place recognition
        place_recognition_config = config['place_recognition']
        self.pointcloud_quantization_size = place_recognition_config['pointcloud_quantization_size']
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
        self.local_jump_threshold = inline_registration_config['jump_threshold']
        # Map frame
        visualization_config = config['visualization']
        self.map_frame = visualization_config['map_frame']

    def correct_rel_pose(self, event=None):
        if self.last_vertex_id is None:
            return
        if self.cur_grid is None:
            return
        self.mutex.acquire()
        rel_x_old, rel_y_old, rel_theta_old = self.get_rel_pose_from_stamp(self.cur_stamp)[0]
        cur_grid_transformed = LocalGrid(
            resolution=self.cur_grid.resolution,
            radius=self.cur_grid.radius,
            max_range=self.cur_grid.max_range,
            grid=self.cur_grid.grid.copy()
        )
        cur_grid_transformed.transform(rel_x_old, rel_y_old, -rel_theta_old)
        x, y, theta = self.graph.get_transform_to_vertex(self.last_vertex_id, cur_grid_transformed)
        # true_rel_pose = get_rel_pose(*self.last_vertex['pose_for_visualization'], *self.cur_global_pose)
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
        # print('Checking path condition between {} and {}'.format(u, v))
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
        # print('Path length to vertex {} is {}'.format(v, path_length))
        straight_length = np.sqrt(rel_pose_along_path[0] ** 2 + rel_pose_along_path[1] ** 2)
        # print('Straight length:', straight_length)
        if path_length > 3 * straight_length or straight_length < 10:
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

    def find_loop_closure(self, vertex_ids, dists, global_pose_for_visualization):
        found_loop_closure = False
        path = []
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
                # print('Path len: {}, dst through cur: {}'.format(path_len, dst_through_cur))
                if path_len > 5 and path_len > 2 * dst_through_cur and self.check_path_condition(u, v):
                    #ux, uy, _ = self.graph.get_vertex(u)['pose_for_visualization']
                    #vx, vy, _ = self.graph.get_vertex(v)['pose_for_visualization']
                    #print('u:', ux, uy)
                    #print('v:', vx, vy)
                    #print('Path in graph:', path_len)
                    #print('Path through cur:', dst_through_cur)
                    found_loop_closure = True
                    break
            if found_loop_closure:
                break
        return found_loop_closure, path

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
            # print('Interpolation:', pose_left, pose_right, result)
        return result

    def is_inside_vcur(self):
        return self.last_vertex['grid'].is_inside(*self.rel_pose_of_vcur)

    def get_rel_pose_from_stamp(self, timestamp, verbose=False):
        if len(self.rel_poses_stamped) == 0:
            self.rel_poses_stamped.append([timestamp] + self.rel_pose_of_vcur)
        j = 0
        while j < len(self.rel_poses_stamped) and self.rel_poses_stamped[j][0] < timestamp:
            j += 1
        if j == len(self.rel_poses_stamped):
            j -= 1
        return self.rel_poses_stamped[j][1:], get_rel_pose(*self.rel_poses_stamped[j][1:], *self.rel_pose_of_vcur)

    def get_rel_pose_since_localization(self):
        if len(self.rel_poses_stamped) == 0:
            return self.rel_pose_of_vcur
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
        return rel_pose_after_localization

    def reattach_by_edge(self, cur_grid, timestamp, require_match=True):
        pose_diffs = []
        edge_poses = []
        neighbours = []
        print('Vcur:', self.last_vertex_id)
        for vertex_id, pose_to_vertex in self.graph.adj_lists[self.last_vertex_id]:
            edge_poses.append(pose_to_vertex)
            pose_diff = np.sqrt((pose_to_vertex[0] - self.rel_pose_of_vcur[0]) ** 2 + (pose_to_vertex[1] - self.rel_pose_of_vcur[1]) ** 2)
            pose_diffs.append(pose_diff)
            neighbours.append(vertex_id)
        dist_to_vcur = np.sqrt(self.rel_pose_of_vcur[0] ** 2 + self.rel_pose_of_vcur[1] ** 2)
        changed = False
        if len(pose_diffs) > 0 and min(pose_diffs) < dist_to_vcur and min(pose_diffs) < 5:
            nearest_vertex_id = neighbours[np.argmin(pose_diffs)]
            pose_on_edge = edge_poses[np.argmin(pose_diffs)]
        else:
            print('Could not find proper edge to change')
            return False
        print('Nearest vertex id:', nearest_vertex_id)
        # print('\n\n\n                    Pose on edge:', pose_on_edge)
        old_rel_pose_of_vcur = self.rel_pose_of_vcur
        rel_pose_to_vertex = get_rel_pose(*self.rel_pose_of_vcur, *pose_on_edge)
        print('Rel pose of vcur:', self.rel_pose_of_vcur)
        print('Pose on edge:', pose_on_edge)
        print('Rel pose to vertex:', rel_pose_to_vertex)
        pipeline = self.graph.inline_registration_pipeline
        save_dir = os.path.join(pipeline.save_dir, str(pipeline.cnt))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        imsave(os.path.join(pipeline.save_dir, str(pipeline.cnt), 'ref_grid_source.png'), cur_grid.grid)
        print('Saved source grid to:', os.path.join(pipeline.save_dir, str(pipeline.cnt), 'ref_grid_source.png'))
        cur_grid_transformed = LocalGrid(
            resolution=self.cur_grid.resolution,
            radius=self.cur_grid.radius,
            max_range=self.cur_grid.max_range,
            grid=self.cur_grid.grid.copy()
        )
        rel_pose_to_vertex_inv = self.graph.inverse_transform(*rel_pose_to_vertex)
        cur_grid_transformed.transform(rel_pose_to_vertex_inv[0], rel_pose_to_vertex_inv[1], -rel_pose_to_vertex_inv[2])
        corr_x, corr_y, corr_theta = self.graph.get_transform_to_vertex(nearest_vertex_id, cur_grid_transformed)
        print('Require match:', require_match)
        if corr_x is not None:
            print('Old transform:', rel_pose_to_vertex)
            corr_x_inv, corr_y_inv, corr_theta_inv = self.graph.inverse_transform(corr_x, corr_y, corr_theta)
            x, y, theta = apply_pose_shift(rel_pose_to_vertex, corr_x_inv, corr_y_inv, corr_theta_inv)
            #theta = -theta
            print('Scan matching transform:', x, y, theta)
            diff = np.sqrt((x - rel_pose_to_vertex[0]) ** 2 + (y - rel_pose_to_vertex[1]) ** 2)
            print('Diff:', diff)
            if diff < self.local_jump_threshold:
                print('\n\n\n Change to vertex {} by edge \n\n\n'.format(nearest_vertex_id))
                self.last_successful_match_time = timestamp.to_sec()
                changed = True
                self.rel_pose_of_vcur = self.graph.inverse_transform(x, y, theta)
            else:
                print('Big jump! Ignore this match')
        if not changed and not require_match:
            changed = True
            print('\n\n\n Change to vertex {} by edge without matching\n\n\n'.format(nearest_vertex_id))
            self.rel_pose_of_vcur = self.graph.inverse_transform(*rel_pose_to_vertex)
        if changed:
            self.last_vertex_id = nearest_vertex_id
            self.need_to_change_vcur = False
            self.last_vertex = self.graph.get_vertex(self.last_vertex_id)
            if self.rel_pose_vcur_to_loc is not None:
                self.rel_pose_vcur_to_loc = apply_pose_shift(self.graph.inverse_transform(*pose_on_edge), *self.rel_pose_vcur_to_loc)
            print('Reset rel_poses_stamped to time', self.current_stamp.to_sec())
            self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
            self.accumulated_curbs = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
        else:
            print('Failed to match current cloud to vertex {}!'.format(nearest_vertex_id))

        return changed

    def reattach_by_localization(self, iou_threshold, cur_grid, timestamp):
        vertex_ids, rel_poses = self.localization_results
        print('Reattach by localization')
        if len(self.rel_poses_stamped) > 0 and self.localizer.localized_stamp < self.rel_poses_stamped[0][0]:
            print('Old localization 1! Ignore it')
            print((self.rel_poses_stamped[0][0] - self.localizer.localized_stamp).to_sec())
            return False
        found_proper_vertex = False
        # First try to pass the nearest edge
        for i, v in enumerate(vertex_ids):
            # if v == self.last_vertex_id:
            #     continue
            rel_pose_robot_to_loc = get_rel_pose(*self.get_rel_pose_since_localization(), *rel_poses[i])
            print('Rel pose robot to loc:', rel_pose_robot_to_loc)
            iou = cur_grid.get_iou(self.graph.get_vertex(v)['grid'], *rel_pose_robot_to_loc, save=False)
            vx, vy, vtheta = self.graph.get_vertex(v)['pose_for_visualization']
            print('v:', v)
            print('IoU between current state and ({}, {}) is {}'.format(vx, vy, iou))
            print('IoU threshold is {}'.format(iou_threshold))
            print('Need to change vcur:', self.need_to_change_vcur)
            if iou > iou_threshold or self.need_to_change_vcur:
                self.last_successful_match_time = timestamp.to_sec()
                found_proper_vertex = True
                print('\n\n\n Change to vertex {} with coords ({}, {})\n\n\n'.format(v, vx, vy))
                #last_x, last_y, last_theta = self.last_vertex['pose_for_visualization']
                if self.mode == 'mapping':
                    self.graph.add_edge(self.last_vertex_id, v, *pred_rel_pose_vcur_to_v)
                self.last_vertex_id = v
                self.need_to_change_vcur = False
                self.last_vertex = self.graph.get_vertex(v)
                _, rel_pose_after_localization = self.get_rel_pose_from_stamp(self.localizer.localized_stamp, verbose=True)
                pred_rel_pose = apply_pose_shift(rel_poses[i], *rel_pose_after_localization)
                save_dir = '/home/kirill/test_rel_pose/{}'.format(self.rel_pose_cnt)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                np.savetxt(os.path.join(save_dir, 'predicted_rel_pose.txt'), pred_rel_pose)
                self.rel_pose_cnt += 1
                self.rel_pose_of_vcur = pred_rel_pose
                #self.rel_pose_vcur_to_loc = apply_pose_shift(self.graph.inverse_transform(*pred_rel_pose_vcur_to_v), *self.rel_pose_vcur_to_loc)
                self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
                self.accumulated_curbs = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
                #self.localization_time = 0
                return True
        return False

    def add_new_vertex(self, timestamp, global_pose_for_visualization, img_front, img_back, cur_cloud, cur_grid, vertex_ids, rel_poses):
        new_vertex_id = self.graph.add_vertex(global_pose_for_visualization,
                                                img_front, img_back,
                                                cur_cloud, cur_grid)
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
            print((self.rel_poses_stamped[0][0] - self.localizer.localized_stamp).to_sec())
        self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        self.accumulated_curbs = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
        self.graph.save_vertex(new_vertex_id)
        self.last_vertex_id = new_vertex_id
        self.need_to_change_vcur = False
        self.last_vertex = new_vertex
        self.graph.save_vertex(new_vertex_id)

    def update_from_localization(self, msg):
        print('Update from localization')
        #self.mutex.acquire()
        if len(self.rel_poses_stamped) > 0 and self.localizer.localized_stamp < self.rel_poses_stamped[0][0]:
            print('Old localization 3! Ignore it')
            print((self.rel_poses_stamped[0][0] - self.localizer.localized_stamp).to_sec())
            #self.mutex.release()
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
            x, y, theta = get_rel_pose(*self.last_vertex['pose_for_visualization'], *self.graph.get_vertex(vertex_id)['pose_for_visualization'])
            dx, dy, dtheta = get_rel_pose(x, y, theta, *self.rel_pose_of_vcur)
            dst = np.sqrt(dx ** 2 + dy ** 2)
            print('dst:', dst)
            # if self.check_path_condition(self.last_vertex_id, vertex_id, \
            #                              apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_pose))):# and \
                                         #vertex_id != self.last_vertex_id and not self.graph.has_edge(self.last_vertex_id, vertex_id):
            print('dt:', self.current_stamp.to_sec() - self.last_successful_match_time)
            if self.current_stamp is None or dst < self.drift_coef * (self.current_stamp.to_sec() - self.last_successful_match_time) + 10:
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
        grid = self.localizer.localized_grid
        if len(vertex_ids) == 0:
            #self.mutex.release()
            return
        self.localization_time = rospy.Time.now().to_sec()
        if self.last_vertex is None and self.start_location is None:
            # print('Initially attached to vertex {} from localization'.format(vertex_ids[0]))
            self.mutex.acquire()
            self.last_vertex_id = vertex_ids[0]
            self.need_to_change_vcur = False
            self.last_vertex = self.graph.get_vertex(vertex_ids[0])
            self.rel_pose_of_vcur = self.graph.inverse_transform(*rel_poses[0])
            self.rel_pose_vcur_to_loc = self.rel_pose_of_vcur
            self.current_stamp = self.localizer.localized_stamp
            self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
            self.accumulated_curbs = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
            self.mutex.release()
        elif self.mode == 'mapping':
            self.mutex.acquire()
            self.found_loop_closure, self.path = self.find_loop_closure(vertex_ids, dists, global_pose_for_visualization)
            if self.found_loop_closure:
                print('Found loop closure. Add new vertex to close loop')
                self.add_new_vertex(stamp, global_pose_for_visualization, 
                                    img_front, img_back, cloud, grid,
                                    vertex_ids, rel_poses)
            self.mutex.release()
        # else: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     changed = self.reattach_by_localization(self.cur_iou, 
        #                                             grid, self.localizer.localized_stamp)
        #     if not changed and self.mode == 'mapping':
        #         for i in range(len(vertex_ids)):
        #             #true_rel_pose_vcur_to_v = get_rel_pose(*self.last_vertex['pose_for_visualization'], *self.graph.get_vertex(i)['pose_for_visualization'])
        #             pred_rel_pose_vcur_to_v = apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_poses[i]))
        #             if np.sqrt(pred_rel_pose_vcur_to_v[0] ** 2 + pred_rel_pose_vcur_to_v[1] ** 2) < 5:
        #                 print('\nAdd edge from {} to {} with rel pose ({}, {}, {})\n'.format(self.last_vertex_id, vertex_ids[i], *pred_rel_pose_vcur_to_v))
        #                 self.graph.add_edge(self.last_vertex_id, vertex_ids[i], *pred_rel_pose_vcur_to_v)
        #     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.mutex.release()

    def init_localization(self, timestamp, global_pose_for_visualization, img_front, img_back, cur_cloud, cur_grid):
        if self.start_location is not None:
            self.last_vertex_id = self.start_location
            self.need_to_change_vcur = False
            self.last_vertex = self.graph.get_vertex(self.start_location)
            if self.start_local_pose is not None:
                self.rel_pose_of_vcur = self.start_local_pose
                self.rel_pose_vcur_to_loc = self.rel_pose_of_vcur
                self.rel_poses_stamped = [[timestamp] + self.rel_pose_of_vcur]
                self.accumulated_curbs = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
        else:
            print('Waiting for localization...')
            rospy.sleep(2.0) # wait for localization
            if self.last_vertex is None: # If localization failed, init with new vertex
                if self.mode == 'mapping':
                    print('Add new vertex at start')
                    self.add_new_vertex(timestamp, global_pose_for_visualization, 
                                        img_front, img_back, cur_cloud, cur_grid,
                                        [], [])
                else:
                    while self.last_vertex is None:
                        print('Still waiting for localization... Try to move forward-backward slightly')
                        rospy.sleep(1.0)

    def update_by_iou(self, global_pose_for_visualization, cur_odom_pose, img_front, img_back, cur_cloud, cur_curbs, timestamp):
        self.mutex.acquire()
        # Update localizer
        cur_grid = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
        cur_grid.load_from_cloud(cur_cloud)
        if cur_curbs is not None and len(self.rel_poses_stamped) > 0:
            grid = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
            grid.load_curb_from_cloud(cur_curbs)
            # print(self.rel_pose_of_vcur, self.rel_poses_stamped[0])
            x, y, theta = get_rel_pose(*self.rel_poses_stamped[0][1:], *self.rel_pose_of_vcur)
            grid.transform(x, y, -theta)
            self.accumulated_curbs.grid += grid.grid
            acc_grid = self.accumulated_curbs.copy()
            x, y, theta = get_rel_pose(*self.rel_pose_of_vcur, *self.rel_poses_stamped[0][1:])
            acc_grid.transform(x, y, -theta)
            th = 2
            acc_grid.grid[acc_grid.grid < th] = 0
            acc_grid.grid[acc_grid.grid >= th] = 3
            cur_grid.grid = np.maximum(cur_grid.grid, acc_grid.grid)
            cur_grid.load_curb_from_cloud(cur_curbs)
        self.localizer.global_pose_for_visualization = global_pose_for_visualization
        self.localizer.img_front = img_front
        self.localizer.img_back = img_back
        self.localizer.cloud = cur_cloud
        self.localizer.grid = cur_grid
        self.localizer.stamp = timestamp
        
        # Update rel_pose_of_vcur
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
        self.rel_poses_stamped.append([timestamp] + self.rel_pose_of_vcur)

        t1 = rospy.Time.now().to_sec()
        if cur_cloud is None:
            print('No point cloud received!')
            return
        self.cur_grid = cur_grid
        self.cur_stamp = timestamp
        if self.last_vertex is None:
            self.init_localization(timestamp, global_pose_for_visualization, img_front, img_back, cur_cloud, cur_grid)
        changed = self.reattach_by_edge(cur_grid, timestamp, require_match=True)
        #last_x, last_y, _ = self.last_vertex['pose_for_visualization']
        inside_vcur = self.is_inside_vcur()
        # print('Rel pose of vcur:', self.rel_pose_of_vcur)
        iou = cur_grid.get_iou(self.last_vertex['grid'], *self.graph.inverse_transform(*self.rel_pose_of_vcur), \
                               save=False, cnt=self.iou_cnt)
        self.iou_cnt += 1
        self.cur_iou = iou
        # print('IoU:', iou)
        # print('Vcur:', self.last_vertex_id)
        dst = np.sqrt(self.rel_pose_of_vcur[0] ** 2 + self.rel_pose_of_vcur[1] ** 2)
        if not inside_vcur or iou < self.iou_threshold or dst > self.max_edge_length:
            self.need_to_change_vcur = True
            if not inside_vcur:
                print('Moved outside vcur {}'.format(self.last_vertex_id))
            elif iou < self.iou_threshold:
                print('Low IoU {}'.format(iou))
            else:
                print('Too far from location center')
            #print('Last localization {} seconds ago'.format(rospy.Time.now().to_sec() - self.localization_time))
            #print(self.path[0], self.last_vertex_id)
            print('Changed:', changed)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if not changed:
                print('Localization dt:', rospy.Time.now().to_sec() - self.localization_time)
                if rospy.Time.now().to_sec() - self.localization_time < 5:
                    #print('Localized stamp:', self.localizer.localized_stamp)
                    changed = self.reattach_by_localization(self.cur_iou, cur_grid, self.localizer.localized_stamp)
                    print('Changed from localization:', changed)
                    if not changed:
                        if self.mode == 'mapping':
                            print('No proper vertex to change. Add new vertex')
                            vertex_ids, rel_poses = self.localization_results
                            self.add_new_vertex(timestamp, global_pose_for_visualization,
                                                img_front, img_back, cur_cloud, cur_grid,
                                                vertex_ids, rel_poses)
                else:
                    if self.mode == 'mapping':
                        print('No recent localization. Add new vertex')
                        self.add_new_vertex(timestamp, global_pose_for_visualization,
                                            img_front, img_back, cur_cloud, cur_grid,
                                            [], [])
                    # else:
                    #     self.reattach_by_edge(cur_grid, timestamp, require_match=False)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if not changed and self.mode == 'localization':
                self.reattach_by_edge(cur_grid, timestamp, require_match=False)
        self.mutex.release()

    def save_graph(self):
        if self.path_to_save_graph is not None:
            self.graph.save_to_json(self.path_to_save_graph)
        print('N of localizer calls:', self.localizer.cnt)
        print('N of localization fails:', self.localizer.n_loc_fails)