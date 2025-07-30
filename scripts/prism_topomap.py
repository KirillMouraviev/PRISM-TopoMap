import numpy as np
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
from typing import Dict
from torch import Tensor
import MinkowskiEngine as ME

class TopoSLAMModel():
    def __init__(self, config,
                 path_to_load_graph=None,
                 path_to_save_graph=None,
                 path_to_save_logs=None):
        print('Intializing...')
        print('File:', __file__)
        self.path_to_load_graph = path_to_load_graph
        self.path_to_save_graph = path_to_save_graph
        self.path_to_save_logs = path_to_save_logs
        if self.path_to_save_logs is not None and not os.path.exists(self.path_to_save_logs):
            os.mkdir(self.path_to_save_logs)
        if self.path_to_save_logs is not None:
            self.path_to_save_iou_results = os.path.join(self.path_to_save_logs, 'test_iou')
        else:
            self.path_to_save_iou_results = None
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
        layer_names = ['occupancy', 'density_map', 'height_map']
        if config['input']['pointcloud']['subscribe_to_curbs']:
            layer_names.append('curbs')
        self.cur_grid = LocalGrid(resolution=self.grid_resolution, 
                                  radius=self.grid_radius, 
                                  max_range=self.max_grid_range,
                                  floor_height=self.floor_height, ceil_height=self.ceil_height,
                                  layer_names=layer_names,
                                  save_dir=self.path_to_save_iou_results)
        self.rel_poses_stamped = []
        self.rel_pose_of_vcur = None
        self.rel_pose_vcur_to_loc = None
        self.found_loop_closure = False
        self.need_to_change_vcur = False
        self.path = []
        self.device = torch.device('cuda:0')

        self.graph = TopologicalGraph(place_recognition_index=self.place_recognition_index,
                                      inline_registration_model=self.inline_registration_model,
                                      inline_registration_score_threshold=self.inline_registration_score_threshold,
                                      grid_resolution=self.grid_resolution,
                                      grid_radius=self.grid_radius,
                                      max_grid_range=self.max_grid_range)
        if self.path_to_load_graph is not None:
            print('Loading graph from {}'.format(self.path_to_load_graph))
            self.graph.load_from_json(self.path_to_load_graph)
            print('Done!')
            # print('Grid max of graph vertex 0:', self.graph.vertices[0]['grid'].grid.max())
        if self.path_to_save_logs is not None:
            path_to_save_localization_results = os.path.join(self.path_to_save_logs, 'localization_results')
        else:
            path_to_save_localization_results = None
        self.localizer = Localizer(self.graph, 
                                   registration_model=self.registration_model,
                                   registration_score_threshold=self.registration_score_threshold,
                                   save_dir=path_to_save_localization_results,
                                   top_k=self.top_k)

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
        self.localization_timeout = topomap_config['localization_timeout']
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
        if self.path_to_save_logs is not None:
            path_to_save_registration_results = os.path.join(self.path_to_save_logs, 'test_registration')
        else:
            path_to_save_registration_results = None
        self.registration_model = get_registration_model(registration_config, \
                                                         save_dir=path_to_save_registration_results)
        self.registration_score_threshold = registration_config['score_threshold']
        inline_registration_config = config['scan_matching_along_edge']
        if self.path_to_save_logs is not None:
            path_to_save_inline_registration_results = os.path.join(self.path_to_save_logs, 'test_inline_registration')
        else:
            path_to_save_inline_registration_results = None
        self.inline_registration_model = get_registration_model(inline_registration_config, \
                                                                save_dir=path_to_save_inline_registration_results)
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
        rel_x_old, rel_y_old, rel_theta_old = self.get_rel_pose_from_stamp(self.current_stamp)[0]
        cur_grid_transformed = self.cur_grid.copy()
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

    def get_path_to_metric_goal(self, x, y):
        min_length = np.inf
        best_path = None
        print('Goal coords:', x, y)
        while self.last_vertex is None:
            print('Waiting for localization to create path...')
            time.sleep(0.5)
        for v_id, v in enumerate(self.graph.vertices):
            rel_pose_in_v = get_rel_pose(*v['pose_for_visualization'], x, y, 0)
            if v['grid'].is_inside(*rel_pose_in_v):
                print('Goal is inside vertex {} with coords ({}, {})'.format(v_id, v['pose_for_visualization'][0], v['pose_for_visualization'][1]))
                path, length = self.graph.get_path_with_length(self.last_vertex_id, v_id)
                assert path is not None
                if length < min_length:
                    min_length = length
                    best_path = path
        if best_path is None:
            print('PATH TO METRIC GOAL NOT FOUND!!!')
        return best_path

    def find_loop_closure(self, vertex_ids, dists):
        self.found_loop_closure = False
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
                    # ux, uy, _ = self.graph.get_vertex(u)['pose_for_visualization']
                    # vx, vy, _ = self.graph.get_vertex(v)['pose_for_visualization']
                    # print('u:', ux, uy)
                    # print('v:', vx, vy)
                    # print('Path in graph:', path_len)
                    # print('Path through cur:', dst_through_cur)
                    self.found_loop_closure = True
                    self.path = path
                    break
            if self.found_loop_closure:
                break
        return self.found_loop_closure

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

    def reattach_by_edge(self, require_match=True):
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
        cur_grid_transformed = self.cur_grid.copy()
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
                self.last_successful_match_time = self.current_stamp
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
            print('Reset rel_poses_stamped to time', self.current_stamp)
            self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        else:
            print('Failed to match current cloud to vertex {}!'.format(nearest_vertex_id))

        return changed

    def reattach_by_localization(self, iou_threshold, localized_stamp):
        vertex_ids = self.localization_results['vertex_ids_matched']
        rel_poses = self.localization_results['rel_poses']
        print('Reattach by localization')
        if len(self.rel_poses_stamped) > 0 and localized_stamp < self.rel_poses_stamped[0][0]:
            print('Old localization 1! Ignore it')
            print((self.rel_poses_stamped[0][0] - localized_stamp))
            return False
        found_proper_vertex = False
        self.rel_pose_vcur_to_loc, _ = self.get_rel_pose_from_stamp(localized_stamp)
        # First try to pass the nearest edge
        for i, v in enumerate(vertex_ids):
            # if v == self.last_vertex_id:
            #     continue
            pred_rel_pose_vcur_to_v = apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_poses[i]))
            rel_pose_robot_to_loc = get_rel_pose(*self.get_rel_pose_since_localization(), *rel_poses[i])
            print('Rel pose robot to loc:', rel_pose_robot_to_loc)
            iou = self.cur_grid.get_iou(self.graph.get_vertex(v)['grid'], *rel_pose_robot_to_loc, save=False)
            vx, vy, vtheta = self.graph.get_vertex(v)['pose_for_visualization']
            x, y, theta = get_rel_pose(*self.last_vertex['pose_for_visualization'], 
                                       *self.graph.get_vertex(v)['pose_for_visualization'])
            dx, dy, dtheta = get_rel_pose(x, y, theta, *self.rel_pose_of_vcur)
            dst = np.sqrt(dx ** 2 + dy ** 2)
            if dst > self.drift_coef * (self.current_stamp - self.last_successful_match_time) + 10:
                print('Vertex {} is too far to match'.format(v))
                continue
            print('v:', v)
            print('IoU between current state and ({}, {}) is {}'.format(vx, vy, iou))
            print('IoU threshold is {}'.format(iou_threshold))
            print('Need to change vcur:', self.need_to_change_vcur)
            if iou > iou_threshold or self.need_to_change_vcur:
                self.last_successful_match_time = localized_stamp
                found_proper_vertex = True
                print('\n\n\n Change to vertex {} with coords ({}, {})\n\n\n'.format(v, vx, vy))
                #last_x, last_y, last_theta = self.last_vertex['pose_for_visualization']
                if self.mode == 'mapping':
                    self.graph.add_edge(self.last_vertex_id, v, *pred_rel_pose_vcur_to_v)
                self.last_vertex_id = v
                self.need_to_change_vcur = False
                self.last_vertex = self.graph.get_vertex(v)
                _, rel_pose_after_localization = self.get_rel_pose_from_stamp(localized_stamp, verbose=True)
                pred_rel_pose = apply_pose_shift(rel_poses[i], *rel_pose_after_localization)
                self.rel_pose_cnt += 1
                self.rel_pose_of_vcur = pred_rel_pose
                self.rel_pose_vcur_to_loc = apply_pose_shift(self.graph.inverse_transform(*pred_rel_pose_vcur_to_v), *self.rel_pose_vcur_to_loc)
                self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
                #self.localization_time = 0
                return True
        return False

    def add_new_vertex(self, vertex_ids, rel_poses):
        new_vertex_id = self.graph.add_vertex(self.global_pose_for_visualization, self.cur_desc, self.cur_grid)
        new_vertex = self.graph.get_vertex(new_vertex_id)
        pose_stamped, new_rel_pose_of_vcur = self.get_rel_pose_from_stamp(self.current_stamp)
        if self.last_vertex is not None:
            #true_rel_pose = get_rel_pose(*self.last_vertex['pose_for_visualization'], *new_vertex['pose_for_visualization'])
            self.graph.add_edge(self.last_vertex_id, new_vertex_id, *pose_stamped)
        self.rel_pose_of_vcur = new_rel_pose_of_vcur
        if self.rel_pose_vcur_to_loc is not None:
            self.rel_pose_vcur_to_loc = get_rel_pose(*pose_stamped, *self.rel_pose_vcur_to_loc)
        for v, rel_pose in zip(vertex_ids, rel_poses):
            if self.rel_pose_vcur_to_loc is not None and rel_pose is not None:
                pred_rel_pose = apply_pose_shift(self.rel_pose_vcur_to_loc, *self.graph.inverse_transform(*rel_pose))
                #if np.sqrt(pred_rel_pose[0] ** 2 + pred_rel_pose[1] ** 2) < 5:
                self.graph.add_edge(new_vertex_id, v, *pred_rel_pose)
        self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        self.last_vertex_id = new_vertex_id
        self.need_to_change_vcur = False
        self.last_vertex = new_vertex

    def init_localization(self):
        if self.start_location is not None:
            self.last_vertex_id = self.start_location
            self.need_to_change_vcur = False
            self.last_vertex = self.graph.get_vertex(self.start_location)
            if self.start_local_pose is not None:
                self.rel_pose_of_vcur = self.start_local_pose
                self.rel_pose_vcur_to_loc = self.rel_pose_of_vcur
                self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        else:
            localized_state = self.localizer.get_localized_state()
            start_time = time.time()
            while localized_state['vertex_ids_matched'] is None or len(localized_state['vertex_ids_matched']) == 0:
                print('Still waiting for localization... Try to move forward-backward slightly')
                time.sleep(1.0)
                localized_state = self.localizer.get_localized_state()
                if self.mode == 'mapping' and time.time() - start_time > self.localization_timeout:
                    print('Localization timed out. Add new vertex at start')
                    self.add_new_vertex([], [])
                    break
            vertex_ids = localized_state['vertex_ids_matched']
            rel_poses = localized_state['rel_poses']
            if vertex_ids is not None and len(vertex_ids) > 0:
                print('Initially attached to vertex {} from localization'.format(vertex_ids[0]))
                self.last_vertex_id = vertex_ids[0]
                self.need_to_change_vcur = False
                self.last_vertex = self.graph.get_vertex(vertex_ids[0])
                self.rel_pose_of_vcur = self.graph.inverse_transform(*rel_poses[0])
                print('Rel pose of vcur is set to', self.rel_pose_of_vcur)
                self.rel_pose_vcur_to_loc = self.rel_pose_of_vcur
                self.current_stamp = self.localizer.localized_stamp
                self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]

    def _preprocess_input(self, input_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preprocess input data."""
        out_dict: Dict[str, Tensor] = {}
        for key in input_data:
            if key.startswith("image_"):
                out_dict[f"images_{key[6:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key.startswith("mask_"):
                out_dict[f"masks_{key[5:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key == "pointcloud_lidar_coords":
                quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                    coordinates=input_data["pointcloud_lidar_coords"],
                    features=input_data["pointcloud_lidar_feats"],
                    quantization_size=self.pointcloud_quantization_size,
                )
                out_dict["pointclouds_lidar_coords"] = ME.utils.batched_coordinates([quantized_coords]).to(
                    self.device
                )
                out_dict["pointclouds_lidar_feats"] = quantized_feats.to(self.device)
        return out_dict

    def process_observations(self, img_front, img_back, cur_cloud, cur_curbs, x, y, theta):
        # Extract descriptor from cloud and images
        input_data = {
                     'pointcloud_lidar_coords': torch.Tensor(cur_cloud[:, :3]).cuda(),
                     'pointcloud_lidar_feats': torch.ones((cur_cloud.shape[0], 1)).cuda(),
                     }
        if img_front is not None:
            img_front_tensor = torch.Tensor(img_front).cuda()
            img_front_tensor = torch.permute(img_front_tensor, (2, 0, 1))
            input_data['image_front'] = img_front_tensor
        if img_back is not None:
            img_back_tensor = torch.Tensor(img_back).cuda()
            img_back_tensor = torch.permute(img_back_tensor, (2, 0, 1))
            input_data['image_back'] = img_back_tensor
        batch = self._preprocess_input(input_data)
        self.cur_desc = self.place_recognition_model(batch)["final_descriptor"].detach().cpu().numpy()
        if len(self.cur_desc.shape) == 1:
                self.cur_desc = self.cur_desc[np.newaxis, :]
        # Project cloud into a grid
        self.cur_grid.update_from_cloud_and_transform(cur_cloud, x, y, -theta)
        if cur_curbs is not None:
            self.cur_grid.update_curbs_from_cloud(cur_curbs)
        else:
            print('NO CURBS!')

    def update_rel_pose_of_vcur_by_odom(self, cur_odom_pose):
        x, y, theta = cur_odom_pose
        print('Odom pose:', self.odom_pose)
        print('Cur odom pose:', cur_odom_pose)
        if self.odom_pose is not None:
            rel_x, rel_y, rel_theta = get_rel_pose(*self.odom_pose, x, y, theta)
        else:
            rel_x, rel_y, rel_theta = x, y, theta
        self.odom_pose = [x, y, theta]
        if self.rel_pose_of_vcur is None:
            print('Rel pose of vcur is None, initialize it as ({}, {}, {})'.format(rel_x, rel_y, rel_theta))
            self.rel_pose_of_vcur = [rel_x, rel_y, rel_theta]
        else:
            # print('Apply pose shift:', rel_x, rel_y, rel_theta)
            self.rel_pose_of_vcur = apply_pose_shift(self.rel_pose_of_vcur, rel_x, rel_y, rel_theta)
            # print('Update rel pose of vcur from odom:', self.rel_pose_of_vcur)
        self.rel_poses_stamped.append([self.current_stamp] + self.rel_pose_of_vcur)

    def update(self, global_pose_for_visualization, cur_odom_pose, img_front, img_back, cur_cloud, cur_curbs):
        if self.odom_pose is None:
            self.odom_pose = cur_odom_pose
        x, y, theta = get_rel_pose(*cur_odom_pose, *self.odom_pose)
        # Update rel_pose_of_vcur by odometry
        self.update_rel_pose_of_vcur_by_odom(cur_odom_pose)
        # Update localizer and localized state
        self.process_observations(img_front, img_back, cur_cloud, cur_curbs, x, y, theta)
        self.global_pose_for_visualization = global_pose_for_visualization
        self.localizer.update_current_state(self.global_pose_for_visualization, self.cur_desc, self.cur_grid, self.current_stamp)
        if self.last_vertex is None:
            self.init_localization()
        self.localization_results = self.localizer.get_localized_state()
        if self.localization_results is not None and self.localization_results['timestamp'] is not None:
            self.localization_time = self.localization_results['timestamp']
        localized_stamp = self.localization_results['timestamp']
        localization_is_fresh = (len(self.rel_poses_stamped) == 0 or localized_stamp is None or localized_stamp >= self.rel_poses_stamped[0][0] - 1e-3)
        print('\n\n\n Localized in vertices: {}. Actual: {} \n\n\n'.format(self.localization_results['vertex_ids_matched'], 
                                                                          localization_is_fresh))
        # print('Rel poses:', self.localization_results['rel_poses'])

        if len(self.rel_poses_stamped) == 0 or localized_stamp is None or localized_stamp >= self.rel_poses_stamped[0][0] - 1e-3:
            vertex_ids = self.localization_results['vertex_ids_matched']
            rel_poses = self.localization_results['rel_poses']
            if vertex_ids is not None and rel_poses is not None:
                dists = [np.sqrt(x ** 2 + y ** 2) for x, y, theta in rel_poses]
                if self.find_loop_closure(vertex_ids, dists):
                    print('Found loop closure. Add new vertex to close loop')
                    self.add_new_vertex(vertex_ids, rel_poses)
                    return
        else:
            print('Could not check loop closure - old localization')

        t1 = time.time()
        if cur_cloud is None:
            print('No point cloud received!')
            return
        changed = self.reattach_by_edge(require_match=True)
        #last_x, last_y, _ = self.last_vertex['pose_for_visualization']
        inside_vcur = self.is_inside_vcur()
        # print('Rel pose of vcur:', self.rel_pose_of_vcur)
        iou = self.cur_grid.get_iou(self.last_vertex['grid'], *self.graph.inverse_transform(*self.rel_pose_of_vcur), \
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
            #print(self.path[0], self.last_vertex_id)
            print('Changed:', changed)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if not changed:
                print('Localization dt:', time.time() - self.localization_time)
                if time.time() - self.localization_results['timestamp'] < 5:
                    #print('Localized stamp:', self.localizer.localized_stamp)
                    changed = self.reattach_by_localization(self.cur_iou, self.localization_results['timestamp'])
                    print('Changed from localization:', changed)
                    if not changed:
                        if self.mode == 'mapping':
                            print('No proper vertex to change. Add new vertex')
                            if localization_is_fresh:
                                vertex_ids = self.localization_results['vertex_ids_matched']
                                rel_poses = self.localization_results['rel_poses']
                            else:
                                vertex_ids = []
                                rel_poses = []
                            self.add_new_vertex(vertex_ids, rel_poses)
                else:
                    if self.mode == 'mapping':
                        print('No recent localization. Add new vertex')
                        self.add_new_vertex([], [])
                    else:
                        print('No recent localization')
                    #     self.reattach_by_edge(require_match=False)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if not changed and self.mode == 'localization':
                self.reattach_by_edge(require_match=False)

    def save_graph(self):
        if self.path_to_save_graph is not None:
            self.graph.save_to_json(self.path_to_save_graph)
        print('N of localizer calls:', self.localizer.cnt)
        print('N of localization fails:', self.localizer.n_loc_fails)