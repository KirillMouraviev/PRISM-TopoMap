import numpy as np
import os
import cv2
import sys
import time
import rospy
import yaml
import torch
from localization import Localizer
from utils import *
from topo_graph import TopologicalGraph
from local_grid import LocalGrid
from models import get_place_recognition_model, get_registration_model
from salsanext_inference import SalsaNextInference
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
        if self.path_to_save_logs is not None and not os.path.exists(os.path.join(self.path_to_save_logs, 'clouds')):
            os.mkdir(os.path.join(self.path_to_save_logs, 'clouds'))
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
        layer_names = ['occupancy', 'density_map', 'height_map', 'semantic']
        if config['input']['pointcloud']['subscribe_to_curbs']:
            layer_names.append('curbs')
        self.cur_grid = LocalGrid(resolution=self.grid_resolution, 
                                  radius=self.grid_radius, 
                                  max_range=self.max_grid_range,
                                  floor_height=self.floor_height, ceil_height=self.ceil_height,
                                  semantic_probability_threshold=self.semantic_probability_threshold,
                                  layer_names=layer_names,
                                  save_dir=self.path_to_save_iou_results)
        self.cur_cloud = None
        self.rel_poses_stamped = []
        self.rel_pose_of_vcur = None
        self.rel_pose_vcur_to_loc = None
        self.found_loop_closure = False
        self.need_to_change_vcur = False
        self.matched = False
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
        print('Loading SalsaNext semantic model...')
        self.semantic_inference = SalsaNextInference(device='cuda:0', verbose=False)
        if self.path_to_save_logs is not None:
            path_to_save_localization_results = os.path.join(self.path_to_save_logs, 'localization_results')
        else:
            path_to_save_localization_results = None
        self.localizer = Localizer(self.graph, 
                                   registration_model=self.registration_model,
                                   registration_score_threshold=self.registration_score_threshold,
                                   save_dir=path_to_save_localization_results,
                                   top_k=self.top_k,
                                   semantic_inference=self.semantic_inference)

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
        self.diag_log_path = rospy.get_param('~diag_log_path', '/tmp/prism_topomap_diag.log')
        self._diag_file = open(self.diag_log_path, 'w')
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
        self.drift_coef_theta = topomap_config.get('drift_coef_theta', 0.01)
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
        self.semantic_probability_threshold = grid_config.get('semantic_probability_threshold', 0.3)
        # Registration
        registration_config = config['scan_matching']
        registration_config['voxel_downsample_size'] = grid_config['resolution']
        if self.path_to_save_logs is not None:
            path_to_save_registration_results = os.path.join(self.path_to_save_logs, 'test_registration')
        else:
            path_to_save_registration_results = None
        self.registration_model = get_registration_model(registration_config, \
                                                         save_dir=path_to_save_registration_results)
        self.registration_score_threshold = registration_config['score_threshold']
        inline_registration_config = config['scan_matching_along_edge']
        inline_registration_config['voxel_downsample_size'] = grid_config['resolution']
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

    def log_diag(self, event, **kwargs):
        parts = [f't={self.current_stamp:.3f}', f'event={event}', f'vertex={self.last_vertex_id}']
        if self.rel_pose_of_vcur is not None:
            parts.append(f'rel_pose=({self.rel_pose_of_vcur[0]:.2f},{self.rel_pose_of_vcur[1]:.2f},{self.rel_pose_of_vcur[2]:.2f})')
        parts.append(f'iou={self.cur_iou:.3f}')
        parts.append(f'matched={self.matched}')
        for k, v in kwargs.items():
            parts.append(f'{k}={v}')
        line = ' '.join(parts)
        print(f'[DIAG] {line}')
        self._diag_file.write(line + '\n')
        self._diag_file.flush()

    def get_estimated_pose(self):
        if self.last_vertex is None or self.rel_pose_of_vcur is None:
            return None
        return apply_pose_shift(self.last_vertex['pose_for_visualization'], *self.rel_pose_of_vcur)

    def log_pose_error(self):
        """Log GT vs estimated pose using the scan-synced GT pose (correct timestamps)."""
        if self.last_vertex is None or self.rel_pose_of_vcur is None:
            return
        if self.global_pose_for_visualization is None:
            return
        gt = self.global_pose_for_visualization
        est = self.get_estimated_pose()
        if est is None:
            return
        dx = gt[0] - est[0]
        dy = gt[1] - est[1]
        pos_err = float(np.sqrt(dx ** 2 + dy ** 2))
        yaw_err = float(abs(normalize(gt[2] - est[2])))
        yaw_err_deg = yaw_err * 180.0 / np.pi
        line = (
            f't={self.current_stamp:.3f} vertex={self.last_vertex_id} '
            f'gt=({gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f}) '
            f'est=({est[0]:.3f},{est[1]:.3f},{est[2]:.3f}) '
            f'pos_err={pos_err:.3f}m yaw_err={yaw_err:.3f}rad ({yaw_err_deg:.1f}deg)'
        )
        print(f'[POSE_ERR] {line}')
        if not hasattr(self, '_pose_err_file'):
            pose_err_path = rospy.get_param('~pose_error_log_path', '/tmp/prism_pose_error.log')
            if self.path_to_save_logs is not None:
                pose_err_path = os.path.join(self.path_to_save_logs, 'pose_error.log')
            os.makedirs(os.path.dirname(pose_err_path) or '.', exist_ok=True)
            self._pose_err_file = open(pose_err_path, 'w')
            self._pose_err_csv = open(pose_err_path.replace('.log', '.csv'), 'w')
            self._pose_err_csv.write('t,vertex,gt_x,gt_y,gt_yaw,est_x,est_y,est_yaw,pos_err_m,yaw_err_rad,yaw_err_deg\n')
        self._pose_err_file.write(line + '\n')
        self._pose_err_file.flush()
        self._pose_err_csv.write(
            f'{self.current_stamp:.3f},{self.last_vertex_id},'
            f'{gt[0]:.6f},{gt[1]:.6f},{gt[2]:.6f},'
            f'{est[0]:.6f},{est[1]:.6f},{est[2]:.6f},'
            f'{pos_err:.6f},{yaw_err:.6f},{yaw_err_deg:.3f}\n'
        )
        self._pose_err_csv.flush()
        if pos_err > 5.0:
            self.log_diag('pose_error_exceeded', pos_err=round(pos_err, 3), yaw_err_deg=round(yaw_err_deg, 1))

    def fill_semantic(self, grid, cloud):
        if cloud is None or grid is None or self.semantic_inference is None:
            return
        if getattr(grid, '_semantic_stamp', None) == self.current_stamp and grid.layers.get('semantic'):
            return
        _, _, _, probabilities = self.semantic_inference.predict(cloud)
        grid.update_semantic_from_cloud(cloud, probabilities)
        grid._semantic_stamp = self.current_stamp

    def correct_rel_pose(self, event=None):
        if self.last_vertex_id is None:
            return
        if self.cur_grid is None:
            return
        rel_x_old, rel_y_old, rel_theta_old = self.get_rel_pose_from_stamp(self.current_stamp)[0]
        self.fill_semantic(self.cur_grid, self.cur_cloud)
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
                    print('Loop: connect {} and {} through current position'.format(u, v))
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
        candidates = []
        # print('Vcur:', self.last_vertex_id)
        for vertex_id, pose_to_vertex in self.graph.adj_lists[self.last_vertex_id]:
            pose_diff = np.sqrt((pose_to_vertex[0] - self.rel_pose_of_vcur[0]) ** 2 + (pose_to_vertex[1] - self.rel_pose_of_vcur[1]) ** 2)
            candidates.append((pose_diff, vertex_id, pose_to_vertex))
        dist_to_vcur = np.sqrt(self.rel_pose_of_vcur[0] ** 2 + self.rel_pose_of_vcur[1] ** 2)
        print('Pose diffs and dist to vcur:', [c[0] for c in candidates], dist_to_vcur)
        # Neighbours no farther from the vehicle than the current vertex, nearest first.
        candidates = [c for c in candidates
                      if c[0] <= dist_to_vcur and c[0] < self.max_edge_length]
        candidates.sort(key=lambda c: c[0])
        if len(candidates) == 0:
            print('Could not find proper edge to change')
            return False
        print('Edge reattach candidates (nearest first):',
              [(vid, d) for d, vid, _ in candidates])

        changed = False
        matched_vertex_id = None
        matched_pose_on_edge = None
        self.fill_semantic(self.cur_grid, self.cur_cloud)
        for pose_diff, vertex_id, pose_on_edge in candidates:
            print('Trying edge neighbour {} (dist={:.3f})'.format(vertex_id, pose_diff))
            rel_pose_to_vertex = get_rel_pose(*self.rel_pose_of_vcur, *pose_on_edge)
            print('Rel pose of vcur:', self.rel_pose_of_vcur)
            print('Pose on edge:', pose_on_edge)
            print('Rel pose to vertex:', rel_pose_to_vertex)
            rel_pose_to_vertex_inv = self.graph.inverse_transform(*rel_pose_to_vertex)
            cur_grid_transformed = self.cur_grid.copy()
            cur_grid_transformed.transform(rel_pose_to_vertex_inv[0], rel_pose_to_vertex_inv[1], -rel_pose_to_vertex_inv[2])
            corr_x, corr_y, corr_theta = self.graph.get_transform_to_vertex(vertex_id, cur_grid_transformed)
            print('Require match:', require_match)
            if corr_x is not None:
                print('Old transform:', rel_pose_to_vertex)
                corr_x_inv, corr_y_inv, corr_theta_inv = self.graph.inverse_transform(corr_x, corr_y, corr_theta)
                x, y, theta = apply_pose_shift(rel_pose_to_vertex, corr_x_inv, corr_y_inv, corr_theta_inv)
                print('Scan matching transform:', x, y, theta)
                diff = np.sqrt((x - rel_pose_to_vertex[0]) ** 2 + (y - rel_pose_to_vertex[1]) ** 2)
                print('Diff:', diff)
                if diff < self.local_jump_threshold or not self.matched:
                    print('\n\n\n Change to vertex {} by edge \n\n\n'.format(vertex_id))
                    self.last_successful_match_time = self.current_stamp
                    changed = True
                    self.matched = True
                    self.rel_pose_of_vcur = self.graph.inverse_transform(x, y, theta)
                    matched_vertex_id = vertex_id
                    matched_pose_on_edge = pose_on_edge
                    break
                else:
                    print('Big jump! Ignore this match')
                    self.log_diag('edge_match_rejected', jump=diff, threshold=self.local_jump_threshold,
                                  nearest_vertex=vertex_id)
            else:
                print('Failed to match current cloud to vertex {}!'.format(vertex_id))

        if not changed and not require_match:
            # Forced jump: use the nearest eligible neighbour.
            pose_diff, vertex_id, pose_on_edge = candidates[0]
            rel_pose_to_vertex = get_rel_pose(*self.rel_pose_of_vcur, *pose_on_edge)
            changed = True
            self.matched = False
            print('\n\n\n Change to vertex {} by edge without matching\n\n\n'.format(vertex_id))
            self.log_diag('forced_edge_jump', nearest_vertex=vertex_id, require_match=False)
            self.rel_pose_of_vcur = self.graph.inverse_transform(*rel_pose_to_vertex)
            matched_vertex_id = vertex_id
            matched_pose_on_edge = pose_on_edge

        if changed:
            self.last_vertex_id = matched_vertex_id
            self.need_to_change_vcur = False
            self.last_vertex = self.graph.get_vertex(self.last_vertex_id)
            if self.rel_pose_vcur_to_loc is not None:
                self.rel_pose_vcur_to_loc = apply_pose_shift(
                    self.graph.inverse_transform(*matched_pose_on_edge), *self.rel_pose_vcur_to_loc)
            print('Reset rel_poses_stamped to time', self.current_stamp)
            self.rel_poses_stamped = [[self.current_stamp] + self.rel_pose_of_vcur]
        elif require_match:
            print('Failed to match current cloud to any edge neighbour among {}'.format(
                [vid for _, vid, _ in candidates]))

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
            max_dst = self.drift_coef * (self.current_stamp - self.last_successful_match_time) + 10
            max_dtheta = self.drift_coef_theta * (self.current_stamp - self.last_successful_match_time) + 0.3
            abs_dtheta = abs(normalize(dtheta))
            if dst > max_dst or abs_dtheta > max_dtheta:
                reason = 'distance' if dst > max_dst else 'theta'
                print('Vertex {} is too far to match (reason={}, dst={:.2f}/{:.2f}, dtheta={:.3f}/{:.3f})'.format(
                    v, reason, dst, max_dst, abs_dtheta, max_dtheta))
                self.log_diag('loc_reattach_rejected_distance', candidate=v, distance=dst,
                              max_allowed=max_dst, dtheta=abs_dtheta, max_dtheta=max_dtheta, reason=reason)
                continue
            print('v:', v)
            print('IoU between current state and ({}, {}) is {}'.format(vx, vy, iou))
            print('IoU threshold is {}'.format(iou_threshold))
            print('Need to change vcur:', self.need_to_change_vcur)
            if iou > iou_threshold or self.need_to_change_vcur:
                self.last_successful_match_time = localized_stamp
                found_proper_vertex = True
                self.matched = True
                print('\n\n\n Change to vertex {} with coords ({}, {})\n\n\n'.format(v, vx, vy))
                self.log_diag('loc_reattach_success', candidate=v, iou=iou)
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
        self.fill_semantic(self.cur_grid, self.cur_cloud)
        new_vertex_id = self.graph.add_vertex(self.global_pose_for_visualization, self.cur_desc, self.cur_grid)
        np.savez(os.path.join(self.path_to_save_logs, 'clouds', '{}.npz'.format(new_vertex_id)), self.cur_cloud)
        new_vertex = self.graph.get_vertex(new_vertex_id)
        pose_stamped, new_rel_pose_of_vcur = self.get_rel_pose_from_stamp(self.current_stamp)
        if self.last_vertex is not None:
            #true_rel_pose = get_rel_pose(*self.last_vertex['pose_for_visualization'], *new_vertex['pose_for_visualization'])
            self.graph.add_edge(self.last_vertex_id, new_vertex_id, *pose_stamped)
        self.rel_pose_of_vcur = new_rel_pose_of_vcur
        if self.rel_pose_vcur_to_loc is not None:
            self.rel_pose_vcur_to_loc = get_rel_pose(*pose_stamped, *self.rel_pose_vcur_to_loc)
        # print('Vertex ids: {}, rel poses: {}'.format(vertex_ids, rel_poses))
        for v, rel_pose in zip(vertex_ids, rel_poses):
            # print('Rel pose vcur to loc:', self.rel_pose_vcur_to_loc)
            # print('Rel pose:', rel_pose)
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
            self.matched = True
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
                self.matched = True
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
        self.cur_cloud = cur_cloud
        self.cur_grid.update_from_cloud_and_transform(cur_cloud, x, y, -theta)
        if cur_curbs is not None:
            self.cur_grid.update_curbs_from_cloud(cur_curbs)
        # else:
        #     print('NO CURBS!')

    def update_rel_pose_of_vcur_by_odom(self, cur_odom_pose):
        x, y, theta = cur_odom_pose
        # print('Odom pose:', self.odom_pose)
        # print('Cur odom pose:', cur_odom_pose)
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
        # print('Update from stamp:', self.current_stamp)
        start_time = time.time()
        if self.odom_pose is None:
            self.odom_pose = cur_odom_pose
        x, y, theta = get_rel_pose(*cur_odom_pose, *self.odom_pose)
        # Update rel_pose_of_vcur by odometry
        self.update_rel_pose_of_vcur_by_odom(cur_odom_pose)
        # Update localizer and localized state
        self.process_observations(img_front, img_back, cur_cloud, cur_curbs, x, y, theta)
        self.global_pose_for_visualization = global_pose_for_visualization
        self.localizer.update_current_state(self.global_pose_for_visualization, self.cur_cloud, self.cur_desc, self.cur_grid, self.current_stamp)
        if self.last_vertex is None:
            self.init_localization()
        self.localization_results = self.localizer.get_localized_state()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # vertex_ids_matched_refined = []
        # rel_poses_refined = []
        # for i, v in enumerate(self.localization_results['vertex_ids_matched']):
        #     true_rel_pose = get_rel_pose(*self.graph.get_vertex(v)['pose_for_visualization'],
        #                                  *self.localization_results['global_pose_for_visualization'])
        #     true_dst = np.sqrt(true_rel_pose[0] ** 2 + true_rel_pose[1] ** 2)
        #     # print('v:', v)
        #     # print('Pose of localization:', self.localization_results['global_pose_for_visualization'])
        #     # print('Pose of v:', self.graph.get_vertex(v)['pose_for_visualization'])
        #     # print('Pred rel pose:', self.localization_results['rel_poses'][i])
        #     # print('True rel pose:', true_rel_pose)
        #     if true_dst < 2 * self.max_edge_length:
        #         vertex_ids_matched_refined.append(v)
        #         rel_poses_refined.append(true_rel_pose)#self.localization_results['rel_poses'][i])
        #     else:
        #         print('Vertex {} is too far (distance is {}), remove it from localization'.format(v, true_dst))
        # self.localization_results['vertex_ids_matched'] = vertex_ids_matched_refined
        # self.localization_results['rel_poses'] = rel_poses_refined
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.localization_results is not None and self.localization_results['timestamp'] is not None:
            self.localization_time = self.localization_results['timestamp']
        localized_stamp = self.localization_results['timestamp']
        localization_is_fresh = (len(self.rel_poses_stamped) == 0 or localized_stamp is None or localized_stamp >= self.rel_poses_stamped[0][0] - 1e-3)
        print('Localized in vertices: {}. Actual: {}'.format(self.localization_results['vertex_ids_matched'], 
                                                                          localization_is_fresh))
        if localization_is_fresh:
            self.rel_pose_vcur_to_loc = self.get_rel_pose_from_stamp(localized_stamp)[0]
        # print('Rel pose vcur to loc:', self.rel_pose_vcur_to_loc)
        # print('Rel poses:', self.localization_results['rel_poses'])

        if self.mode == 'mapping' and localization_is_fresh:
            vertex_ids = list(self.localization_results['vertex_ids_matched'])
            rel_poses = list(self.localization_results['rel_poses'])
            if vertex_ids is not None and rel_poses is not None:
                if self.last_vertex_id not in vertex_ids:
                    vertex_ids.append(self.last_vertex_id)
                    rel_poses.append(self.graph.inverse_transform(*self.rel_pose_vcur_to_loc))
                dists = [np.sqrt(x ** 2 + y ** 2) for x, y, theta in rel_poses]
                if self.find_loop_closure(vertex_ids, dists):
                    print('\n\n\nFound loop closure. Add new vertex to close loop\n\n\n')
                    self.add_new_vertex(vertex_ids, rel_poses)
                    return
        else:
            print('Could not check loop closure - old localization')

        if cur_cloud is None:
            print('No point cloud received!')
            return
        changed = self.reattach_by_edge(require_match=True)
        #last_x, last_y, _ = self.last_vertex['pose_for_visualization']
        inside_vcur = self.is_inside_vcur()
        print('Rel pose of vcur:', self.rel_pose_of_vcur)
        iou = self.cur_grid.get_iou(self.last_vertex['grid'], *self.graph.inverse_transform(*self.rel_pose_of_vcur), \
                               save=False, cnt=self.iou_cnt)
        self.iou_cnt += 1
        self.cur_iou = iou
        # print('IoU:', iou)
        # print('Vcur:', self.last_vertex_id)
        dst = np.sqrt(self.rel_pose_of_vcur[0] ** 2 + self.rel_pose_of_vcur[1] ** 2)
        if not inside_vcur or iou < self.iou_threshold or dst > self.max_edge_length:
            self.need_to_change_vcur = True
            reason = 'outside' if not inside_vcur else ('low_iou' if iou < self.iou_threshold else 'too_far')
            self.log_diag('need_vertex_change', reason=reason, inside=inside_vcur, dst=dst)
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
                print('Localization dt:', self.current_stamp - self.localization_time)
                if self.current_stamp - self.localization_time < 5:
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
                        self.log_diag('no_recent_localization',
                                      loc_age=self.current_stamp - self.localization_results['timestamp'])
                    #     self.reattach_by_edge(require_match=False)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if not changed and self.mode == 'localization':
                self.reattach_by_edge(require_match=False)
        self.log_pose_error()
        print('Update done at {} seconds, now is {}'.format(time.time() - start_time, rospy.Time.now().to_sec()))

    def save_graph(self):
        if self.path_to_save_graph is not None:
            self.graph.save_to_json(self.path_to_save_graph)
        print('N of localizer calls:', self.localizer.cnt)
        print('N of localization fails:', self.localizer.n_loc_fails)