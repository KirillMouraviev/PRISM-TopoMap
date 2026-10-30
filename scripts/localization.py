import os
import numpy as np
import torch
import time
import rospy
import rospkg
from utils import apply_pose_shift
from copy import deepcopy
from threading import Lock
from scipy.spatial.transform import Rotation
import sys
rp = rospkg.RosPack()
path_to_prism_topomap = rp.get_path('prism_topomap')
sys.path.append(os.path.join(path_to_prism_topomap, 'submodules'))

class Localizer():
    def __init__(self, graph, registration_model,
                 registration_score_threshold=0.6, top_k=5, save_dir=None,
                 semantic_inference=None):
        self.graph = graph
        self.registration_pipeline = registration_model
        self.registration_score_threshold = registration_score_threshold
        self.top_k = top_k
        self.descriptor = None
        self.grid = None
        self.cloud = None
        self.semantic_inference = semantic_inference
        self.global_pose_for_visualization = None
        self.stamp = None
        self.localized_x = None
        self.localized_y = None
        self.localized_theta = None
        self.localized_stamp = 0
        self.vertex_ids_matched = None
        self.vertex_ids_unmatched = None
        self.dists = None
        self.rel_poses = None
        self.cnt = 0
        self.n_loc_fails = 0
        self.tests_dir = save_dir
        if self.tests_dir is not None and not os.path.exists(self.tests_dir):
            os.mkdir(self.tests_dir)
        self.mutex = Lock()
        self.device = torch.device('cuda:0')

    def save_reg_test_data(self, ref_grid, start_global_pose, vertex_ids, transforms, pr_scores, reg_scores, reg_test_ids, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir_ref_grid = os.path.join(save_dir, 'ref_grid')
        ref_grid.save(save_dir_ref_grid)
        # np.savez(os.path.join(save_dir, 'ref_cloud.npz'), self.cloud)
        #print('Mean of the ref cloud:', self.localized_cloud[:, :3].mean())
        tf_data = []
        if start_global_pose is not None:
            gt_pose_data = [list(start_global_pose)]
        else:
            print('No global pose for visualization to save reg test data!')
            return
        for idx, tf in zip(vertex_ids, transforms):
            if idx >= 0:
                vertex_dict = self.graph.vertices[idx]
                save_dir_cand_grid = os.path.join(save_dir, 'cand_grid_{}'.format(idx))
                x, y, theta = vertex_dict['pose_for_visualization']
                grid = vertex_dict['grid']
                grid.save(save_dir_cand_grid)
                #print('Grid max:', grid.max())
                #print('GT x, y, theta:', x, y, theta)
                if tf is not None:
                    tf_data.append([idx] + list(tf))
                else:
                    tf_data.append([idx, 0, 0, 0, 0, 0, 0])
                gt_pose_data.append([x, y, theta])
        #print('TF data:', tf_data)
        np.savetxt(os.path.join(save_dir, 'gt_poses.txt'), np.array(gt_pose_data))
        np.savetxt(os.path.join(save_dir, 'transforms.txt'), np.array(tf_data))
        np.savetxt(os.path.join(save_dir, 'pr_scores.txt'), np.array(pr_scores))
        np.savetxt(os.path.join(save_dir, 'reg_scores.txt'), np.array(reg_scores))
        np.savetxt(os.path.join(save_dir, 'reg_test_ids.txt'), np.array(reg_test_ids).astype(int))

    def update_current_state(self, global_pose_for_visualization, cur_cloud, cur_desc, cur_grid, timestamp):
        self.mutex.acquire()
        print('Update current state')
        self.global_pose_for_visualization = global_pose_for_visualization
        self.descriptor = cur_desc
        self.grid = cur_grid
        self.cloud = cur_cloud
        self.stamp = timestamp
        self.mutex.release()

    def write_localized_state(self, vertex_ids_matched, rel_poses, vertex_ids_pr_unmatched, start_global_pose, start_stamp):
        self.mutex.acquire()
        if start_global_pose is not None:
            self.localized_x, self.localized_y, self.localized_theta = start_global_pose
        self.localized_stamp = start_stamp
        self.vertex_ids_matched = vertex_ids_matched
        self.rel_poses = rel_poses
        self.vertex_ids_unmatched = vertex_ids_pr_unmatched
        self.mutex.release()

    def get_current_state(self):
        self.mutex.acquire()
        result = {
            'global_pose_for_visualization': self.global_pose_for_visualization, 
            'descriptor': self.descriptor.copy(), 
            'grid': self.grid.copy(), 
            'cloud': None if self.cloud is None else self.cloud.copy(),
            'timestamp': self.stamp
            }
        self.mutex.release()
        return result

    def get_localized_state(self):
        self.mutex.acquire()
        result = {
            'vertex_ids_matched': deepcopy(self.vertex_ids_matched), 
            'rel_poses': deepcopy(self.rel_poses),
            'vertex_ids_unmatched': deepcopy(self.vertex_ids_unmatched),
            'global_pose_for_visualization': (self.localized_x, self.localized_y, self.localized_theta),
            'timestamp': self.localized_stamp}
        self.mutex.release()
        return result

    def localize(self):
        if self.stamp is None:
            print('Waiting for message to initialize localizer...')
            return None, None, None
        print('\nLocalize from stamp {}, now is {}'.format(self.stamp, rospy.Time.now().to_sec()))
        start_time = time.time()
        vertex_ids = []
        rel_poses = []
        current_state = self.get_current_state()
        start_global_pose = current_state['global_pose_for_visualization']
        start_stamp = current_state['timestamp']
        start_grid = current_state['grid']
        start_desc = current_state['descriptor']
        start_cloud = current_state.get('cloud')
        if start_grid is not None and self.semantic_inference is not None and start_cloud is not None:
            _, _, _, probabilities = self.semantic_inference.predict(start_cloud)
            start_grid.update_semantic_from_cloud(start_cloud, probabilities)
        if start_grid is not None:
            dists, pred_i = self.graph.index.search(start_desc, self.top_k)
            pr_scores = dists[0]
            pred_i = pred_i[0]
            pred_tf = []
            pred_i_filtered = []
            reg_scores = []
            reg_test_ids = []
            for i, idx in enumerate(pred_i):
                if idx < 0:
                    continue
                cand_vertex_dict = self.graph.get_vertex(idx)
                cand_grid = cand_vertex_dict['grid'].copy()
                grid_copy = start_grid.copy()
                start_time = time.time()
                ref_grid_tensor = torch.Tensor(grid_copy.layers['occupancy']).to(self.device)
                cand_grid_tensor = torch.Tensor(cand_grid.layers['occupancy']).to(self.device)
                reg_test_ids.append(self.registration_pipeline.cnt)
                transform, score = self.registration_pipeline.infer(
                    ref_grid_tensor,
                    grid_copy.layers.get('semantic', []),
                    cand_grid_tensor,
                    cand_grid.layers.get('semantic', []),
                    verbose=False)
                print('Semantic scan matching time:', time.time() - start_time)
                if transform is None or score is None:
                    transform = None
                    score = 0.0
                reg_scores.append(score)
                print('Registration score of vertex {} is {}'.format(idx, score))
                if transform is None or score < self.registration_score_threshold:
                    pred_i_filtered.append(-1)
                    pred_tf.append([0, 0, 0, 0, 0, 0])
                else:
                    tf_matrix = cand_grid.get_tf_matrix_xy(*transform)
                    pred_i_filtered.append(idx)
                    tf_rotation = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
                    tf_translation = tf_matrix[:3, 3]
                    pred_tf.append(list(tf_rotation) + list(tf_translation))
            if self.tests_dir is not None:
                save_dir = os.path.join(self.tests_dir, 'test_{}'.format(self.cnt))
                self.cnt += 1
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                self.save_reg_test_data(start_grid, start_global_pose, pred_i, pred_tf, pr_scores, reg_scores, reg_test_ids, save_dir)
            vertex_ids_pr_unmatched = [idx for idx in pred_i if idx not in pred_i_filtered]
            #print('Matched indices:', [idx for idx in vertex_ids_pr if idx >= 0])
            #print('Unmatched indices:', vertex_ids_pr_unmatched)
            #rel_poses = [[tf[3], tf[4], tf[2]] for idx, tf in zip(pred_i_filtered, pred_tf) if idx >= 0]
        else:
            print('Localizer not initialized!')
            vertex_ids_pr = []
        pr_scores = [pr_scores[i] for i, idx in enumerate(pred_i_filtered) if idx >= 0]
        reg_scores = [reg_scores[i] for i, idx in enumerate(pred_i_filtered) if idx >= 0]
        transforms = [pred_tf[i] for i, idx in enumerate(pred_i_filtered) if idx >= 0]
        transforms = np.array(transforms)
        vertex_ids_pr = [i for i in pred_i_filtered if i >= 0]
        vertex_ids_pr_unmatched = [idx for idx in pred_i if idx not in pred_i_filtered and idx >= 0]
        if len(pred_i_filtered) == 0:
            self.n_loc_fails += 1
        #for i, v in enumerate(self.graph.vertices):
        for i, idx in enumerate(vertex_ids_pr):
            v = self.graph.vertices[idx]
            vertex_ids.append(idx)
            #print('Transform:', transforms[i])
            rel_poses.append(transforms[i, [3, 4, 2]])
            #print('True dist:', dist)
            #print('Descriptor dist:', pr_scores[i])
            #print('Reg score:', reg_scores[i])
        rel_poses = np.array(rel_poses)
        self.write_localized_state(vertex_ids, rel_poses, vertex_ids_pr_unmatched, start_global_pose, start_stamp)
        print('Localization done at {} seconds, now is {}\n'.format(time.time() - start_time, rospy.Time.now().to_sec()))