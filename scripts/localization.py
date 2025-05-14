import os
import numpy as np
import time
from utils import apply_pose_shift

tests_dir = '/home/kirill/TopoSLAM/OpenPlaceRecognition/test_registration'

class Localizer():
    def __init__(self, graph, map_frame='map', top_k=5):
        self.graph = graph
        self.top_k = top_k
        self.img_front = None
        self.img_back = None
        self.cloud = None
        self.grid = None
        self.global_pose_for_visualization = None
        self.stamp = None
        self.localized_x = None
        self.localized_y = None
        self.localized_theta = None
        self.localized_stamp = None
        self.localized_img_front = None
        self.localized_img_back = None
        self.localized_cloud = None
        self.rel_poses = None
        self.dists = None
        self.cnt = 0
        self.n_loc_fails = 0
        if not os.path.exists(tests_dir):
            os.mkdir(tests_dir)
        self.map_frame = map_frame

    def save_reg_test_data(self, vertex_ids, transforms, pr_scores, reg_scores, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savez(os.path.join(save_dir, 'ref_cloud.npz'), self.cloud)
        np.savez(os.path.join(save_dir, 'ref_grid.npz'), self.grid.grid)
        #print('Mean of the ref cloud:', self.localized_cloud[:, :3].mean())
        tf_data = []
        if self.global_pose_for_visualization is not None:
            gt_pose_data = [list(self.global_pose_for_visualization)]
        else:
            print('No global pose for visualization to save reg test data!')
            return
        for idx, tf in zip(vertex_ids, transforms):
            if idx >= 0:
                vertex_dict = self.graph.vertices[idx]
                x, y, theta = vertex_dict['pose_for_visualization']
                grid = vertex_dict['grid'].grid
                #print('Grid max:', grid.max())
                #print('GT x, y, theta:', x, y, theta)
                np.savez(os.path.join(save_dir, 'cand_grid_{}.npz'.format(idx)), grid)
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

    def localize(self, event=None):
        # if self.global_pose_for_visualization is None:
        #     print('No global pose provided!')
        #     return
        if self.stamp is None:
            print('Waiting for message to initialize localizer...')
            return None, None, None
        print('Localized from stamp', self.stamp)
        vertex_ids = []
        rel_poses = []
        t1 = time.time()
        start_global_pose = self.global_pose_for_visualization
        start_stamp = self.stamp
        start_img_front = self.img_front
        start_img_back = self.img_back
        start_cloud = self.cloud
        start_grid = self.grid
        #print('Position at start:', self.global_pose_for_visualization)
        self.graph.global_pose_for_visualization = self.global_pose_for_visualization
        if self.cloud is not None:
            vertex_ids_pr_raw, vertex_ids_pr, transforms, pr_scores, reg_scores = self.graph.get_k_most_similar(self.img_front, 
                                                                                                                self.img_back, 
                                                                                                                self.cloud, 
                                                                                                                self.grid,
                                                                                                                self.stamp,
                                                                                                                k=self.top_k)
            t2 = time.time()
            #print('Get k most similar time:', t2 - t1)
            save_dir = os.path.join(tests_dir, 'test_{}'.format(self.cnt))
            self.cnt += 1
            if not os.path.exists(save_dir):
               os.mkdir(save_dir)
            self.save_reg_test_data(vertex_ids_pr_raw, transforms, pr_scores, reg_scores, save_dir)
            t3 = time.time()
            #print('Saving time:', t3 - t2)
            vertex_ids_pr_unmatched = [idx for idx in vertex_ids_pr_raw if idx not in vertex_ids_pr]
            #print('Matched indices:', [idx for idx in vertex_ids_pr if idx >= 0])
            #print('Unmatched indices:', vertex_ids_pr_unmatched)
            rel_poses = [[tf[3], tf[4], tf[2]] for idx, tf in zip(vertex_ids_pr, transforms) if idx >= 0]
            t4 = time.time()
        else:
            vertex_ids_pr = []
        t4 = time.time()
        pr_scores = [pr_scores[i] for i, idx in enumerate(vertex_ids_pr) if idx >= 0]
        reg_scores = [reg_scores[i] for i, idx in enumerate(vertex_ids_pr) if idx >= 0]
        transforms = [transforms[i] for i, idx in enumerate(vertex_ids_pr) if idx >= 0]
        transforms = np.array(transforms)
        vertex_ids_pr = [i for i in vertex_ids_pr if i >= 0]
        vertex_ids_pr_unmatched = [idx for idx in vertex_ids_pr_raw if idx not in vertex_ids_pr]
        if len(vertex_ids_pr) == 0:
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
        t5 = time.time()
        #print('Localization time:', t5 - t1)
        if len(vertex_ids) > 0:
            if start_global_pose is not None:
                self.localized_x, self.localized_y, self.localized_theta = start_global_pose
            self.localized_stamp = start_stamp
        #     self.localized_img_front = start_img_front
        #     self.localized_img_back = start_img_back
            self.localized_cloud = start_cloud
            self.localized_grid = self.grid
        return vertex_ids, rel_poses, vertex_ids_pr_unmatched