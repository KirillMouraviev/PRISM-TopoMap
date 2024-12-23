import json
import rospy
import heapq
import numpy as np
import time
import os
import copy
import ros_numpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from collections import deque
from typing import Dict
from torch import Tensor

import faiss
import torch
from scipy.spatial.transform import Rotation
from skimage.io import imread, imsave
from opr.pipelines.registration import PointcloudRegistrationPipeline, RansacGlobalRegistrationPipeline, Feature2DGlobalRegistrationPipeline
from opr.datasets.augmentations import DefaultHM3DImageTransform
import MinkowskiEngine as ME
import open3d as o3d
import open3d.pipelines.registration as registration
from toposlam_msgs.msg import Edge
from toposlam_msgs.msg import TopologicalGraph as TopologicalGraphMessage
from local_grid import LocalGrid
from utils import get_rel_pose

from memory_profiler import profile

class TopologicalGraph():
    def __init__(self,
                 place_recognition_model,
                 place_recognition_index,
                 registration_model,
                 inline_registration_model,
                 map_frame='map',
                 registration_score_threshold=0.5,
                 inline_registration_score_threshold=0.5,
                 grid_resolution=0.1,
                 grid_radius=18.0,
                 max_grid_range=8.0,
                 floor_height=-1.0,
                 ceil_height=2.0):
        self.vertices = []
        self.adj_lists = []
        self.map_frame = map_frame
        self.graph_viz_pub = rospy.Publisher('topological_map', MarkerArray, latch=True, queue_size=100)
        self.graph_pub = rospy.Publisher('graph', TopologicalGraphMessage, latch=True, queue_size=100)
        self.place_recognition_model = place_recognition_model
        self.index = place_recognition_index
        self.registration_pipeline = registration_model
        self.registration_score_threshold = registration_score_threshold
        self.inline_registration_pipeline = inline_registration_model
        self.inline_registration_score_threshold = inline_registration_score_threshold
        self.grid_resolution = grid_resolution
        self.grid_radius = grid_radius
        self.max_grid_range = max_grid_range
        self.ref_cloud_pub = rospy.Publisher('/ref_cloud', PointCloud2, latch=True, queue_size=100)
        self._pointcloud_quantization_size = 0.2
        self.floor_height = floor_height
        self.ceil_height = ceil_height
        self.device = torch.device('cuda:0')
        self.image_transform = DefaultHM3DImageTransform(train=False)
        self.graph_save_path = '/home/kirill/TopoSLAM/toposlam_ws/src/simple_toposlam_model/test_husky_rosbag_minkloc3d_5/graph_data'
        if not os.path.exists(self.graph_save_path):
            os.mkdir(self.graph_save_path)
        self.pr_results_save_path = '/home/kirill/TopoSLAM/toposlam_ws/src/simple_toposlam_model/test_husky_rosbag_minkloc3d_5/place_recognition_data'
        if not os.path.exists(self.pr_results_save_path):
            os.mkdir(self.pr_results_save_path)
        self.global_pose_for_visualization = None

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
                    quantization_size=self._pointcloud_quantization_size,
                )
                out_dict["pointclouds_lidar_coords"] = ME.utils.batched_coordinates([quantized_coords]).to(
                    self.device
                )
                out_dict["pointclouds_lidar_feats"] = quantized_feats.to(self.device)
        return out_dict

    def normalize(self, angle):
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def load_from_json(self, input_path):
        fin = open(os.path.join(input_path, 'graph.json'), 'r')
        j = json.load(fin)
        fin.close()
        self.vertices = j['vertices']
        self.adj_lists = j['edges']
        for i in range(len(self.vertices)):
            grid = np.load(os.path.join(input_path, '{}_grid.npz'.format(i)))['arr_0']
            self.vertices[i]['grid'] = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, \
                                                 max_range=self.max_grid_range, grid=grid)
            #img_front = imread(os.path.join(input_path, '{}_img_front.png'.format(i)))
            #self.vertices[i]['img_front'] = img_front
            #img_back = imread(os.path.join(input_path, '{}_img_back.png'.format(i)))
            #self.vertices[i]['img_back'] = img_back
            self.index.add(np.array(self.vertices[i]['descriptor'])[np.newaxis, :])

    #@profile
    def add_vertex(self, global_pose_for_visualization, img_front, img_back, cloud=None):
        x, y, theta = global_pose_for_visualization
        print('\n\n\n Add new vertex ({}, {}, {}) with idx {} \n\n\n'.format(x, y, theta, len(self.vertices)))
        self.adj_lists.append([])
        if cloud is not None:
            img_front_transformed = self.image_transform(img_front)
            #print(img_front_transformed.shape, img_front_transformed.min(), img_front_transformed.mean(), img_front_transformed.max())
            img_back_transformed = self.image_transform(img_back)
            #print(img_back_transformed.shape, img_back_transformed.min(), img_back_transformed.mean(), img_back_transformed.max())
            #print('Transformed image shape:', img_front_transformed.shape)
            img_front_tensor = torch.Tensor(img_front).cuda()
            img_back_tensor = torch.Tensor(img_back).cuda()
            #print('Img front min and max:', img_front_transformed.min(), img_front_transformed.max())
            img_front_tensor = torch.permute(img_front_tensor, (2, 0, 1))
            img_back_tensor = torch.permute(img_back_tensor, (2, 0, 1))
            input_data = {
                     'pointcloud_lidar_coords': torch.Tensor(cloud[:, :3]).cuda(),
                     'pointcloud_lidar_feats': torch.ones((cloud.shape[0], 1)).cuda(),
                     'image_front': img_front_tensor,
                     'image_back': img_back_tensor
                     }
            batch = self._preprocess_input(input_data)
            descriptor = self.place_recognition_model(batch)["final_descriptor"].detach().cpu().numpy()
            #descriptor = np.random.random(256)
            if len(descriptor.shape) == 1:
                descriptor = descriptor[np.newaxis, :]
            grid = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
            grid.load_from_cloud(cloud[:, :3])
            #print('X y theta:', x, y, theta)
            vertex_dict = {
                'stamp': rospy.Time.now(),
                'pose_for_visualization': [x, y, theta],
                'img_front': img_front,
                'img_back': img_back,
                'grid': grid,
                'descriptor': descriptor
            }
            self.vertices.append(vertex_dict)
            #print('Descriptor shape:', descriptor.shape)
            self.index.add(descriptor)
        return len(self.vertices) - 1

    def save_vertex(self, vertex_id):
        save_dir = os.path.join(self.graph_save_path, str(vertex_id))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        vertex_dict = self.vertices[vertex_id]
        pose_stamped = np.array([vertex_dict['stamp'].to_sec()] + vertex_dict['pose_for_visualization'])
        #print('Pose stamped:', pose_stamped)
        np.savetxt(os.path.join(save_dir, 'pose_stamped.txt'), pose_stamped)
        #imsave(os.path.join(save_dir, 'img_front.png'), vertex_dict['img_front'])
        #imsave(os.path.join(save_dir, 'img_back.png'), vertex_dict['img_back'])
        np.savez(os.path.join(save_dir, 'grid.npz'), vertex_dict['grid'].grid)
        np.savetxt(os.path.join(save_dir, 'descriptor.txt'), vertex_dict['descriptor'])
        edges = []
        for v, rel_pose in self.adj_lists[vertex_id]:
            edges.append([v] + rel_pose)
        edges = np.array(edges)
        np.savetxt(os.path.join(save_dir, 'edges.txt'), edges)

    def save_localization_results(self, state_dict, vertex_ids, transforms, pr_scores, reg_scores):
        save_dir = os.path.join(self.pr_results_save_path, str(state_dict['stamp']))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        imsave(os.path.join(save_dir, 'img_front.png'), state_dict['img_front'])
        imsave(os.path.join(save_dir, 'img_back.png'), state_dict['img_back'])
        np.savez(os.path.join(save_dir, 'grid.npz'), state_dict['grid'].grid)
        np.savetxt(os.path.join(save_dir, 'descriptor.txt'), state_dict['descriptor'])
        np.savetxt(os.path.join(save_dir, 'vertex_ids.txt'), vertex_ids)
        gt_pose_data = [state_dict['pose_for_visualization']]
        tf_data = []
        for idx, tf in zip(vertex_ids, transforms):
            if idx >= 0:
                vertex_dict = self.vertices[idx]
                if tf is not None:
                    tf_data.append([idx] + list(tf))
                else:
                    tf_data.append([idx, 0, 0, 0, 0, 0, 0])
                gt_pose_data.append(vertex_dict['pose_for_visualization'])
        np.savetxt(os.path.join(save_dir, 'gt_poses.txt'), np.array(gt_pose_data))
        np.savetxt(os.path.join(save_dir, 'transforms.txt'), np.array(tf_data))
        np.savetxt(os.path.join(save_dir, 'pr_scores.txt'), np.array(pr_scores))
        np.savetxt(os.path.join(save_dir, 'reg_scores.txt'), np.array(reg_scores))

    #@profile
    def get_k_most_similar(self, img_front, img_back, cloud, stamp, k=1):
        t1 = time.time()
        input_data = {'pointcloud_lidar_coords': torch.Tensor(cloud[:, :3]).cuda(),
                    'pointcloud_lidar_feats': torch.ones((cloud.shape[0], 1)).cuda()}
        if img_front is not None:
            img_front_tensor = torch.Tensor(img_front).cuda()
            img_front_tensor = torch.permute(img_front_tensor, (2, 0, 1))
            input_data['image_front'] = img_front_tensor
        if img_back is not None:
            img_back_tensor = torch.Tensor(img_back).cuda()
            input_data['image_back'] = img_back_tensor
            img_back_tensor = torch.permute(img_back_tensor, (2, 0, 1))
        if cloud is not None:
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
            #cloud_with_fields = ros_numpy.point_cloud2.merge_rgb_fields(cloud_with_fields)
            cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_with_fields)
            if stamp is not None:
                cloud_msg.header.stamp = stamp
            else:
                cloud_msg.header.stamp = rospy.Time.now()
            cloud_msg.header.frame_id = 'base_link'
            self.ref_cloud_pub.publish(cloud_msg)
        t2 = time.time()
        #print('Ref cloud publish time:', t2 - t1)
        batch = self._preprocess_input(input_data)
        grid = LocalGrid(resolution=self.grid_resolution, radius=self.grid_radius, max_range=self.max_grid_range)
        grid.load_from_cloud(cloud)
        t3 = time.time()
        #print('Preprocessing time:', t3 - t2)
        descriptor = self.place_recognition_model(batch)["final_descriptor"].detach().cpu().numpy()
        #descriptor = np.random.random(256)
        if len(descriptor.shape) == 1:
            descriptor = descriptor[np.newaxis, :]
        reg_scores = []
        dists, pred_i = self.index.search(descriptor, k)
        t4 = time.time()
        #print('Place recognition time:', t4 - t3)
        pr_scores = dists[0]
        pred_i = pred_i[0]
        # print('Pred i:', pred_i)
        pred_tf = []
        pred_i_filtered = []
        for idx in pred_i:
            #print('Stamp {}, vertex id {}'.format(stamp, idx))
            if idx < 0:
                continue
            t1 = time.time()
            cand_vertex_dict = self.vertices[idx]
            cand_grid = cand_vertex_dict['grid']
            cand_grid_tensor = torch.Tensor(cand_grid.grid).to(self.device)
            ref_grid_tensor = torch.Tensor(grid.grid).to(self.device)
            start_time = time.time()
            save_dir = os.path.join(self.pr_results_save_path, str(stamp))
            transform, score = self.registration_pipeline.infer(ref_grid_tensor, cand_grid_tensor, save_dir=save_dir)
            t2 = time.time()
            #print('Registration time:', t2 - t1)
            #t3 = time.time()
            #print('ICP time:', t3 - t2)
            #if score_icp < 0.8:
            reg_scores.append(score)
            #print('Registration score of vertex {} is {}'.format(idx, score))
            if score < self.registration_score_threshold:
                pred_i_filtered.append(-1)
                pred_tf.append([0, 0, 0, 0, 0, 0])
            else:
                tf_matrix = cand_grid.get_tf_matrix_xy(*transform)
                pred_i_filtered.append(idx)
                tf_rotation = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
                tf_translation = tf_matrix[:3, 3]
                pred_tf.append(list(tf_rotation) + list(tf_translation))
                #print('Tf rotation:', tf_rotation)
                #print('Tf translation:', tf_translation)
        #print('Pred tf:', np.array(pred_tf))
        #print('Pred i filtered:', pred_i_filtered)
        # state_dict = {
        #     'stamp': stamp,
        #     'pose_for_visualization': self.global_pose_for_visualization,
        #     'img_front': img_front,
        #     'img_back': img_back,
        #     'grid': grid,
        #     'descriptor': descriptor
        # }
        # self.save_localization_results(state_dict, pred_i, pred_tf, pr_scores, reg_scores)
        return pred_i, pred_i_filtered, np.array(pred_tf), pr_scores, reg_scores

    def get_transform_to_vertex(self, vertex_id, grid):
        #return get_rel_pose(*self.global_pose_for_visualization, *self.vertices[vertex_id]['pose_for_visualization'])
        cand_grid = self.vertices[vertex_id]['grid']
        cand_grid_tensor = torch.Tensor(cand_grid.grid).to(self.device)
        ref_grid_tensor = torch.Tensor(grid.grid).to(self.device)
        #print('                Ref grid:', grid.max())
        #print('                Cand grid:', cand_grid.max())
        transform, score = self.inline_registration_pipeline.infer(ref_grid_tensor, cand_grid_tensor)
        if score > self.inline_registration_score_threshold:
            tf_matrix = cand_grid.get_tf_matrix_xy(*transform)
            x = tf_matrix[0, 3]
            y = tf_matrix[1, 3]
            _, __, theta = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
            return x, y, theta
        return None, None, None

    def inverse_transform(self, x, y, theta):
        x_inv = -x * np.cos(theta) - y * np.sin(theta)
        y_inv = x * np.sin(theta) - y * np.cos(theta)
        theta_inv = -theta
        return [x_inv, y_inv, theta_inv]
    
    def add_edge(self, i, j, x, y, theta):
        if i == j:
            return
        if j in [x[0] for x in self.adj_lists[i]]:
            return
        xi, yi, _ = self.vertices[i]['pose_for_visualization']
        xj, yj, _ = self.vertices[j]['pose_for_visualization']
        print('\nAdd edge from ({}, {}) to ({}, {}) with rel pose ({}, {}, {})\n'.format(xi, yi, xj, yj, x, y, theta))
        self.adj_lists[i].append((j, [x, y, theta]))
        self.adj_lists[j].append((i, self.inverse_transform(x, y, theta)))

    def get_vertex(self, vertex_id):
        return self.vertices[vertex_id]

    def has_edge(self, u, v):
        for x, rel_pose in self.adj_lists[u]:
            if x == v:
                return True
        return False

    def get_edge(self, u, v):
        for x, rel_pose in self.adj_lists[u]:
            if x == v:
                return rel_pose
        return None

    def get_edges_from(self, u):
        return self.adj_lists[u]

    def get_path_with_length(self, u, v):
        # Initialize distances and previous nodes dictionaries
        start_time = time.time()
        distances = [float('inf')] * len(self.adj_lists)
        prev_nodes = [None] * len(self.adj_lists)
        # Set distance to start node as 0
        distances[u] = 0
        # Create priority queue with initial element (distance to start node, start node)
        heap = [(0, u)]
        # Run Dijkstra's algorithm
        while heap:
            # Pop node with lowest distance from heap
            current_distance, current_node = heapq.heappop(heap)
            if current_node == v:
                path = [current_node]
                cur = current_node
                while cur != u:
                    cur = prev_nodes[cur]
                    path.append(cur)
                path = path[::-1]
                #print('Path planning time:', time.time() - start_time)
                return path, distances[v]
            # If current node has already been visited, skip it
            if current_distance > distances[current_node]:
                continue
            # For each neighbour of current node
            for neighbour, pose in self.adj_lists[current_node]:
                weight = np.sqrt(pose[0] ** 2 + pose[1] ** 2)
                # Calculate tentative distance to neighbour through current node
                tentative_distance = current_distance + weight
                # Update distance and previous node if tentative distance is better than current distance
                if tentative_distance < distances[neighbour]:
                    distances[neighbour] = tentative_distance
                    prev_nodes[neighbour] = current_node
                    # Add neighbour to heap with updated distance
                    heapq.heappush(heap, (tentative_distance, neighbour))
        #print('Path planning time:', time.time() - start_time)
        return None, float('inf')
        
    def publish_graph(self):
        # Publish graph for visualization
        graph_msg = MarkerArray()
        vertices_marker = Marker()
        #vertices_marker = ns = 'points_and_lines'
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.5
        vertices_marker.scale.y = 0.5
        vertices_marker.scale.z = 0.5
        vertices_marker.color.r = 1
        vertices_marker.color.g = 0
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        for vertex_dict in self.vertices:
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
        for u in range(len(self.vertices)):
            for v, pose in self.adj_lists[u]:
                ux, uy, _ = self.vertices[u]['pose_for_visualization']
                vx, vy, _ = self.vertices[v]['pose_for_visualization']
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
        for vertex_dict in self.vertices:
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
        for i, vertex_dict in enumerate(self.vertices):
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
        for u in range(len(self.vertices)):
            for v, pose in self.adj_lists[u]:
                if u >= v:
                    continue
                ux, uy, _ = self.vertices[u]['pose_for_visualization']
                vx, vy, _ = self.vertices[v]['pose_for_visualization']
                text_marker.id = cnt
                text_marker.pose.position.x = (ux + vx) / 2
                text_marker.pose.position.y = (uy + vy) / 2
                text_marker.text = '({}, {}, {})'.format(round(pose[0], 1), round(pose[1], 1), round(pose[2], 2))
                graph_msg.markers.append(copy.deepcopy(text_marker))
                cnt += 1

        self.graph_viz_pub.publish(graph_msg)

    def save_to_json(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.vertices = list(self.vertices)
        for i in range(len(self.vertices)):
            vertex_dict = self.vertices[i]
            np.savez(os.path.join(output_path, '{}_grid.npz'.format(i)), vertex_dict['grid'].grid)
            #imsave(os.path.join(output_path, '{}_img_front.png'.format(i)), vertex_dict['img_front'])
            #imsave(os.path.join(output_path, '{}_img_back.png'.format(i)), vertex_dict['img_back'])
            x, y, theta = vertex_dict['pose_for_visualization']
            descriptor = vertex_dict['descriptor']
            self.vertices[i] = {'pose_for_visualization': (x, y, theta), 'descriptor': [float(x) for x in list(descriptor[0])]}
        j = {'vertices': self.vertices, 'edges': self.adj_lists}
        fout = open(os.path.join(output_path, 'graph.json'), 'w')
        json.dump(j, fout)
        fout.close()