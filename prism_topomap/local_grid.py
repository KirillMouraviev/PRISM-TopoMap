import numpy as np
np.float = np.float64
from cv2 import warpAffine
from prism_topomap.utils import *
import yaml
import cv2
import os
from skimage.io import imread, imsave

class LocalGrid:
    def __init__(self, 
                 resolution=0.1, 
                 radius=18.0, 
                 max_range=8.0, 
                 floor_height=0.0, ceil_height=1.0,
                 obstacles_attenuation=0.9,
                 curbs_attenuation=0.99,
                 layer_names=['curbs', 'occupancy', 'height_map', 'density_map', 'density_map_cur'],
                 save_dir=None):
        self.resolution = resolution
        self.radius = radius
        self.max_range = max_range
        self.floor_height = floor_height
        self.ceil_height = ceil_height
        self.obstacles_attenuation = obstacles_attenuation
        self.curbs_attenuation = curbs_attenuation
        self.layer_names = layer_names
        self.grid_size = 2 * int(radius / resolution)
        self.layers = {}
        for layer_name in self.layer_names:
            if layer_name == 'height_map':
                dtype = np.float32
            else:
                dtype = np.uint8
            self.layers[layer_name] = np.zeros((self.grid_size, self.grid_size), dtype=dtype)
        self.save_dir = save_dir
        if self.save_dir is not None and not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def copy(self):
        grid_copy = LocalGrid(resolution=self.resolution, 
                              radius=self.radius, max_range=self.max_range,
                              floor_height=self.floor_height, ceil_height=self.ceil_height,
                              layer_names=self.layer_names,
                              obstacles_attenuation=self.obstacles_attenuation, curbs_attenuation=self.curbs_attenuation,
                              save_dir=self.save_dir)
        grid_copy.layers = {}
        for layer_name in self.layer_names:
            grid_copy.layers[layer_name] = self.layers[layer_name].copy()
        return grid_copy

    def raycast_grid(self, n_rays=1000, center_point=None):
        grid_raycasted = self.layers['occupancy'].copy()
        if center_point is None:
            center_point = (self.grid_size // 2, self.grid_size // 2)
        for sector in range(n_rays):
            angle = sector / n_rays * 2 * np.pi - np.pi
            ii = center_point[0] + np.sin(angle) * np.arange(0, self.grid_size // 2)
            jj = center_point[1] + np.cos(angle) * np.arange(0, self.grid_size // 2)
            ii = ii.astype(int)
            jj = jj.astype(int)
            good_ids = ((ii > 0) * (ii < self.grid_size) ** (jj > 0) * (jj < self.grid_size)).astype(bool)
            ii = ii[good_ids]
            jj = jj[good_ids]
            points_on_ray = self.layers['occupancy'][ii, jj]
            if len(points_on_ray.nonzero()[0]) > 0:
                last_obst = points_on_ray.nonzero()[0][-1]
                grid_raycasted[ii[:last_obst], jj[:last_obst]] = 1
            else:
                grid_raycasted[ii, jj] = 1
        self.layers['occupancy'] = grid_raycasted

    def update_from_cloud_and_transform(self, points_xyz, 
                                        x=0, y=0, theta=0):
        self.transform(x, y, theta)
        index = np.isnan(points_xyz).any(axis=1)
        points_xyz = np.delete(points_xyz, index, axis=0)
        points_xyz = points_xyz[(points_xyz[:, 0] > -self.max_range) * (points_xyz[:, 0] < self.max_range) * \
                                (points_xyz[:, 1] > -self.max_range) * (points_xyz[:, 1] < self.max_range)]
        points_xyz_obstacles = remove_floor_and_ceil(points_xyz, floor_height=self.floor_height, ceil_height=self.ceil_height)
        grid_radius = int(self.radius / self.resolution)
        #print('Points xyz:', points_xyz.shape, points_xyz[0], points_xyz.min(), points_xyz.max())
        points_ij_all = np.round(points_xyz[:, :2] / self.resolution).astype(int) + \
                            [grid_radius, grid_radius]
        mask = (points_ij_all[:, 0] >= 0) * (points_ij_all[:, 0] < self.grid_size) * \
               (points_ij_all[:, 1] >= 0) * (points_ij_all[:, 1] < self.grid_size)
        points_ij_all = points_ij_all[mask]
        points_ij_obstacles = np.round(points_xyz_obstacles[:, :2] / self.resolution).astype(int) + \
                            [grid_radius, grid_radius]
        points_ij_obstacles = points_ij_obstacles[(points_ij_obstacles[:, 0] >= 0) * (points_ij_obstacles[:, 0] < self.grid_size) * \
                              (points_ij_obstacles[:, 1] >= 0) * (points_ij_obstacles[:, 1] < self.grid_size)]
        # Fill occupancy layer
        self.layers['occupancy'] = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.layers['occupancy'][points_ij_all[:, 0], points_ij_all[:, 1]] = 1
        self.raycast_grid()
        self.layers['occupancy'][points_ij_obstacles[:, 0], points_ij_obstacles[:, 1]] = 2
        # Fill density map
        if 'density_map' in self.layer_names:
            self.layers['density_map_cur'], _, _ = np.histogram2d(points_ij_obstacles[:, 0], points_ij_obstacles[:, 1],
                                               bins=self.grid_size, range=[[0, self.grid_size], [0, self.grid_size]])
            self.layers['density_map_cur'] = self.layers['density_map_cur'].astype(np.float32)
            self.layers['density_map'] = self.layers['density_map'] * self.obstacles_attenuation + self.layers['density_map_cur']
        # Fill height map
        if 'height_map' in self.layer_names:
            self.layers['height_map'] = np.zeros((self.grid_size, self.grid_size))
            np.maximum.at(self.layers['height_map'], (points_ij_all[:, 0], points_ij_all[:, 1]), points_xyz[mask][:, 2])

    def update_curbs_from_cloud(self, points_xyz):
        index = np.isnan(points_xyz).any(axis=1)
        points_xyz = np.delete(points_xyz, index, axis=0)
        points_xyz = points_xyz[(points_xyz[:, 0] > -self.max_range) * (points_xyz[:, 0] < self.max_range) * \
                                (points_xyz[:, 1] > -self.max_range) * (points_xyz[:, 1] < self.max_range)]
        points_xyz_obstacles = points_xyz
        points_ij = np.round(points_xyz_obstacles[:, :2] / self.resolution).astype(int) + \
                            [int(self.radius / self.resolution), int(self.radius / self.resolution)]
        points_ij = points_ij[(points_ij[:, 0] >= 0) * (points_ij[:, 0] < self.grid_size) * \
                              (points_ij[:, 1] >= 0) * (points_ij[:, 1] < self.grid_size)]
        self.layers['curbs_cur'] = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.layers['curbs_cur'][points_ij[:, 0], points_ij[:, 1]] = 1
        self.layers['curbs'] = self.layers['curbs'] * self.curbs_attenuation + self.layers['curbs_cur']

    def get_transformed_grid(self, grid, x, y, theta):
        minus8 = np.array([
            [1, 0, self.radius / self.resolution],
            [0, 1, self.radius / self.resolution],
            [0, 0, 1]
        ])
        tf_matrix = np.array([
            [np.cos(-theta), np.sin(-theta), y / self.resolution],
            [-np.sin(-theta), np.cos(-theta), x / self.resolution],
            [0, 0, 1]
        ])
        plus8 = np.array([
            [1, 0, -self.radius / self.resolution],
            [0, 1, -self.radius / self.resolution],
            [0, 0, 1]
        ])
        tf_matrix_shifted = minus8 @ tf_matrix @ plus8
        return warpAffine(grid, tf_matrix_shifted[:2], grid.shape)

    def transform(self, x, y, theta):
        for layer_name in self.layer_names:
            self.layers[layer_name] = self.get_transformed_grid(self.layers[layer_name], x, y, theta)

    def is_inside(self, x, y, theta):
        i = int((x + self.radius) / self.resolution)
        j = int((y + self.radius) / self.resolution)
        if i < 0 or i >= self.grid_size or j < 0 or j >= self.grid_size:
            return False
        return (self.layers['occupancy'][i, j] > 0)

    def get_iou(self, other, rel_x, rel_y, rel_theta, save=False, cnt=0):
        rel_x_rotated = -rel_x * np.cos(rel_theta) - rel_y * np.sin(rel_theta)
        rel_y_rotated = rel_x * np.sin(rel_theta) - rel_y * np.cos(rel_theta)
        rel_x, rel_y = rel_x_rotated, rel_y_rotated
        # if np.sqrt(rel_x ** 2 + rel_y ** 2) > 5:
        #     return 0
        cur_grid_transformed = self.get_transformed_grid(self.layers['occupancy'], rel_x, rel_y, rel_theta)
        cur_grid_transformed[cur_grid_transformed > 0] = 1
        v_grid_copy = other.layers['occupancy'].copy()
        v_grid_copy[v_grid_copy > 0] = 1
        intersection = np.sum(v_grid_copy * cur_grid_transformed)
        union = np.sum(v_grid_copy | cur_grid_transformed)
        grid_aligned = np.zeros((v_grid_copy.shape[0], v_grid_copy.shape[1], 3))
        grid_aligned[:, :, 0] = cur_grid_transformed
        grid_aligned[:, :, 1] = v_grid_copy
        grid_aligned = (grid_aligned * 255).astype(np.uint8)
        if save and self.save_dir is not None:
            # print(cnt)
            save_dir = os.path.join(self.save_dir, str(cnt))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savez(os.path.join(save_dir, 'cur_grid.npz'), self.layers['occupancy'])
            np.savez(os.path.join(save_dir, 'cur_grid_transformed.npz'), cur_grid_transformed)
            np.savez(os.path.join(save_dir, 'v_grid.npz'), v_grid_copy)
            np.savetxt(os.path.join(save_dir, 'rel_pose.txt'), np.array([rel_x, rel_y, rel_theta]))
            imsave(os.path.join(save_dir, 'grid_aligned.png'), grid_aligned)
        return intersection / union

    def get_tf_matrix_xy(self, trans_i, trans_j, rot_angle):
        #print('Trans i trans j rot angle:', trans_i, trans_j, rot_angle)
        plus8 = np.eye(4)
        plus8[0, 3] = self.radius
        plus8[1, 3] = self.radius
        minus8 = np.eye(4)
        minus8[0, 3] = -self.radius
        minus8[1, 3] = -self.radius
        tf_matrix = np.array([
            [np.cos(rot_angle), np.sin(rot_angle), 0, trans_i * self.resolution],
            [-np.sin(rot_angle), np.cos(rot_angle), 0, trans_j * self.resolution],
            [0,                  0,                 1, 0],
            [0,                  0,                 0, 1]
        ])
        tf_matrix = minus8 @ tf_matrix @ plus8
        #print('Translation from tf matrix:', tf_matrix[:, 3])
        return tf_matrix

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for layer_name in self.layers:
            print(layer_name, self.layers[layer_name].sum())
            if layer_name == 'height_map':
                self.layers[layer_name] = self.layers[layer_name].astype(np.float32)
            else:
                self.layers[layer_name] = self.layers[layer_name].astype(np.uint8)
            if layer_name == 'height_map':
                # self.layers[layer_name] = np.clip(self.layers[layer_name], self.floor_height, self.ceil_height)
                np.savez(os.path.join(save_dir, '{}.npz'.format(layer_name)), self.layers[layer_name])
            else:
                # print(layer_name, self.layers[layer_name].shape, self.layers[layer_name].dtype)
                imsave(os.path.join(save_dir, '{}.png'.format(layer_name)), self.layers[layer_name])
        metadata = {
            'layer_names': self.layer_names,
            'resolution': self.resolution,
            'radius': self.radius,
            'max_range': self.max_range,
            'floor_height': self.floor_height,
            'ceil_height': self.ceil_height,
            'obstacles_attenuation': self.obstacles_attenuation,
            'curbs_attenuation': self.curbs_attenuation
        }
        fout = open(os.path.join(save_dir, 'metadata.yaml'), 'w')
        yaml.dump(metadata, fout, default_flow_style=False)
        fout.close()

def load_local_grid(save_dir):
    fin = open(os.path.join(save_dir, 'metadata.yaml'), 'r')
    metadata = yaml.safe_load(fin)
    fin.close()
    grid = LocalGrid(
        layer_names=metadata['layer_names'],
        resolution = metadata['resolution'],
        radius = metadata['radius'],
        max_range = metadata['max_range'],
        floor_height = metadata['floor_height'],
        ceil_height = metadata['ceil_height'],
        obstacles_attenuation = metadata['obstacles_attenuation'],
        curbs_attenuation = metadata['curbs_attenuation']
    )
    for layer_name in metadata['layer_names']:
        if layer_name == 'height_map':
            grid.layers[layer_name] = np.load(os.path.join(save_dir, '{}.npz'.format(layer_name)))['arr_0']
        else:
            grid.layers[layer_name] = imread(os.path.join(save_dir, '{}.png'.format(layer_name)))
        # print(layer_name, grid.layers[layer_name].shape, grid.layers[layer_name].sum())
    return grid