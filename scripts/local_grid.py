import numpy as np
np.float = np.float64
from cv2 import warpAffine
from utils import *

class LocalGrid:
    def __init__(self, resolution=0.1, radius=18.0, max_range=8.0, grid=None):
        self.resolution = resolution
        self.radius = radius
        self.max_range = max_range
        grid_size = 2 * int(radius / resolution)
        if grid is None:
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        else:
            self.grid = grid

    def copy(self):
        grid_copy = LocalGrid(self.resolution, self.radius, self.max_range)
        grid_copy.grid = self.grid.copy()
        return grid_copy

    def raycast_grid(self, n_rays=1000, center_point=None):
        grid_raycasted = self.grid.copy()
        if center_point is None:
            center_point = (self.grid.shape[0] // 2, self.grid.shape[1] // 2)
        for sector in range(n_rays):
            angle = sector / n_rays * 2 * np.pi - np.pi
            ii = center_point[0] + np.sin(angle) * np.arange(0, self.grid.shape[0] // 2)
            jj = center_point[1] + np.cos(angle) * np.arange(0, self.grid.shape[0] // 2)
            ii = ii.astype(int)
            jj = jj.astype(int)
            good_ids = ((ii > 0) * (ii < self.grid.shape[0]) ** (jj > 0) * (jj < self.grid.shape[1])).astype(bool)
            ii = ii[good_ids]
            jj = jj[good_ids]
            points_on_ray = self.grid[ii, jj]
            if len(points_on_ray.nonzero()[0]) > 0:
                last_obst = points_on_ray.nonzero()[0][-1]
                grid_raycasted[ii[:last_obst], jj[:last_obst]] = 1
            else:
                grid_raycasted[ii, jj] = 1
        self.grid = grid_raycasted

    def load_from_cloud(self, points_xyz):
        index = np.isnan(points_xyz).any(axis=1)
        points_xyz = np.delete(points_xyz, index, axis=0)
        points_xyz = points_xyz[(points_xyz[:, 0] > -self.max_range) * (points_xyz[:, 0] < self.max_range) * \
                                (points_xyz[:, 1] > -self.max_range) * (points_xyz[:, 1] < self.max_range)]
        points_xyz_obstacles = remove_floor_and_ceil(points_xyz, floor_height=-0.3, ceil_height=0.5)
        #print('Points xyz:', points_xyz.shape, points_xyz[0], points_xyz.min(), points_xyz.max())
        points_ij = np.round(points_xyz[:, :2] / self.resolution).astype(int) + \
                            [int(self.radius / self.resolution), int(self.radius / self.resolution)]
        points_ij = points_ij[(points_ij[:, 0] >= 0) * (points_ij[:, 0] < self.grid.shape[0]) * \
                              (points_ij[:, 1] >= 0) * (points_ij[:, 1] < self.grid.shape[1])]
        self.grid[points_ij[:, 0], points_ij[:, 1]] = 1
        self.raycast_grid()
        points_ij = np.round(points_xyz_obstacles[:, :2] / self.resolution).astype(int) + \
                            [int(self.radius / self.resolution), int(self.radius / self.resolution)]
        points_ij = points_ij[(points_ij[:, 0] >= 0) * (points_ij[:, 0] < self.grid.shape[0]) * \
                              (points_ij[:, 1] >= 0) * (points_ij[:, 1] < self.grid.shape[1])]
        self.grid[points_ij[:, 0], points_ij[:, 1]] = 2

    def load_curb_from_cloud(self, points_xyz):
        index = np.isnan(points_xyz).any(axis=1)
        points_xyz = np.delete(points_xyz, index, axis=0)
        points_xyz = points_xyz[(points_xyz[:, 0] > -self.max_range) * (points_xyz[:, 0] < self.max_range) * \
                                (points_xyz[:, 1] > -self.max_range) * (points_xyz[:, 1] < self.max_range)]
        points_xyz_obstacles = points_xyz
        points_ij = np.round(points_xyz_obstacles[:, :2] / self.resolution).astype(int) + \
                            [int(self.radius / self.resolution), int(self.radius / self.resolution)]
        points_ij = points_ij[(points_ij[:, 0] >= 0) * (points_ij[:, 0] < self.grid.shape[0]) * \
                              (points_ij[:, 1] >= 0) * (points_ij[:, 1] < self.grid.shape[1])]
        self.grid[points_ij[:, 0], points_ij[:, 1]] = 3

    def get_transformed_grid(self, x, y, theta):
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
        return warpAffine(self.grid, tf_matrix_shifted[:2], self.grid.shape)

    def transform(self, x, y, theta):
        self.grid = self.get_transformed_grid(x, y, theta)

    def is_inside(self, x, y, theta):
        i = int((x + self.radius) / self.resolution)
        j = int((y + self.radius) / self.resolution)
        if i < 0 or i >= self.grid.shape[0] or j < 0 or j >= self.grid.shape[1]:
            return False
        return (self.grid[i, j] == 1)

    def get_iou(self, other, rel_x, rel_y, rel_theta, save=False, cnt=0):
        rel_x_rotated = -rel_x * np.cos(rel_theta) - rel_y * np.sin(rel_theta)
        rel_y_rotated = rel_x * np.sin(rel_theta) - rel_y * np.cos(rel_theta)
        rel_x, rel_y = rel_x_rotated, rel_y_rotated
        # if np.sqrt(rel_x ** 2 + rel_y ** 2) > 5:
        #     return 0
        cur_grid_transformed = self.get_transformed_grid(rel_x, rel_y, rel_theta)
        cur_grid_transformed[cur_grid_transformed > 0] = 1
        v_grid_copy = other.grid.copy()
        v_grid_copy[v_grid_copy > 0] = 1
        intersection = np.sum(v_grid_copy * cur_grid_transformed)
        union = np.sum(v_grid_copy | cur_grid_transformed)
        grid_aligned = np.zeros((v_grid_copy.shape[0], v_grid_copy.shape[1], 3))
        grid_aligned[:, :, 0] = cur_grid_transformed
        grid_aligned[:, :, 1] = v_grid_copy
        grid_aligned = (grid_aligned * 255).astype(np.uint8)
        # if save:
        #     # print(cnt)
        #     save_dir = '/home/kirill/test_iou/{}'.format(cnt)
        #     if not os.path.exists(save_dir):
        #         os.mkdir(save_dir)
        #     np.savez(os.path.join(save_dir, 'cur_grid.npz'), self.grid)
        #     np.savez(os.path.join(save_dir, 'cur_grid_transformed.npz'), cur_grid_transformed)
        #     np.savez(os.path.join(save_dir, 'v_grid.npz'), v_grid_copy)
        #     np.savetxt(os.path.join(save_dir, 'rel_pose.txt'), np.array([rel_x, rel_y, rel_theta]))
        #     imsave(os.path.join(save_dir, 'grid_aligned.png'), grid_aligned)
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