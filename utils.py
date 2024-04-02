import ros_numpy
import numpy as np
from skimage.transform import rotate as image_rotate
from scipy.spatial.transform import Rotation

def get_xyz_coords_from_msg(msg, fields, rotation):
    points_numpify = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    points_numpify = points_numpify.ravel()
    if fields == 'xyz':
        points_x = np.array([x[0] for x in points_numpify])[:, np.newaxis]
        points_y = np.array([x[1] for x in points_numpify])[:, np.newaxis]
        points_z = np.array([x[2] for x in points_numpify])[:, np.newaxis]
        points_xyz = np.concatenate([points_x, points_y, points_z], axis=1)
    elif fields == 'xyzrgb':
        points_numpify = ros_numpy.point_cloud2.split_rgb_field(points_numpify)
        points_x = np.array([x[0] for x in points_numpify])[:, np.newaxis]
        points_y = np.array([x[1] for x in points_numpify])[:, np.newaxis]
        points_z = np.array([x[2] for x in points_numpify])[:, np.newaxis]
        points_r = np.array([x[3] for x in points_numpify])[:, np.newaxis]
        points_g = np.array([x[4] for x in points_numpify])[:, np.newaxis]
        points_b = np.array([x[5] for x in points_numpify])[:, np.newaxis]
        points_xyz = np.concatenate([points_x, points_y, points_z, points_r, points_g, points_b], axis=1)
    else:
        print('Incorrect pointcloud fields {}. Fields must be `xyz` or `xyzrgb`'.format(fields))
        points_xyz = None
    points_xyz = rotate_pcd(points_xyz, rotation)
    return points_xyz

def rotate_pcd(points, rotation_matrix):
    points_xyz = points[:, :3]
    #rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()
    points_xyz_rotated = points_xyz @ rotation_matrix
    points_rotated = points.copy()
    points_rotated[:, :3] = points_xyz_rotated
    return points_rotated

def transform_pcd(points, x, y, theta):
    points_transformed = points.copy()
    points_transformed[:, 0] = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
    points_transformed[:, 1] = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
    points_transformed[:, 0] += x
    points_transformed[:, 1] += y
    return points_transformed

def get_occupancy_grid(points_xyz):
    index = np.isnan(points_xyz).any(axis=1)
    points_xyz = np.delete(points_xyz, index, axis=0)
    points_xyz = np.clip(points_xyz, -8, 8)
    resolution = 0.1
    radius = 18
    #print('Points xyz:', points_xyz.shape, points_xyz[0], points_xyz.min(), points_xyz.max())
    points_ij = np.round(points_xyz[:, :2] / resolution).astype(int) + [int(radius / resolution), int(radius / resolution)]
    grid = np.zeros((int(2 * radius / resolution), int(2 * radius / resolution)), dtype=np.uint8)
    grid[points_ij[:, 0], points_ij[:, 1]] = 1
    return grid

def normalize(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle

def rotate(x, y, angle):
    x_new = x * np.cos(angle) + y * np.sin(angle)
    y_new = -x * np.sin(angle) + y * np.cos(angle)
    return x_new, y_new

def remove_floor_and_ceil(cloud, floor_height=-0.9, ceil_height=1.5):
    heights = np.linspace(-4.0, 4.0, 41)
    floor_index = None
    if floor_height == 'auto':
        bins = []
        for i, height in enumerate(heights[:-1]):
            bins.append(len(cloud[(cloud[:, 2] > height) * (cloud[:, 2] < heights[i + 1])]))
        #print('Bins:', bins)
        floor_index = np.argmax(bins[:20]) + 1
        floor_height = heights[floor_index]
        assert floor_index < len(heights) - 5
    if ceil_height == 'auto':
        if floor_index is None:
            floor_index = 0
            while floor_index < len(heights) - 6 and heights[floor_index] < floor_height:
                floor_index += 1
        ceil_index = floor_index + 5 + np.argmax(bins[floor_index + 5:])
        ceil_height = heights[ceil_index]
    #print('Floor height:', floor_height)
    #print('Ceil height:', ceil_height)
    return cloud[(cloud[:, 2] > floor_height) * (cloud[:, 2] < ceil_height)]

def raycast(grid, n_rays=1000):
    grid_raycasted = grid.copy()
    for sector in range(n_rays):
        angle = sector / n_rays * 2 * np.pi - np.pi
        ii = grid.shape[0] // 2 + np.sin(angle) * np.arange(0, grid.shape[0] // 2)
        jj = grid.shape[0] // 2 + np.cos(angle) * np.arange(0, grid.shape[0] // 2)
        ii = ii.astype(int)
        jj = jj.astype(int)
        points_on_ray = grid[ii, jj]
        if len(points_on_ray.nonzero()[0]) > 0:
            first_obst = points_on_ray.nonzero()[0][0]
            grid_raycasted[ii[:first_obst], jj[:first_obst]] = 1
        else:
            grid_raycasted[ii, jj] = 1
    return grid_raycasted

def get_rel_pose(x, y, theta, x2, y2, theta2):
    rel_x, rel_y = rotate(x2 - x, y2 - y, theta)
    return [rel_x, rel_y, normalize(theta2 - theta)]

def get_iou(rel_x, rel_y, rel_theta, cur_cloud, v_cloud, save=False):
    rel_x_rotated = -rel_x * np.cos(rel_theta) - rel_y * np.sin(rel_theta)
    rel_y_rotated = rel_x * np.sin(rel_theta) - rel_y * np.cos(rel_theta)
    rel_x, rel_y = rel_x_rotated, rel_y_rotated
    if np.sqrt(rel_x ** 2 + rel_y ** 2) > 5:
        return 0
    cur_cloud_transformed = transform_pcd(cur_cloud, rel_x, rel_y, rel_theta)
    cur_grid_transformed = get_occupancy_grid(cur_cloud_transformed)
    cur_grid_transformed = raycast(cur_grid_transformed)
    v_grid = get_occupancy_grid(v_cloud)
    v_grid[163:198, 163:198] = 1
    v_grid = raycast(v_grid)
    intersection = np.sum(v_grid * cur_grid_transformed)
    union = np.sum(v_grid | cur_grid_transformed)
    grid_aligned = np.zeros((v_grid.shape[0], v_grid.shape[1], 3))
    grid_aligned[:, :, 0] = cur_grid_transformed
    grid_aligned[:, :, 1] = v_grid
    grid_aligned = (grid_aligned * 255).astype(np.uint8)
    return intersection / union

def apply_pose_shift(pose, rel_x, rel_y, rel_theta):
    x, y, theta = pose
    new_x = x + rel_x * np.cos(-theta) + rel_y * np.sin(-theta)
    new_y = y - rel_x * np.sin(-theta) + rel_y * np.cos(-theta)
    new_theta = theta + rel_theta
    return [new_x, new_y, new_theta]

def rotate_vertical(cloud, angle):
    cloud_rotated = cloud.copy()
    cloud_rotated[:, 0] = cloud[:, 0] * np.cos(angle) + cloud[:, 2] * np.sin(angle)
    cloud_rotated[:, 2] = -cloud[:, 0] * np.sin(angle) + cloud[:, 2] * np.cos(angle)
    return cloud_rotated