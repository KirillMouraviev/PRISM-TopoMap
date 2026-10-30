#!/usr/bin/env python3
"""
SalsaNext inference wrapper for outdoor LiDAR semantic segmentation.
Based on SemanticKITTI pretrained model.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from time import time

# Add SalsaNext to path
sys.path.insert(0, '/home/docker_prism/SalsaNext/train/tasks/semantic')
sys.path.insert(0, '/home/docker_prism/SalsaNext/train')

from modules.SalsaNext import SalsaNext


class SalsaNextInference:
    def __init__(self,
                 model_path='/data/weights/pretrained/SalsaNext',
                 arch_cfg_path='/data/weights/pretrained/arch_cfg.yaml',
                 data_cfg_path='/data/weights/pretrained/data_cfg.yaml',
                 device='cuda:0',
                 verbose=False):
        """
        Initialize SalsaNext model for inference

        Args:
            model_path: Path to pretrained model weights
            arch_cfg_path: Path to architecture config
            data_cfg_path: Path to data config (label mapping)
            device: Device to run inference on
            verbose: Print per-call timing
        """
        self.verbose = verbose
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load configs
        with open(arch_cfg_path, 'r') as f:
            self.arch_cfg = yaml.safe_load(f)
        with open(data_cfg_path, 'r') as f:
            self.label_cfg = yaml.safe_load(f)

        # Get sensor params for range image projection
        self.sensor_cfg = self.arch_cfg['dataset']['sensor']
        self.img_height = self.sensor_cfg['img_prop']['height']  # 64
        self.img_width = self.sensor_cfg['img_prop']['width']     # 2048
        self.fov_up = self.sensor_cfg['fov_up'] / 180.0 * np.pi   # 3 deg
        self.fov_down = self.sensor_cfg['fov_down'] / 180.0 * np.pi  # -25 deg
        self.fov = abs(self.fov_down) + abs(self.fov_up)

        # Get label mapping from data config
        self.learning_map = self.label_cfg.get('learning_map', {})
        self.learning_map_inv = self.label_cfg.get('learning_map_inv', {})
        self.num_classes = len(set(self.learning_map.values()))  # 20 classes

        self.model = SalsaNext(nclasses=self.num_classes)

        print(f"Loading SalsaNext weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"SalsaNext model loaded successfully on {self.device}")
        print(f"Number of classes: {self.num_classes}")

    def project_to_range_image(self, points):
        """
        Project 3D point cloud to spherical range image

        Args:
            points: Nx3 array (x, y, z)

        Returns:
            range_image: 5xHxW tensor (range, x, y, z, remission)
            proj_idx: indices for unprojection
            proj_mask: valid pixel mask
        """
        # Compute depth
        depth = np.linalg.norm(points[:, :3], axis=1)

        # Compute pitch and yaw
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # Get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # [0, 1]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov  # [0, 1]

        # Scale to image size
        proj_x = np.floor(proj_x * self.img_width).astype(np.int32)
        proj_y = np.floor(proj_y * self.img_height).astype(np.int32)

        # Clamp values
        proj_x = np.clip(proj_x, 0, self.img_width - 1)
        proj_y = np.clip(proj_y, 0, self.img_height - 1)

        # Order by depth (keep closest point at each pixel)
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        scan_x = scan_x[order]
        scan_y = scan_y[order]
        scan_z = scan_z[order]

        # Create range image
        proj_range = np.full((self.img_height, self.img_width), -1, dtype=np.float32)
        proj_xyz = np.full((self.img_height, self.img_width, 3), -1, dtype=np.float32)
        proj_remission = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        proj_idx = np.full((self.img_height, self.img_width), -1, dtype=np.int32)
        proj_mask = np.zeros((self.img_height, self.img_width), dtype=np.int32)

        # Fill in range image
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = np.stack([scan_x, scan_y, scan_z], axis=-1)
        proj_idx[proj_y, proj_x] = order
        proj_mask[proj_y, proj_x] = 1

        # Stack channels: [range, x, y, z, remission]
        range_image = np.concatenate([
            proj_range[np.newaxis, :, :],
            proj_xyz.transpose(2, 0, 1),  # x, y, z
            proj_remission[np.newaxis, :, :]
        ], axis=0).astype(np.float32)

        return range_image, proj_idx, proj_mask

    def predict(self, points_xyz):
        """
        Run semantic segmentation on point cloud

        Args:
            points_xyz: Nx3 or Nx4+ array (x, y, z, [intensity, ...])

        Returns:
            points_down: Nx3 array (x, y, z) — all input points
            colors: Nx3 float32 RGB colors derived from predicted class
            predictions: N integer class labels (0 for unmapped/unlabeled)
            probabilities: Nx20 float32 class probabilities (all-zero for unmapped points)
        """
        t0 = time()

        # Handle input format
        if points_xyz.shape[1] > 3:
            points = points_xyz[:, :3]  # Only use XYZ for outdoor
        else:
            points = points_xyz

        # Project to range image
        range_image, proj_idx, proj_mask = self.project_to_range_image(points)

        # Normalize range image
        img_means = np.array(self.sensor_cfg['img_means'])[:, np.newaxis, np.newaxis]
        img_stds = np.array(self.sensor_cfg['img_stds'])[:, np.newaxis, np.newaxis]
        range_image = (range_image - img_means) / (img_stds + 1e-8)

        t1 = time()
        if self.verbose:
            print(f'  [SalsaNext] projection + normalization: {t1 - t0:.3f}s')

        # Convert to tensor (ensure float32)
        range_tensor = torch.from_numpy(range_image).unsqueeze(0).float().to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(range_tensor)
            output = output.squeeze(0)  # CHW

            # Get predictions
            probs = F.softmax(output, dim=0)  # Class probabilities
            preds_2d = torch.argmax(probs, dim=0)  # HxW

        t2 = time()
        if self.verbose:
            print(f'  [SalsaNext] model forward pass:         {t2 - t1:.3f}s')

        # Unproject to 3D
        preds_2d_np = preds_2d.cpu().numpy()
        probs_np = probs.cpu().numpy().transpose(1, 2, 0)  # HxW x C

        # Extract predictions for valid points only
        valid_mask = proj_mask > 0

        # Get all valid projected indices
        valid_y, valid_x = np.where(valid_mask)
        valid_point_indices = proj_idx[valid_y, valid_x]

        # Get predictions and probabilities for these pixels
        valid_preds = preds_2d_np[valid_y, valid_x]
        valid_probs = probs_np[valid_y, valid_x]

        # Map back to original point order
        preds_1d = np.zeros(len(points), dtype=np.int32)
        probs_1d = np.zeros((len(points), self.num_classes), dtype=np.float32)
        preds_1d[valid_point_indices] = valid_preds
        probs_1d[valid_point_indices] = valid_probs

        t3 = time()
        if self.verbose:
            print(f'  [SalsaNext] unproject to 3D:            {t3 - t2:.3f}s')

        # Keep ALL points so the occupancy grid is fully populated.
        # Points that were occluded, outside the vertical FOV, or predicted as
        # class-0 (unlabeled) are left with all-zero probabilities.  LocalGrid's
        # semantic_probability_threshold will naturally exclude them from semantic
        # layers while still using them to build the occupancy grid.
        points_down = points
        preds_down = preds_1d
        probs_down = probs_1d

        # Map learning labels back to semantic labels for coloring
        sem_labels = np.vectorize(self.learning_map_inv.get)(preds_down, preds_down)

        # Create fake RGB from class labels for visualization
        colors = np.zeros((len(points_down), 3), dtype=np.float32)
        # Create color lookup array
        max_label = max(self.label_cfg['color_map'].keys())
        color_lut = np.zeros((max_label + 1, 3), dtype=np.float32)
        for label, bgr in self.label_cfg['color_map'].items():
            color_lut[label] = [bgr[2], bgr[1], bgr[0]]  # BGR to RGB
        color_lut = color_lut / 255.0  # Normalize

        # Apply colors (clip sem_labels to valid range)
        valid_labels = np.clip(sem_labels, 0, max_label)
        colors = color_lut[valid_labels]

        t4 = time()
        if self.verbose:
            print(f'  [SalsaNext] colorization:               {t4 - t3:.3f}s')
            print(f'  [SalsaNext] total predict():            {t4 - t0:.3f}s  ({len(points)} points)')

        return points_down, colors, preds_down, probs_down


if __name__ == '__main__':
    print("Testing SalsaNext inference...")
    inf = SalsaNextInference(device='cuda:0')

    test_cloud_path = '/data/topo_graph_from_route1_auto_logs/clouds/65.npz'
    cloud = np.load(test_cloud_path)['arr_0']
    print(f"Loaded test cloud: {cloud.shape}")

    pts, colors, preds, probs = inf.predict(cloud)
    print(f"Output: {pts.shape}, {colors.shape}, {preds.shape}, {probs.shape}")
    print(f"Unique classes: {np.unique(preds)}")
    print(f"Class distribution:")
    for cls in np.unique(preds):
        count = np.sum(preds == cls)
        print(f"  Class {cls}: {count} points ({count/len(preds)*100:.1f}%)")
