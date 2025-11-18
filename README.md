# PRISM-TopoMap
The PRISM-TopoMap - online topological mapping method with place recognition and scan matching. It is able to build a lightweight topological maps (graph of locations) of large indoor or outdoor environments and localize in the built maps without error accumulation.



https://github.com/user-attachments/assets/2608eb34-f8a1-4062-be5b-5ae8985fff35



## Paper

The PRISM-TopoMap method is described in the [paper](https://arxiv.org/abs/2404.01674) accepted to IEEE Robotics and Automation Letters in 2025. If you use this code in your research, please cite this paper. A bibtex entry is provided below.

```
@article{muravyev2025prism,
  title={PRISM-TopoMap: online topological mapping with place recognition and scan matching},
  author={Muravyev, Kirill and Melekhin, Alexander and Yudin, Dmitry and Yakovlev, Konstantin},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```

Comparison of Place Recognition methods is in the [PR_ablation.md](PR_ablation.md).

## Overview

**Modes**:
- SLAM
- Localization only

**Inputs**
- Odometry
- Point cloud
- Images (optional)

PRISM-TopoMap takes input data using ROS1 (ROS2 version will release soon). In SLAM mode, it can build map from scratch or update a pre-built map, and save the updated map by a specified path. In localization only mode, PRISM-TopoMap loads a pre-built map from a specified path and localizes in it without any map change.

To run PRISM-TopoMap on your own robot or dataset, you need to install it as a ROS package, create a ROS launch file, and create a .yaml config file with all the input topic names and algorithm parameters (desribed below).

## Prerequisites:
- [OpenPlaceRecognition](https://github.com/alexmelekhin/openplacerecognition)
- [ROS](https://ros.org) Melodic or Noetic
- For simulated demo: [habitat-sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7), [habitat-lab v0.1.7](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7), and [ROS tools for Habitat](https://github.com/CnnDepth/habitat_ros/tree/toposlam_experiments)

## Docker env

We provide a Dockerfile for building the environment with all the dependencies.

There are two versions:

- `prism-dev`: basic environment without ROS2
- `prism-dev-ros2`: environment with ROS2 support

To build the image run:

```bash
# without ros2
docker compose build prism-dev

# with ros2
docker compose build prism-dev-ros2
```

After you build the image, you can run it and attach interactive terminal:

```bash
# without ros2
docker compose run --rm -it prism-dev bash

# with ros2
docker compose run --rm -it prism-dev-ros2 bash
```


## Installation
After installing ROS and OpenPlaceRecognition, build PRISM-TopoMap as ROS package:
```
cd your_ros_workspace/src
git clone https://github.com/KirillMouraviev/toposlam_msgs
git clone https://github.com/KirillMouraviev/PRISM-TopoMap
cd ../..
catkin_make
```

After that, download the actual weights: [multimodal](https://drive.google.com/file/d/1r4Nw0YqHC9PKiZXDmUAWZkOTvgporPnS/view?usp=sharing) or [point cloud only](https://drive.google.com/file/d/19uPohPxQUa71jQzApjGPVbSyJeVdggwF/view?usp=drive_link) for the place recognition model and set correct path to the weights in the config files `habitat_mp3d.yaml` and `scout_rosbag.yaml`.

## Demo
We provide two examples of usage: SLAM and localization in Habitat simulation and SLAM on rosbag from AgileX Scout Mini robot. The scenes for simulation demo can be downloaded from [Yandex](https://disk.yandex.ru/d/_dJPBG213Wc5CA) or [Google](https://drive.google.com/file/d/1ex1fyK5xDVIptnTVlzxpcSp5Zd0Cby0a/view?usp=sharing) drive. The rosbag from Scout robot can be downloaded [here](https://disk.yandex.ru/d/BT2iuAQKsmswmg).

Launch files for the demo contain the following args:
- `scene_name` (for simulated demo only): name of the scene from the dataset
- `path_to_save_graph`: absolute path to save the resulting topological map
- `path_to_load_graph` (for localization mode only): absolute path to the pre-built topological map
- `path_to_save_logs`: absolute path to dump all the debugging output files
- `config_file`: name of the config file (must be in `config` folder)

### SLAM on Scout robot

Terminal 1
```
roscore
```

Terminal 2
```
rosparam set use_sim_time true
cd <path to rosbags>
for bag in $(ls); do rosbag play $bag --clock; done;
```

Terminal 3
```
cd your_ros_workspace
source devel/setup.bash
roslaunch prism_topomap build_map_by_iou_scout_rosbag.launch
```

### SLAM in simulation

Terminal 1
```
roscore
```

Terminal 2
```
sudo -s
<source ROS and Habitat ros workspace>
roslaunch habitat_ros toposlam_experiment_mp3d_4x90_large_noise.launch
```

Terminal 3
```
cd your_ros_workspace
source devel/setup.bash
roslaunch prism_topomap build_map_by_iou_habitat.launch
```

### Localization in simulation

May be launched after running SLAM on the same simulated scene. Specify the path to the map saved by SLAM via `path_to_load_graph` launch arg.

Terminal 1
```
roscore
```

Terminal 2
```
sudo -s
<source ROS and Habitat ros workspace>
roslaunch habitat_ros toposlam_experiment_mp3d_4x90_large_noise.launch
```

Terminal 3
```
cd your_ros_workspace
source devel/setup.bash
roslaunch prism_topomap habitat_mp3d_localization.launch
```

## Parameters in config

**Input**

  - `subscribe_to_images`: use image input or not (`true/false`).
  - `subscribe_to_gt_pose`: use ground truth input pose (for visualization, evaluation, building a high-precision map for subsequent localization, etc.) or not (`true/false`).

  `pointcloud`:
  - `topic`: the ROS topic for input point cloud.
  - `fields`: point cloud fields (may be `xyz` or `xyzrgb` for colored point clouds).
  - `rotation_matrix`: the matrix of point cloud rotation relative to the standard axes directions (x is forward, y is left, z is up).
  - `floor_height`: the floor level in the input point clouds (relative to the observation point). If the floor on the scene is uneven, the highest floor level should be set. May be set `auto` for dense point clouds.
  - `ceiling_height`: the ceiling level int the input point clouds (relative to the observation point). If the ceiling on the scene is uneven, the lowest ceiling level should be set. May be set `auto` for dense point clouds.
  - `subscribe_to_curbs`: use curbs data to refine scan matching or not (for outdoor usage, the curbs must be in PointCloud2 format).
  - `curb_detection_topic`: the ROS topic for curbs in point cloud format (if `subscribe_to_curbs` param is set to `true`).

  `odometry`:
  - `topic`: the ROS topic for input odometry.
  - `type`: ROS message type in the odometry topic (may be `PoseStamped` or `Odometry`).

  `gt_pose` (if `subscribe_to_gt_pose` is set to true):
  - `topic`: the ROS topic for ground truth input pose (for visualization, evaluation, building a high-precision map for subsequent localization, etc.).
  - `type`: ROS message type in the odometry topic (may be `PoseStamped` or `Odometry`).

  `image_front` (if `subscribe_to_images` is set to true):
  - `topic`: the ROS topic for input front-view RGB image.
  - `color`: the color order (may be `rgb` or `bgr`).

  `image_back` (if `subscribe_to_images` is set to true):
  - `topic`: the ROS topic for input back-view RGB image.
  - `color`: the color order (may be `rgb` or `bgr`).

**Topomap**
- `mode`: PRISM-TopoMap mode (may be `mapping` or `localization`).
- `iou_threshold`: overlapping threshold for detachment from the current location.
- `max_edge_length`: maximum distance between the locations and between the robot and the current location's pose (if exceeded, a location change or addition is forced).
- `localization_frequency`: frequency of the localization module call in seconds.
- `localization_timeout`: time of waiting for localization at start (in seconds). After this time, PRISM-TopoMap initializes in a new node.
- `start_location` (optional): ID of the location at start of the algorithm if known.
- `start_local_pose` (optional): an array [x, y, theta] - position and rotation relative to the start location (if known).
- `drift_coef`: coefficient of odometry drift for preventing big position jumps due to localization errors. Larger value means larger position volatility. Typical values: 0.2 for a slowly moving robot, 1.0 for a car.

**Place recognition**
- `model`: the place recognition model type (supported types are `mssplace` (multimodal) or `minkloc3d` (point clouds only)).
- `weights_path`: path to the place recognition model weights.
- `model_config_path`: path to the place recognition model config.
- `pointcloud_quantization_size`: voxelization size in point cloud preprocessing for a place recognition model.

**Local occupancy grid**
- `resolution`: size of a cell of local occupancy grid (in meters) used by scan matching.
- `radius`: size of grid in meters.
- `max_range`: maximum distance from the robot to the objects projected by a grid (in meters).

**Scan matching**

- `model`: the scan matching model type (supported types are `icp`, `geotransformer`, `feature2d`).
- `detector_type`: the type of feature detector (for `feature2d` model type only). Supported types are `ORB`, `SIFT`, `HarrisWithDistance`.
- `score_threshold`: the scans are considered matched if the score is above this value. May be set in range (0, 1), the optimal values are in range (0.5, 0.8).
- `outlier_thresholds`: the thresholds for outlier removal from the matched keypoints (see Algorithm 1 in the paper). The length of the array sets the number of iterations.
- `min_matches`: the minimal number of the matches for the scans be considered matched.

**Scan matching along edge**

- `model`: the scan matching model type (supported types are `icp`, `geotransformer`, `feature2d`).
- `detector_type`: the type of feature detector (for `feature2d` model type only). Supported types are `ORB`, `SIFT`, `HarrisWithDistance`. `HarrisWithDistance` is highly recommended.
- `score_threshold`: the scans are considered matched if the score is above this value. May be set in range (0, 1), the optimal values are in range (0.5, 0.8).
- `outlier_thresholds`: the thresholds for outlier removal from the matched keypoints (see Algorithm 1 in the paper). The length of the array sets the number of iterations.
- `min_matches`: the minimal number of the matches for the scans be considered matched.
- `jump_threshold`: threshold of the position change due to transition along edge. If the position shift exceeds this threshold, the match is considered spurious and ignored.

**Visualization**
- `publish_gt_map`: the flag for the ground truth 2D grid map publication (should be set `true` only if `path_to_gt_map` parameter is correctly set).
- `map_frame`: the ROS frame for the ground truth map (if `publish_gt_map` is set `true`).
- `publish_tf_from_odom`: the flag for the ROS transformation publicatoin into topic `tf` from the input odometry.
- `vertex_marker_size`: size of graph vertex marker in RViz.
- `edge_marker_size`: size of graph edge marker in RViz.
- `text_marker_size`: size of the annotations of graph vertices and edges (numbers of vertices, positions).
- `match_marker_size`: size of marker denoting localization results.
- `vcur_marker_size`: size of marker denoting the current vertex in graph.
- `loop_closure_marker_size`: size of markers for vertices in the detected loop.
- `loop_closure_edge_marker_size`: thickness of edges in the detected loop.
- `path_marker_size`: thickness of edges on a built path in RViz (for navigation tasks).
- `vertex_orientation_marker_size`: thickness of a marker denoting the orientation vectors for each location in graph.
