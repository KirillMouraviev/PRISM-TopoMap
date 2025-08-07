#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

class GTBroadcaster(Node):
    def __init__(self):
        super().__init__('itlp_gt_tf_publisher')
        
        # Parameters
        self.declare_parameter('path_to_gt')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('odom_topic', '/odom')
        
        self.path_to_gt = self.get_parameter('path_to_gt').value
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.odom_topic = self.get_parameter('odom_topic').value
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tfbr = TransformBroadcaster(self)
        
        # Subscriber
        self.glim_odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        
        # Initialize variables
        self.gt_idx = 0
        track = pd.read_csv(self.path_to_gt)
        self.gt_stamps = track['timestamp'].values
        self.gt_poses = track.values[:, -7:]
        self.tf_map_to_reper = None
        self.tf_reper_to_odom = None

    def odom_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        gt_update = False
        
        while self.gt_idx < len(self.gt_stamps) and float(self.gt_stamps[self.gt_idx]) * 1e-6 < stamp:
            gt_update = True
            self.gt_idx += 1
            
        if gt_update:
            tx, ty, tz = self.gt_poses[self.gt_idx - 1, :3]
            qx, qy, qz, qw = self.gt_poses[self.gt_idx - 1, 3:]
            
            # Create map to reper transform
            self.tf_map_to_reper = TransformStamped()
            self.tf_map_to_reper.header.stamp = msg.header.stamp
            self.tf_map_to_reper.header.frame_id = self.source_frame
            self.tf_map_to_reper.child_frame_id = 'reper'
            self.tf_map_to_reper.transform.translation.x = tx
            self.tf_map_to_reper.transform.translation.y = ty
            self.tf_map_to_reper.transform.translation.z = tz
            self.tf_map_to_reper.transform.rotation.x = qx
            self.tf_map_to_reper.transform.rotation.y = qy
            self.tf_map_to_reper.transform.rotation.z = qz
            self.tf_map_to_reper.transform.rotation.w = qw

            # Create reper to odom transform
            self.tf_reper_to_odom = TransformStamped()
            self.tf_reper_to_odom.header.stamp = msg.header.stamp
            self.tf_reper_to_odom.header.frame_id = 'reper'
            self.tf_reper_to_odom.child_frame_id = msg.header.frame_id
            
            tf_matrix_odom_to_reper = np.eye(4)
            R = Rotation.from_quat([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]).as_matrix()
            
            tf_matrix_odom_to_reper[:3, :3] = R
            tf_matrix_odom_to_reper[:3, 3] = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            
            tf_matrix_reper_to_odom = np.linalg.inv(tf_matrix_odom_to_reper)
            q = Rotation.from_matrix(tf_matrix_reper_to_odom[:3, :3]).as_quat()
            
            self.tf_reper_to_odom.transform.translation.x = tf_matrix_reper_to_odom[0, 3]
            self.tf_reper_to_odom.transform.translation.y = tf_matrix_reper_to_odom[1, 3]
            self.tf_reper_to_odom.transform.translation.z = tf_matrix_reper_to_odom[2, 3]
            self.tf_reper_to_odom.transform.rotation.x = q[0]
            self.tf_reper_to_odom.transform.rotation.y = q[1]
            self.tf_reper_to_odom.transform.rotation.z = q[2]
            self.tf_reper_to_odom.transform.rotation.w = q[3]
        
        # Broadcast transforms if they exist
        if self.tf_map_to_reper is not None:
            self.tf_map_to_reper.header.stamp = msg.header.stamp
            self.tfbr.sendTransform(self.tf_map_to_reper)
            
        if self.tf_reper_to_odom is not None:
            self.tf_reper_to_odom.header.stamp = msg.header.stamp
            self.tfbr.sendTransform(self.tf_reper_to_odom)

def main(args=None):
    rclpy.init(args=args)
    gt_broadcaster = GTBroadcaster()
    rclpy.spin(gt_broadcaster)
    gt_broadcaster.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()