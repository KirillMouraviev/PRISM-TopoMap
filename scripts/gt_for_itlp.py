#!/usr/bin/env python

import rospy
import numpy as np
import pandas as pd
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

class GTBroadcaster:
    def __init__(self):
        rospy.init_node('itlp_gt_tf_publisher')
        self.path_to_gt = rospy.get_param('~path_to_gt')
        self.source_frame = rospy.get_param('~source_frame', 'map')
        self.target_frame = rospy.get_param('~target_frame', 'base_link')
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.glim_odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.rate = rospy.Rate(10)
        self.tfbr = tf2_ros.TransformBroadcaster()
        self.gt_idx = 0

        track = pd.read_csv(self.path_to_gt)
        self.gt_stamps = track['timestamp'].values
        self.gt_poses = track.values[:, -7:]
        self.tf_map_to_reper = None
        self.tf_reper_to_odom = None

    def odom_callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        gt_update = False
        while self.gt_idx < len(self.gt_stamps) and float(self.gt_stamps[self.gt_idx]) * 1e-6 < stamp:
            gt_update = True
            self.gt_idx += 1
        if gt_update:
            tx, ty, tz = self.gt_poses[self.gt_idx - 1, :3]
            qx, qy, qz, qw = self.gt_poses[self.gt_idx - 1, 3:]
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
            self.tf_reper_to_odom.transform.translation.x, \
            self.tf_reper_to_odom.transform.translation.y, \
            self.tf_reper_to_odom.transform.translation.z = tf_matrix_reper_to_odom[:3, 3]
            self.tf_reper_to_odom.transform.rotation.x, \
            self.tf_reper_to_odom.transform.rotation.y, \
            self.tf_reper_to_odom.transform.rotation.z, \
            self.tf_reper_to_odom.transform.rotation.w = q
        if self.tf_map_to_reper is not None:
            self.tf_map_to_reper.header.stamp = msg.header.stamp
            self.tfbr.sendTransform(self.tf_map_to_reper)
        if self.tf_reper_to_odom is not None:
            self.tf_reper_to_odom.header.stamp = msg.header.stamp
            self.tfbr.sendTransform(self.tf_reper_to_odom)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    gt_broadcaster = GTBroadcaster()
    gt_broadcaster.run()