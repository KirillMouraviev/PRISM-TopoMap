#!/usr/bin/env python
import rospy
import numpy as np
import pandas as pd
import tf
from tf import TransformBroadcaster
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

class GlobalTransformPublisher:
    def __init__(self):
        rospy.init_node('global_transform_publisher')
        path_to_track = rospy.get_param('~path_to_track')
        self.track = pd.read_csv(path_to_track, index_col=0)
        self.i = 0
        self.cloud_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/glim_ros/odom', Odometry, self.odom_callback)
        self.tfbr = TransformBroadcaster()
        self.odom_position = None
        self.odom_rotation = None
        self.landmark_pos = [0, 0, 0]
        self.landmark_quat = [0, 0, 0, 1]
        self.inv_odom_pos = None
        self.inv_odom_quat = None

    def odom_callback(self, msg):
        self.odom_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.odom_rotation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

    def lidar_callback(self, msg):
        if self.i >= self.track.shape[0]:
            return
        stamp = int(msg.header.stamp.to_sec() * 1e6)
        row = self.track.iloc[self.i]
        cur_reper_stamp = int(row['lidar_ts'])
        #print('Before', self.i, cur_reper_stamp, stamp)
        while self.i < self.track.shape[0] and int(self.track.iloc[self.i]['lidar_ts']) < stamp:
            row = self.track.iloc[self.i]
            cur_reper_stamp = int(row['lidar_ts'])
            #print('While', self.i, cur_reper_stamp, stamp)
            self.i += 1
        #print(self.i, cur_reper_stamp, stamp)
        if self.i >= self.track.shape[0]:
            print('Passed all the track')
            return
        if stamp == cur_reper_stamp:
            print('Point {}'.format(self.i))
            pos = [row['tx'], row['ty'], row['tz']]
            quat = [row['qx'], row['qy'], row['qz'], row['qw']]
            self.landmark_pos = pos
            self.landmark_quat = quat
            if self.odom_position is not None:
                odom_tf_matrix = tf.transformations.quaternion_matrix(self.odom_rotation)
                odom_tf_matrix[:3, 3] = self.odom_position
                odom_tf_inverse_matrix = np.linalg.inv(odom_tf_matrix)
                inverse_quat = tf.transformations.quaternion_from_matrix(odom_tf_inverse_matrix)
                inverse_pos = odom_tf_inverse_matrix[:3, 3]
                self.inv_odom_pos = inverse_pos
                self.inv_odom_quat = inverse_quat
            else:
                print('No odometry!')
        self.tfbr.sendTransform(self.landmark_pos, self.landmark_quat, msg.header.stamp, 'landmark', 'map')
        if self.inv_odom_quat is not None:
            self.tfbr.sendTransform(self.inv_odom_pos, self.inv_odom_quat, msg.header.stamp, 'odom', 'landmark')

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    global_transform_publisher = GlobalTransformPublisher()
    global_transform_publisher.run()