#!/usr/bin/env python

import rospy
import numpy as np
import tf
from nav_msgs.msg import Odometry

class OdometryPublisher:
    def __init__(self):
        rospy.init_node('odometry_publisher')
        self.source_frame = rospy.get_param('~source_frame', 'map')
        self.target_frame = rospy.get_param('~target_frame', 'base_footprint')
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(10)
        self.odometry_publisher = rospy.Publisher('/odom_corrected', Odometry, latch=True, queue_size=100)

    def run(self):
        odom_msg = Odometry()
        odom_msg.header.frame_id = self.source_frame
        odom_msg.child_frame_id = self.target_frame
        while not rospy.is_shutdown():
            try:
                pos, quat = self.tf_listener.lookupTransform(self.source_frame, self.target_frame,
                                                            self.tf_listener.getLatestCommonTime(self.source_frame, self.target_frame))
            except:
                print('Could not get transform from {} to {}'.format(self.source_frame, self.target_frame))
                continue
            x, y, _ = pos
            qx, qy, qz, qw = quat
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.pose.pose.position.x = x
            odom_msg.pose.pose.position.y = y
            odom_msg.pose.pose.position.z = 0
            odom_msg.pose.pose.orientation.x = qx
            odom_msg.pose.pose.orientation.y = qy
            odom_msg.pose.pose.orientation.z = qz
            odom_msg.pose.pose.orientation.w = qw
            self.odometry_publisher.publish(odom_msg)
            self.rate.sleep()

if __name__ == '__main__':
    odometry_publisher = OdometryPublisher()
    odometry_publisher.run()