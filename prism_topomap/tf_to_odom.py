#!/usr/bin/env python

import rospy
import numpy as np
import tf2_ros
from nav_msgs.msg import Odometry

class OdometryPublisher:
    def __init__(self):
        rospy.init_node('odometry_publisher')
        self.source_frame = rospy.get_param('~source_frame', 'map')
        self.target_frame = rospy.get_param('~target_frame', 'base_footprint')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.rate = rospy.Rate(10)
        self.odometry_publisher = rospy.Publisher('/odom_gt', Odometry, latch=True, queue_size=100)

    def run(self):
        odom_msg = Odometry()
        odom_msg.header.frame_id = self.source_frame
        odom_msg.child_frame_id = self.target_frame
        while not rospy.is_shutdown():
            try:
                transform_stamped = self.tf_buffer.lookup_transform(self.source_frame, self.target_frame, rospy.Time(0))
                                                            #self.tf_listener.getLatestCommonTime(self.source_frame, self.target_frame))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print('Could not get transform from {} to {}'.format(self.source_frame, self.target_frame))
                continue
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.pose.pose.position.x = transform_stamped.transform.translation.x
            odom_msg.pose.pose.position.y = transform_stamped.transform.translation.y
            odom_msg.pose.pose.position.z = 0
            odom_msg.pose.pose.orientation = transform_stamped.transform.rotation
            self.odometry_publisher.publish(odom_msg)
            self.rate.sleep()

if __name__ == '__main__':
    odometry_publisher = OdometryPublisher()
    odometry_publisher.run()