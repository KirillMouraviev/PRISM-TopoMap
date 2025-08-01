#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class OdometryPublisher(Node):
    def __init__(self):
        super().__init__('odometry_publisher')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('target_frame', 'base_footprint')
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.odometry_publisher = self.create_publisher(
            Odometry, 
            '/odom_gt', 
            qos_profile=rclpy.qos.qos_profile_system_default
        )
        self.timer = self.create_timer(0.1, self.publish_odometry)  # 10Hz

    def publish_odometry(self):
        odom_msg = Odometry()
        odom_msg.header.frame_id = self.source_frame
        odom_msg.child_frame_id = self.target_frame
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.source_frame, 
                self.target_frame, 
                rclpy.time.Time().to_msg()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f'Could not get transform from {self.source_frame} to {self.target_frame}: {str(e)}'
            )
            return
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.pose.pose.position.x = transform_stamped.transform.translation.x
        odom_msg.pose.pose.position.y = transform_stamped.transform.translation.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = transform_stamped.transform.rotation
        self.odometry_publisher.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    odometry_publisher = OdometryPublisher()
    rclpy.spin(odometry_publisher)
    odometry_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()