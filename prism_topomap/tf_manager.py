#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

class OdometryPublisher(Node):
    def __init__(self):
        super().__init__('odometry_publisher')
        
        # Parameters
        self.declare_parameter('publish_odom_from_tf', False)
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('publish_tf_from_odom', False)
        self.declare_parameter('odometry_topic', '/odom')
        self.declare_parameter('tf_from_odom_target_frame', 'base_link')
        
        self.publish_odom_from_tf = self.get_parameter('publish_odom_from_tf').value
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.publish_tf_from_odom = self.get_parameter('publish_tf_from_odom').value
        self.odometry_topic = self.get_parameter('odometry_topic').value
        self.tf_from_odom_target_frame = self.get_parameter('tf_from_odom_target_frame').value
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher
        self.odometry_publisher = self.create_publisher(
            Odometry,
            '/odom_gt',
            100  # QoS profile depth (replaced queue_size)
        )
        
        # Subscriber and TF broadcaster if needed
        if self.publish_tf_from_odom:
            self.odom_sub = self.create_subscription(
                Odometry,
                self.odometry_topic,
                self.odom_callback,
                10  # QoS profile depth
            )
            self.tfbr = TransformBroadcaster(self)
        
        # Timer for periodic execution if not using subscriber
        if not self.publish_tf_from_odom:
            self.timer = self.create_timer(0.1, self.extract_tf_and_publish_odom)  # 10Hz

    def odom_callback(self, msg):
        tf_msg = TransformStamped()
        tf_msg.header = msg.header
        tf_msg.child_frame_id = self.tf_from_odom_target_frame
        
        try:
            tf_sensor_to_base = self.tf_buffer.lookup_transform(
                msg.child_frame_id,
                self.tf_from_odom_target_frame,
                rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f'Could not get transform from {msg.child_frame_id} to {self.tf_from_odom_target_frame}: {str(e)}'
            )
            return
        
        # Create transformation matrices
        tf_matrix_odom_to_sensor = np.eye(4)
        tf_matrix_odom_to_sensor[:3, 3] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        
        R = Rotation.from_quat([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]).as_matrix()
        tf_matrix_odom_to_sensor[:3, :3] = R
        
        tf_matrix_sensor_to_base = np.eye(4)
        tf_matrix_sensor_to_base[:3, 3] = [
            tf_sensor_to_base.transform.translation.x,
            tf_sensor_to_base.transform.translation.y,
            tf_sensor_to_base.transform.translation.z
        ]
        
        R = Rotation.from_quat([
            tf_sensor_to_base.transform.rotation.x,
            tf_sensor_to_base.transform.rotation.y,
            tf_sensor_to_base.transform.rotation.z,
            tf_sensor_to_base.transform.rotation.w
        ]).as_matrix()
        tf_matrix_sensor_to_base[:3, :3] = R
        
        # Combine transformations
        tf_matrix_odom_to_base = tf_matrix_odom_to_sensor @ tf_matrix_sensor_to_base
        
        # Fill transform message
        tf_msg.transform.translation.x = tf_matrix_odom_to_base[0, 3]
        tf_msg.transform.translation.y = tf_matrix_odom_to_base[1, 3]
        tf_msg.transform.translation.z = tf_matrix_odom_to_base[2, 3]
        
        q = Rotation.from_matrix(tf_matrix_odom_to_base[:3, :3]).as_quat()
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        
        # Broadcast transform
        self.tfbr.sendTransform(tf_msg)
        self.extract_tf_and_publish_odom()

    def extract_tf_and_publish_odom(self):
        odom_msg = Odometry()
        odom_msg.header.frame_id = self.source_frame
        odom_msg.child_frame_id = self.target_frame
        
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f'Could not get transform from {self.source_frame} to {self.target_frame}: {str(e)}'
            )
            return
        
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.pose.pose.position.x = transform_stamped.transform.translation.x
        odom_msg.pose.pose.position.y = transform_stamped.transform.translation.y
        odom_msg.pose.pose.position.z = 0.0  # Explicitly setting Z to 0
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