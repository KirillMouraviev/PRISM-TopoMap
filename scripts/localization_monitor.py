#!/usr/bin/env python
"""Monitor PRISM-TopoMap localization error by comparing GT and estimated poses."""

import csv
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import rospy
import tf2_ros
from std_msgs.msg import Int32
from geometry_msgs.msg import TransformStamped


class LocalizationMonitor:
    def __init__(self):
        rospy.init_node('localization_monitor', anonymous=True)
        self.gt_parent = rospy.get_param('~gt_parent_frame', 'odom')
        self.gt_child = rospy.get_param('~gt_child_frame', 'base_link')
        self.est_parent = rospy.get_param('~est_parent_frame', 'odom')
        self.est_child = rospy.get_param('~est_child_frame', 'current_state')
        self.failure_threshold = rospy.get_param('~failure_threshold', 5.0)
        self.log_dir = rospy.get_param('~log_dir', '/tmp/prism_localization_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.log_dir, f'pose_error_{timestamp}.csv')
        self.events_path = os.path.join(self.log_dir, f'failure_events_{timestamp}.csv')

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.current_vertex_id = -1
        rospy.Subscriber('/last_vertex_id', Int32, self.vertex_callback, queue_size=10)

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'ros_time', 'sim_time', 'pos_error_m', 'yaw_error_deg',
            'gt_x', 'gt_y', 'gt_yaw', 'est_x', 'est_y', 'est_yaw',
            'vertex_id', 'status'
        ])

        self.events_file = open(self.events_path, 'w', newline='')
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow([
            'ros_time', 'sim_time', 'event', 'pos_error_m', 'vertex_id', 'details'
        ])

        self.in_failure = False
        self.failure_start_time = None
        self.max_error = 0.0
        self.sample_count = 0
        self.failure_count = 0
        self.rate = rospy.Rate(10)

    def vertex_callback(self, msg):
        self.current_vertex_id = msg.data

    @staticmethod
    def transform_to_pose(transform):
        x = transform.transform.translation.x
        y = transform.transform.translation.y
        q = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return x, y, yaw

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def lookup_pose(self, parent, child):
        try:
            transform = self.tf_buffer.lookup_transform(
                parent, child, rospy.Time(0), rospy.Duration(0.1)
            )
            return self.transform_to_pose(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            return None

    def log_event(self, event, pos_error, details=''):
        sim_time = rospy.Time.now().to_sec()
        self.events_writer.writerow([
            rospy.Time.now().to_sec(), sim_time, event, pos_error,
            self.current_vertex_id, details
        ])
        self.events_file.flush()
        rospy.logwarn(f'[{event}] pos_error={pos_error:.2f}m vertex={self.current_vertex_id} {details}')

    def run(self):
        rospy.loginfo(f'Monitoring {self.gt_parent}->{self.gt_child} vs '
                      f'{self.est_parent}->{self.est_child}, threshold={self.failure_threshold}m')
        rospy.loginfo(f'Logging to {self.csv_path}')

        while not rospy.is_shutdown():
            gt_pose = self.lookup_pose(self.gt_parent, self.gt_child)
            est_pose = self.lookup_pose(self.est_parent, self.est_child)

            if gt_pose is None or est_pose is None:
                self.rate.sleep()
                continue

            gt_x, gt_y, gt_yaw = gt_pose
            est_x, est_y, est_yaw = est_pose
            pos_error = math.sqrt((gt_x - est_x) ** 2 + (gt_y - est_y) ** 2)
            yaw_error = math.degrees(abs(self.angle_diff(gt_yaw, est_yaw)))

            self.sample_count += 1
            self.max_error = max(self.max_error, pos_error)

            if pos_error > self.failure_threshold:
                status = 'FAILURE'
                if not self.in_failure:
                    self.in_failure = True
                    self.failure_count += 1
                    self.failure_start_time = rospy.Time.now().to_sec()
                    self.log_event('FAILURE_START', pos_error,
                                   f'gt=({gt_x:.1f},{gt_y:.1f}) est=({est_x:.1f},{est_y:.1f})')
            else:
                status = 'OK'
                if self.in_failure:
                    duration = rospy.Time.now().to_sec() - self.failure_start_time
                    self.log_event('FAILURE_END', pos_error, f'duration={duration:.1f}s')
                    self.in_failure = False

            sim_time = rospy.Time.now().to_sec()
            self.csv_writer.writerow([
                rospy.Time.now().to_sec(), sim_time, pos_error, yaw_error,
                gt_x, gt_y, gt_yaw, est_x, est_y, est_yaw,
                self.current_vertex_id, status
            ])

            if self.sample_count % 50 == 0:
                self.csv_file.flush()
                rospy.loginfo(
                    f'samples={self.sample_count} error={pos_error:.2f}m '
                    f'max={self.max_error:.2f}m failures={self.failure_count} '
                    f'vertex={self.current_vertex_id} status={status}'
                )

            self.rate.sleep()

    def cleanup(self):
        self.csv_file.close()
        self.events_file.close()
        rospy.loginfo(f'Done. samples={self.sample_count} max_error={self.max_error:.2f}m '
                      f'failures={self.failure_count}')


if __name__ == '__main__':
    monitor = LocalizationMonitor()
    try:
        monitor.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        monitor.cleanup()
