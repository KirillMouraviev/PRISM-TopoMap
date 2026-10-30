#!/usr/bin/env python
# Compatibility wrapper: launch files reference tf_to_odom.py
from tf_manager import OdometryPublisher

if __name__ == '__main__':
    odometry_publisher = OdometryPublisher()
    odometry_publisher.run()
