#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /home/docker_prism/ros2_ws/install/local_setup.bash
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

exec "$@"
