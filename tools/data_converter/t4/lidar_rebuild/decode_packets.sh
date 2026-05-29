#!/usr/bin/env bash
# Decode raw Hesai pandar_packets from a T4 input_bag into per-lidar point clouds
# with real ring (channel) + per-point time, using Autoware nebula offline decoder.
#
# The T4 LIDAR_CONCAT/*.pcd.bin lacks ring and per-point time (they are dropped
# when the online concatenation runs with num_lidar_feats=5). This recovers them
# from the raw UDP packets, which carry ring+time at the protocol level.
#
# Usage:
#   decode_packets.sh <input_bag_dir> <out_dir> [autoware_setup.bash]
#
# Sensor models are auto-selected per position below. Verified from packet
# headers for the XX1 Odaiba kit: top=Pandar128E4X (128-line), the five 32-line
# units = PandarXT32. Adjust MODELS if your kit differs.
set -euo pipefail

BAG=${1:?usage: decode_packets.sh <input_bag_dir> <out_dir> [autoware_setup.bash]}
OUT=${2:?usage: decode_packets.sh <input_bag_dir> <out_dir> [autoware_setup.bash]}
AUTOWARE_SETUP=${3:-/home/binwang/work/autoware-core/repos/autoware/install/setup.bash}

source /opt/ros/humble/setup.bash
source "$AUTOWARE_SETUP"

declare -A MODELS=(
  [top]=Pandar128E4X
  [front_left]=PandarXT32
  [front_right]=PandarXT32
  [side_left]=PandarXT32
  [side_right]=PandarXT32
  [rear]=PandarXT32
)

mkdir -p "$OUT"
for pos in top front_left front_right side_left side_right rear; do
  model=${MODELS[$pos]}
  echo "=========== decode $pos ($model) ==========="
  rm -rf "${OUT:?}/$pos"
  ros2 launch nebula_examples hesai_offline_bag_pcd.xml \
    sensor_model:="$model" \
    bag_path:="$BAG" \
    input_topic:=/sensing/lidar/"$pos"/pandar_packets \
    output_topic:=/sensing/lidar/"$pos"/points \
    out_path:="$OUT/$pos" \
    only_xyz:=false output_pcd:=false output_rosbag:=true \
    out_num:=0 2>&1 | grep -iE "Ending|finished cleanly|error|died|return mode" | tail -2
  echo "  $pos done"
done
echo "ALL_DECODE_DONE"
