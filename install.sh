#!/usr/bin/env bash
# ============================================================
#  RAS598 Assignment 3 – Install & Build Script
#  Run this script ONCE after copying the package to your VM.
#
#  Usage:
#    chmod +x install.sh
#    ./install.sh
# ============================================================
set -e

echo "=== [1/4] Copying package to ROS2 workspace ==="
DEST=~/ros2_ws/src/ras598_assignment_3
if [ -d "$DEST" ]; then
    echo "  -> Removing old copy at $DEST"
    rm -rf "$DEST"
fi
cp -r "$(dirname "$0")" "$DEST"
echo "  -> Copied to $DEST"

echo ""
echo "=== [2/4] Installing Python dependencies ==="
pip3 install scipy numpy --break-system-packages 2>/dev/null || true

echo ""
echo "=== [3/4] Building the workspace ==="
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ras598_assignment_3 --symlink-install
echo "  -> Build complete"

echo ""
echo "=== [4/4] Sourcing the workspace ==="
source ~/ros2_ws/install/setup.bash

echo ""
echo "=============================================="
echo " DONE!  To run the assignment:"
echo ""
echo "   source ~/ros2_ws/install/setup.bash"
echo "   ros2 launch ras598_assignment_3 bayes_launch.py"
echo "=============================================="
