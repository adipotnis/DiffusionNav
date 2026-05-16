#!/bin/bash
# One-time setup: create a conda env with rosbag (Python 3, no full ROS needed).
# Run this interactively once: bash train/setup_rosbag_env.sh
#
# Uses the rosbag-only pip package which provides the reader without needing
# a full ROS install. If it fails, fall back to the conda-forge ROS stack.

set -euo pipefail

ENV_NAME="rosbag_proc"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "env ${ENV_NAME} already exists -- skipping create"
else
    conda create -y -n "$ENV_NAME" python=3.10 pip
fi

source ~/.bashrc
conda activate "$ENV_NAME"

# rosbag reader (pure python, no ROS daemon needed)
pip install rosbag gnupg pycryptodomex catkin-pkg rospkg
# image / pickle / tqdm deps used by process_bags.py
pip install Pillow tqdm PyYAML numpy

echo ""
echo "Done. Test with:"
echo "  conda activate ${ENV_NAME} && python -c 'import rosbag; print(rosbag.__file__)'"
