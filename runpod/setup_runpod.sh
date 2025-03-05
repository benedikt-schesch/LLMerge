#!/bin/bash

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root or with sudo"
   exit 1
fi

### FUNCTIONS

apt_maintenance() {
    echo "apt_maintenance() ..."
    apt update -y
    apt upgrade -y
    apt dist-upgrade -y
    apt autoremove -y
    apt autoclean -y
    echo "apt_maintenance() done"
}

### MAIN

apt update && apt install screen rsync htop vim less tree unzip htop lshw ffmpeg nvidia-cuda-toolkit libfuse2 -y
mv /root/.cache /workspace/.cache && ln -s /workspace/.cache /root/.cache

apt_maintenance
pip install --upgrade pip
pip install uv

cd
git clone https://github.com/benedikt-schesch/LLMerge.git
cd LLMerge
uv sync
screen
