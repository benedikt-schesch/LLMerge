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
    apt install rsync
    echo "apt_maintenance() done"
}

## TODO: Fix. this breaks pip somehow
# # Install Python 3.12 and deps
# install_python_and_deps() {
#     echo "installing Python 3.12 ..."
#     add-apt-repository ppa:deadsnakes/ppa -y
#     apt_maintenance
#     apt install -y python3.12 python3.12-dev python3.12-venv python3.12-distutils
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
#     update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
#     apt install python3-pip
#     pip install --upgrade pip && pip install transformers torch tqdm accelerate xformers pretty_errors virtualenv
#     echo "Python installation done"
# }

cleanup_downloads() {
    # rm -rf ./cuda-*
    # rm -rf pytorch
    # rm -rf bitsandbytes
}

### MAIN

# necessary packages cuz runpod has a minimal distro
apt update && apt install tmux neovim less tree unzip htop lshw ffmpeg nvidia-cuda-toolkit libfuse2 -y

# this makes it so that pip will install in attached storage, instead of /root which gets filled up
mv /root/.cache /workspace/.cache && ln -s /workspace/.cache /root/.cache

apt_maintenance
# install_python_and_deps
pip install --upgrade pip
pip install uv
