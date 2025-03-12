#!/bin/bash
apt-get install screen -y
git clone git@github.com:benedikt-schesch/LLMerge.git
cd LLMerge
pip install uv
uv sync
screen
