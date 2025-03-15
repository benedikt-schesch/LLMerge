#!/bin/bash
apt-get install screen -y
git clone https://github.com/benedikt-schesch/LLMerge.git
cd LLMerge
pip install uv
uv sync
screen
