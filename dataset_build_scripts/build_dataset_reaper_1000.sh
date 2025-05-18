#!/bin/bash
# Script to collect all merges on a small dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_0_1000.csv merges/repos_reaper_1000 "$@" --max_num_merges 1000
