#!/bin/bash
# Script to collect all merges on a small dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_1000_1200.csv merges/repos_reaper_test "$@" --test_size 1.0
