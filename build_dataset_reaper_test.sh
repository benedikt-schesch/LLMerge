#!/bin/bash
# Script to collect all merges on a small dataset

./build_dataset.sh input_data/repos_reaper_1000_1500.csv merges/repos_reaper_test "$@" --test_size 1.0
