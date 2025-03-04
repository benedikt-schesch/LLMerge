#!/bin/bash
# Script to collect all merges on a small dataset

./build_dataset.sh input_data/repos_small.csv merges/repos_small -keep_trivial_resolution "$@"
