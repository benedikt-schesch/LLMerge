#!/bin/bash
# Script to collect all merges

REPOS_DIR=$1
OUT_DIR=$2

python3 src/find_merges.py --repos $REPOS_DIR --output_file $OUT_DIR/merges.csv

python3 src/extract_conflict_files.py --merges $OUT_DIR/merges.csv --output_dir $OUT_DIR
