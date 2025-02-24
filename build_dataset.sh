#!/bin/bash
# Script to collect all merge conflicts and their resolution

REPOS_DIR=$1
OUT_DIR=$2


# Skip if merges.csv already exists
python3 src/find_merges.py \
    --repos $REPOS_DIR \
    --output_file $OUT_DIR/merges.csv

python3 src/extract_conflict_files.py \
    --merges $OUT_DIR/merges.csv \
    --output_dir $OUT_DIR/file_conflicts

python3 src/extract_conflict_blocks.py \
    --conflict_dir $OUT_DIR/file_conflicts \
    --output_dir $OUT_DIR/conflict_blocks

python src/metrics_conflict_blocks.py \
    --input_dir $OUT_DIR/conflict_blocks \
    --csv_out $OUT_DIR/conflict_metrics.csv \
