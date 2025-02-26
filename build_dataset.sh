#!/bin/bash
# Script to collect all merge conflicts and their resolution

REPOS_DIR=$1
OUT_DIR=$2

python3 src/get_conflict_files.py \
    --repos $REPOS_DIR \
    --output_dir $OUT_DIR

python3 src/extract_conflict_blocks.py \
    --input_dir $OUT_DIR/conflict_files \
    --output_dir $OUT_DIR/conflict_blocks

python3 src/metrics_conflict_blocks.py \
    --input_dir $OUT_DIR/conflict_blocks \
    --csv_out $OUT_DIR/conflict_metrics.csv \

python3 src/build_dataset.py \
    --conflict_blocks_dir $OUT_DIR/conflict_blocks \
    --output_dir $OUT_DIR/dataset
