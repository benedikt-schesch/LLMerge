#!/bin/bash
# Script to collect all merge conflicts and their resolution

REPOS_DIR=$1
OUT_DIR=$2
KEEP_FLAG=${3:-}  # Optional third argument, e.g. "-keep_trivial_resolution"

# Remove the current logs
rm run.log

./gradlew -q assemble -g ../.gradle/ --rerun-tasks

java -cp build/libs/findmerges-all.jar findmerges.FindMergeCommits $REPOS_DIR $OUT_DIR/merges

python3 src/get_conflict_files.py \
    --repos "$REPOS_DIR" \
    --merges "$OUT_DIR/merges" \
    --output_dir "$OUT_DIR"

python3 src/extract_conflict_blocks.py \
    --input_dir "$OUT_DIR/conflict_files" \
    --output_dir "$OUT_DIR/conflict_blocks"

python3 src/metrics_conflict_blocks.py \
    --input_dir "$OUT_DIR/conflict_blocks" \
    --filtered_output_dir "$OUT_DIR/filtered_merges" \
    --csv_out "$OUT_DIR/conflict_metrics.csv" $KEEP_FLAG

python3 src/build_dataset.py \
    --conflict_blocks_dir "$OUT_DIR/filtered_merges" \
    --output_dir "$OUT_DIR/dataset"
