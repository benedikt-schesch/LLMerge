#!/bin/bash
# Script to collect all merges

REPOS_DIR=$1
OUT_DIR=$2

python3 src/find_merges.py $REPOS_DIR $OUT_DIR/merges
