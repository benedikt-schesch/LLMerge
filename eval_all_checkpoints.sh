#!/bin/bash

set -e

# Default maximum number of parallel jobs (can be overridden by command line)
MAX_PARALLEL=${1:-1}

echo "Running with maximum of $MAX_PARALLEL parallel jobs"

count=0
running=0

# Process directories in descending order of checkpoint numbers
while IFS= read -r dir; do
  if [ -d "$dir" ]; then
    # Wait if we've reached the maximum number of parallel jobs
    if [ $running -ge $MAX_PARALLEL ]; then
      wait -n
      running=$((running - 1))
    fi

    echo "Running eval.py on $dir"
    python3 eval.py --lora_weights "$dir" &
    running=$((running + 1))
    count=$((count + 1))
  fi
done < <(ls -d outputs/checkpoint-* 2>/dev/null | sort -t- -k2 -n -r)

# Wait for all remaining background jobs
echo "Waiting for $running remaining jobs to complete..."
wait
echo "All $count evaluations completed."
