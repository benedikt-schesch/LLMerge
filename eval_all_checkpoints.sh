#!/bin/bash

# Loop through each directory in outputs/ that starts with checkpoint-
count=0
for dir in outputs/checkpoint-*; do
  if [ -d "$dir" ]; then
    echo "Running eval.py on $dir"
    python3 eval.py --lora_weights "$dir" &
    count=$((count + 1))
    if (( count % 2 == 0 )); then
      wait
    fi
  fi
done

# Wait for any remaining background jobs
wait
