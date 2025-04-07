#!/bin/bash

# Loop through each directory in outputs/ that starts with checkpoint-
for dir in outputs/checkpoint-*; do
  if [ -d "$dir" ]; then
    echo "Running eval.py on $dir"
    python3 eval.py --model_name "$dir" --lora_weights "$dir"
  fi
done
