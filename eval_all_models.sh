#!/bin/bash

# This script evaluates a set of models using eval.py.
# It loops over each model specified in the list and runs evaluation with given parameters.

# List of models to evaluate
for model_name in \
    unsloth/deepseek-r1-distill-qwen-1.5b \
    unsloth/deepseek-r1-distill-qwen-7b \
    unsloth/deepseek-r1-distill-qwen-14b \
    unsloth/deepseek-r1-distill-qwen-32b \
    unsloth/QwQ-32B
do
    echo "Evaluating model: $model_name"
    python3 eval.py \
        --model_name "$model_name" \
        --dataset_path "merges/repos_reaper_test/dataset" \
        --output_dir "eval_outputs" \
        --split "test"
done
