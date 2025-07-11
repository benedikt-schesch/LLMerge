#!/bin/bash

# Configuration arrays
LR=(1e-3 1e-4 1e-5)
WEIGHT_DECAY=(0 0.01)
SCHEDULER=("linear" "cosine")
EPOCHS=(1 3)
USE_GPUS=(4 5 6 7)

# Counter for GPU assignment
gpu_index=0
job_count=0

# Function to wait for background jobs if we've reached GPU limit
wait_for_gpu() {
    if [ $job_count -ge ${#USE_GPUS[@]} ]; then
        wait -n  # Wait for any background job to finish
        ((job_count--))
    fi
}

echo "Starting SFT training with all parameter combinations..."
echo "Total GPUs available: ${#USE_GPUS[@]}"

# Generate all combinations and run training
for lr in "${LR[@]}"; do
    for wd in "${WEIGHT_DECAY[@]}"; do
        for sched in "${SCHEDULER[@]}"; do
            for epochs in "${EPOCHS[@]}"; do
                # Get current GPU
                current_gpu=${USE_GPUS[$gpu_index]}

                echo "Starting training: LR=$lr, WD=$wd, Scheduler=$sched, Epochs=$epochs on GPU $current_gpu"

                # Run training in background on specific GPU
                CUDA_VISIBLE_DEVICES=$current_gpu python3 sft_train.py \
                    --lr $lr \
                    --weight_decay $wd \
                    --lr_scheduler_type $sched \
                    --epochs $epochs &

                # Update counters
                ((job_count++))
                ((gpu_index++))

                # Reset GPU index if we've used all GPUs
                if [ $gpu_index -ge ${#USE_GPUS[@]} ]; then
                    gpu_index=0
                fi

                # Wait if we've filled all GPUs
                wait_for_gpu

                # Small delay to avoid overwhelming the system
                sleep 2
            done
        done
    done
done

# Wait for all remaining jobs to complete
echo "Waiting for all training jobs to complete..."
wait

echo "All SFT training jobs completed!"
