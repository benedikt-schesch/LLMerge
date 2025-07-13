#!/bin/bash

# Parse flags
SKIP_TRAINING=false
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --skip-training)
      SKIP_TRAINING=true; shift;;
    *)
      echo "Unknown flag: $1"; exit 1;;
  esac
done

# Configuration arrays
LR=(1e-3 1e-4 1e-5)
WEIGHT_DECAY=(0 0.01)
SCHEDULER=("linear" "cosine")
EPOCHS=(1 3)
USE_GPUS=(2 3 4 5 6)

# Collect all model directories (for checking if training is needed)
declare -a model_dirs
declare -a configs
for lr in "${LR[@]}"; do
  for wd in "${WEIGHT_DECAY[@]}"; do
    for sched in "${SCHEDULER[@]}"; do
      for epochs in "${EPOCHS[@]}"; do
        lr_fmt=$(case $lr in 1e-3) echo 0.001;; 1e-4) echo 0.0001;; 1e-5) echo 1e-05;; *) echo $lr;; esac)
        wd_fmt=$([[ "$wd" == "0" ]] && echo 0.0 || echo $wd)
        model_dir="checkpoints/unsloth/DeepSeek-R1-Distill-Qwen-14B/distill_model_lr${lr_fmt}_epochs${epochs}_wd${wd_fmt}_${sched}/final_model"
        model_dirs+=("$model_dir")
        configs+=("$lr,$wd,$sched,$epochs")
      done
    done
  done
done

# GPU counters
gpu_index=0; job_count=0
eval_gpu_index=0; eval_job_count=0
# Helpers
wait_for_gpu() { [[ $job_count -ge ${#USE_GPUS[@]} ]] && wait -n && ((job_count--)); }
wait_for_eval_gpu() { [[ $eval_job_count -ge $(( ${#USE_GPUS[@]} * 4 )) ]] && wait -n && ((eval_job_count--)); }

# ─── Training ───────────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == false ]]; then
  echo "Training ${#model_dirs[@]} configs on GPUs: ${USE_GPUS[*]}"
  for i in "${!model_dirs[@]}"; do
    dir="${model_dirs[$i]}"
    config="${configs[$i]}"
    IFS=',' read -r lr wd sched epochs <<< "$config"

    [[ -d "$dir" ]] && { echo "Skipped existing: $dir"; continue; }
    gpu=${USE_GPUS[$gpu_index]}
    echo "Training config: lr=$lr, wd=$wd, sched=$sched, epochs=$epochs on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 sft_train.py \
      --dataset "merges/repos_reaper_java_train/dataset_sft" \
      --run_name "distill_model" \
      --lr "$lr" --epochs "$epochs" --weight_decay "$wd" --lr_scheduler_type "$sched" \
      --add_system_prompt &
    ((job_count++)); gpu_index=$(( (gpu_index+1)%${#USE_GPUS[@]} )); wait_for_gpu; sleep 1
  done
  wait; echo "Training done"
else
  echo "Skipped training (--skip-training)"
fi
