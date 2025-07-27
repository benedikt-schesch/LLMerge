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
LR=(1e-6 1e-5 1e-4)
WEIGHT_DECAY=(0 0.01)
SCHEDULER=("cosine" "linear")
EPOCHS=(1 3)
USE_GPUS=(0 1 2 3 4 6 7)

# Collect all model directories
declare -a model_dirs
for lr in "${LR[@]}"; do
  for wd in "${WEIGHT_DECAY[@]}"; do
    for sched in "${SCHEDULER[@]}"; do
      for epochs in "${EPOCHS[@]}"; do
        lr_fmt=$(case $lr in 1e-3) echo 0.001;; 1e-4) echo 0.0001;; 1e-5) echo 1e-05;; *) echo $lr;; esac)
        wd_fmt=$([[ "$wd" == "0" ]] && echo 0.0 || echo $wd)
        model_dirs+=("checkpoints/unsloth_Phi-4/direct_sft_lr${lr_fmt}_epochs${epochs}_wd${wd_fmt}_${sched}/final_model")
      done
    done
  done
done

# GPU counters
gpu_index=0; job_count=0
eval_gpu_index=0; eval_job_count=0
# Helpers
wait_for_gpu() { [[ $job_count -ge ${#USE_GPUS[@]} ]] && wait -n && ((job_count--)); }
wait_for_eval_gpu() { [[ $eval_job_count -ge $(( ${#USE_GPUS[@]} * 2 )) ]] && wait -n && ((eval_job_count--)); }

# ─── Training ───────────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == false ]]; then
  echo "Training ${#model_dirs[@]} configs on GPUs: ${USE_GPUS[*]}"
  for lr in "${LR[@]}"; do
    for wd in "${WEIGHT_DECAY[@]}"; do
      for sched in "${SCHEDULER[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
          lr_fmt=$(case $lr in 1e-3) echo 0.001;; 1e-4) echo 0.0001;; 1e-5) echo 1e-05;; *) echo $lr;; esac)
          wd_fmt=$([[ "$wd" == "0" ]] && echo 0.0 || echo $wd)
          model_dir="checkpoints/unsloth_Phi-4/direct_sft_lr${lr_fmt}_epochs${epochs}_wd${wd_fmt}_${sched}/final_model"

          [[ -d "$model_dir" ]] && { echo "Skipped existing: $model_dir"; continue; }

          gpu=${USE_GPUS[$gpu_index]}
          echo "Training config: lr=$lr, wd=$wd, sched=$sched, epochs=$epochs on GPU $gpu"

          # Run SFT with proper arguments
          # CUDA_VISIBLE_DEVICES=$gpu python3 sft_train.py \
          #   --dataset "merges/repos_reaper_java_train/dataset" \
          #   --run_name "direct_sft" \
          #   --model_name "unsloth/Phi-4" \
          #   --lr "$lr" --epochs "$epochs" --weight_decay "$wd" --lr_scheduler_type "$sched" &

          ((job_count++)); gpu_index=$(( (gpu_index+1)%${#USE_GPUS[@]} )); wait_for_gpu; sleep 1
        done
      done
    done
  done
  wait; echo "Training done"
else
  echo "Skipped training (--skip-training)"
fi

# ─── Evaluation ─────────────────────────────────────────────────────────────
echo "Evaluating on GPUs: ${USE_GPUS[*]}"
for dir in "${model_dirs[@]}"; do
  logfile="eval_outputs/repos_reaper_test/test/${dir}/eval.log"
  if [[ -f "$logfile" ]]; then
    echo "Log exists: $logfile"
  elif [[ -d "$dir" ]]; then
    gpu=${USE_GPUS[$eval_gpu_index]}
    echo "Evaluating $dir on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 eval.py --model_name "$dir" &
    ((eval_job_count++)); eval_gpu_index=$(( (eval_gpu_index+1)%${#USE_GPUS[@]} )); wait_for_eval_gpu; sleep 1
  else
    echo "Missing dir, skipping eval: $dir"
  fi
done
wait; echo "Evaluation done"

# ─── Table Generation ───────────────────────────────────────────────────────
ROOT="eval_outputs/repos_reaper_test/test"
TEX="tables/results_table_sft.tex"
MD="tables/results_table_sft.md"
mkdir -p "$(dirname "$TEX")" "$(dirname "$MD")"

# Determine best config by 'correct merges'
best=0; best_dir=""
for dir in "${model_dirs[@]}"; do
  lf="$ROOT/$dir/eval.log"; [[ -f "$lf" ]] || continue
  val=$(awk '/correctly resolved merges:/ {c=$NF; sub(/%/,"",c)} END {print c+0}' "$lf")
  (( $(echo "$val > $best" | bc -l) )) && best=$val && best_dir=$dir
done

# LaTeX header
cat << 'EOF' > "$TEX"
\begin{table}[ht]
\centering
\begin{tabular}{l l l l r r r r}
\toprule
Epochs & LR & Weight decay & Scheduler & Correct merges & Semantic merges & Raising conflict & Valid Java markdown \\
\midrule
EOF

# Markdown header
echo "| Epochs | LR | Weight decay | Scheduler | Correct merges | Semantic merges | Raising conflict | Valid Java markdown |" > "$MD"
echo "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" >> "$MD"

# Rows, bold best
for dir in "${model_dirs[@]}"; do
  lf="$ROOT/$dir/eval.log"; [[ -f "$lf" ]] || continue
  entry=$(basename "$(dirname "$dir")")
  lr_val=${entry#*lr}; lr_val=${lr_val%%_*}
  epochs_val=${entry#*epochs}; epochs_val=${epochs_val%%_*}
  wd_val=${entry#*wd}; wd_val=${wd_val%%_*}
  sched_val=${entry##*_}
  read valid raise semantic correct < <(
    awk '/valid Java markdown format:/ {v=$NF; sub(/%/,"",v)} /raising merge conflict:/ {r=$NF; sub(/%/,"",r)} /semantically correctly resolved merges:/ {s=$NF; sub(/%/,"",s)} /correctly resolved merges:/ {c=$NF; sub(/%/,"",c)} END {print v, r, s, c}' "$lf"
  )
  if [[ "$dir" == "$best_dir" ]]; then
    echo "${epochs_val} & ${lr_val} & ${wd_val} & ${sched_val} & \\textbf{${correct}}\\% & \\textbf{${semantic}}\\% & \\textbf{${raise}}\\% & \\textbf{${valid}}\\% \\\\" >> "$TEX"
    echo "| ${epochs_val} | ${lr_val} | ${wd_val} | ${sched_val} | **${correct}%** | **${semantic}%** | **${raise}%** | **${valid}%** |" >> "$MD"
  else
    echo "${epochs_val} & ${lr_val} & ${wd_val} & ${sched_val} & ${correct}\\% & ${semantic}\\% & ${raise}\\% & ${valid}\\% \\\\" >> "$TEX"
    echo "| ${epochs_val} | ${lr_val} | ${wd_val} | ${sched_val} | ${correct}% | ${semantic}% | ${raise}% | ${valid}% |" >> "$MD"
  fi
done

cat << 'EOF' >> "$TEX"
\bottomrule
\end{tabular}
\caption{Merge-resolution performance across configurations.}
\end{table}
EOF

# ─── Build PDF & PNG ─────────────────────────────────────────────────────────
JPEG="$(dirname "$TEX")/results_table_sft.jpg"
WRAPPER="$(dirname "$TEX")/results_table_sft_wrapper.tex"
cat << LATEX > "$WRAPPER"
\documentclass{article}
\usepackage[margin=5mm]{geometry}
\usepackage{booktabs}
\usepackage{pdflscape}
\pagestyle{empty}
\begin{document}
\begin{landscape}
\input{$TEX}
\end{landscape}
\end{document}
LATEX

echo "Generating PDF and JPEG from LaTeX"
pdflatex -output-directory "$(dirname "$TEX")" "$WRAPPER"
PDFWRAP="$(dirname "$TEX")/results_table_sft_wrapper.pdf"
convert -density 300 "$PDFWRAP" -quality 90 "$JPEG"

echo "Cleaning up auxiliary files"
rm -f "$(dirname "$TEX")"/*.aux "$(dirname "$TEX")"/*.log "$(dirname "$TEX")"/*.out
rm -f "$WRAPPER"
mv "$(dirname "$TEX")/results_table_sft_wrapper.pdf" "$(dirname "$TEX")/results_table_sft.pdf"

echo "✅ Done! Table: $TEX, Markdown: $MD, PDF: $(dirname "$TEX")/results_table_sft.pdf, JPEG: $JPEG"
