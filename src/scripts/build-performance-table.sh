#!/usr/bin/env bash

set -euo pipefail

# ─── Confirm overwrite if output exists ────────────────────────────────────
OUTPUT_FILE="tables/results_table.tex"
mkdir -p "$(dirname "$OUTPUT_FILE")"
if [[ -f "$OUTPUT_FILE" ]]; then
    read -p "$OUTPUT_FILE already exists. Overwrite? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# ─── 1. Where to look for logs ────────────────────────────────────────────────
ROOT_DIR="eval_outputs/repos_reaper_test/test"

# ─── 2a. Whitelist of model prefixes to include ─────────────────────────────
# Only these models will be processed
MODELS=(
  "openai/gpt-4.1"
  "anthropic/claude-3.7-sonnet"
  "meta-llama/llama-4-maverick"
  "meta-llama/llama-3.3-70b-instruct"
  "google/gemini-2.5-pro-preview"
  "qwen/qwen3-235b-a22b"
  "x-ai/grok-3-beta"
  "qwen/qwq-32b"
  "o3"
)


# ─── 2b. Run evaluation scripts in parallel ─────────────────────────────────
echo "🚀 Running evaluations for models: ${MODELS[*]}"
# ensure base directory exists
mkdir -p "$ROOT_DIR"
 # use GNU parallel to run as many evals as there are processors
printf "%s\n" "${MODELS[@]}" | parallel -j "$(nproc)" --bar '
  model={};
  echo "🛠 Running eval for $model";
  python3 eval.py --model_name "$model"
  echo "✅ Finished eval for $model";
'
echo "✅ All evaluations completed"

# ─── 2. Start the LaTeX table ────────────────────────────────────────────────
cat << 'EOF' > "$OUTPUT_FILE"
\begin{table}[ht]
\centering
\begin{tabular}{lrrrr}
\toprule
Model & Correct merges & Semantic merges & Raising conflict & Valid Java markdown \\
\midrule
EOF

echo "📝 Building LaTeX table from eval.log files in $ROOT_DIR"

# Process only whitelisted models
for model in "${MODELS[@]}"; do
    logfile="$ROOT_DIR/$model/eval.log"
    if [[ -f "$logfile" ]]; then
        echo "⚙️ Processing $logfile"
        # extract the last-occurring values for each metric
        read valid raise semantic correct < <(
            awk '
                /Percentage with valid Java markdown format:/ { v = $NF; sub(/%$/,"",v) }
                /Percentage correctly raising merge conflict:/ { r = $NF; sub(/%$/,"",r) }
                /Percentage semantically correctly resolved merges:/ { s = $NF; sub(/%$/,"",s) }
                /Percentage correctly resolved merges:/ { c = $NF; sub(/%$/,"",c) }
                END { print v, r, s, c }
            ' "$logfile"
        )
        # Format model name: strip prefix, handle special cases for GPT and o3
        name="${model##*/}"
        if [[ "$name" == "o3" ]]; then
            display_model="$name"
        elif [[ "$name" == gpt-* ]]; then
            rest="${name#gpt-}"
            display_model="GPT ${rest}"
        else
            display_model=$(echo "$name" | sed 's/-/ /g; s/\b\(.\)/\u\1/g')
        fi
        # print one row
        echo "${display_model} & ${correct}\\% & ${semantic}\\% & ${raise}\\% & ${valid}\\% \\\\" >> "$OUTPUT_FILE"
        echo "✅ Added $model to table"
    else
        echo "⚠️ Logfile for model $model not found, skipping"
    fi
done

echo "📝 Finished processing all eval.log files"
# ─── 4. Close out the table ───────────────────────────────────────────────────
cat << 'EOF' >> "$OUTPUT_FILE"
\bottomrule
\end{tabular}
\caption{Merge-resolution performance across models.}
\end{table}
EOF

echo "✅ Done! Table written to $OUTPUT_FILE"
