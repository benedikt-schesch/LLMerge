#!/usr/bin/env bash

set -euo pipefail

# ─── 1. Confirm overwrite if output exists ────────────────────────────────────
OUTPUT_FILE="tables/results_table.tex"
mkdir -p "$(dirname "$OUTPUT_FILE")"
if [[ -f "$OUTPUT_FILE" ]]; then
    read -p "$OUTPUT_FILE already exists. Overwrite? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# ─── 2. Where to look for logs ────────────────────────────────────────────────
ROOT_DIR="eval_outputs/repos_reaper_test/test"

# ─── 3a. Whitelist of model prefixes to include ─────────────────────────────
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
    # "qwen/qwen3-8b"
    "qwen/qwen3-14b"
    "qwen/qwen3-32b"
    "deepseek/deepseek-r1-distill-qwen-1.5b"
    "deepseek/deepseek-r1-distill-llama-8b"
    "deepseek/deepseek-r1-distill-qwen-14b"
    "deepseek/deepseek-r1-distill-qwen-32b"
    "deepseek/deepseek-r1-distill-llama-70b"
    "api/deepseek-r1"
    "outputs/unsloth/DeepSeek-R1-Distill-Qwen-14B/checkpoint-2000/"
)


# ─── 3b. Run evaluation scripts in parallel ─────────────────────────────────
echo "🚀 Running evaluations for models: ${MODELS[*]}"
# ensure base directory exists
mkdir -p "$ROOT_DIR"
# use GNU parallel to run as many evals as there are processors
printf "%s\n" "${MODELS[@]}" | parallel -j "$(nproc)" --bar '
  model={};
  echo "🛠 Running eval for $model";
  if [[ "$model" == outputs/* ]]; then
    python3 eval.py --lora_weights "$model" > /dev/null 2>&1
  else
    python3 eval.py --model_name "$model" > /dev/null 2>&1
  fi
  echo "✅ Finished eval for $model";
'
echo "✅ All evaluations completed"

# ─── 4. Start the LaTeX table ────────────────────────────────────────────────
cat << 'EOF' > "$OUTPUT_FILE"
\begin{table}[ht]
\centering
\begin{tabular}{lrrrr}
\toprule
Model & Correct merges & Semantic merges & Raising conflict & Valid Java markdown \\
\midrule
EOF

echo "📝 Building LaTeX table from eval.log files in $ROOT_DIR"
MD_OUTPUT_FILE="tables/results_table.md"
mkdir -p "$(dirname "$MD_OUTPUT_FILE")"
echo "| Model | Correct merges | Semantic merges | Raising conflict | Valid Java markdown |" > "$MD_OUTPUT_FILE"
echo "| --- | ---: | ---: | ---: | ---: |" >> "$MD_OUTPUT_FILE"

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
        elif [[ "$model" == outputs/* ]]; then
            display_model="Ours"
        elif [[ "$name" == "qwq-32b" ]]; then
            display_model="QwQ 32B"
        elif [[ "$name" == gpt-* ]]; then
            rest="${name#gpt-}"
            display_model="GPT ${rest}"
        else
            display_model=$(echo "$name" | sed 's/-/ /g; s/\b\(.\)/\u\1/g')
            # Capitalize 'b' suffix for billions
            display_model=$(echo "$display_model" | sed -r 's/([0-9]+(\.[0-9]+)?)b/\1B/g')
        fi
        # print one row
        echo "${display_model} & ${correct}\\% & ${semantic}\\% & ${raise}\\% & ${valid}\\% \\\\" >> "$OUTPUT_FILE"
        echo "✅ Added $model to table"
        echo "| ${display_model} | ${correct}% | ${semantic}% | ${raise}% | ${valid}% |" >> "$MD_OUTPUT_FILE"
    else
        echo "⚠️ Logfile for model $model not found, skipping"
    fi
done

echo "📝 Finished processing all eval.log files"
# ─── 5. Close out the table ───────────────────────────────────────────────────
cat << 'EOF' >> "$OUTPUT_FILE"
\bottomrule
\end{tabular}
\caption{Merge-resolution performance across models.}
\end{table}
EOF

# ─── 6. Generate JPEG version of the table ───────────────────────────────────
JPEG_OUTPUT_FILE="$(dirname "$OUTPUT_FILE")/results_table.jpg"
TEX_WRAPPER="$(dirname "$OUTPUT_FILE")/results_table_wrapper.tex"
cat << LATEX > "$TEX_WRAPPER"
\documentclass{article}
\usepackage[margin=5mm]{geometry}
\usepackage{booktabs}
\usepackage{pdflscape}
\pagestyle{empty}
\begin{document}
\begin{landscape}
\input{$OUTPUT_FILE}
\end{landscape}
\end{document}
LATEX
echo "🖨 Generating PDF version of the table for JPEG conversion"
pdflatex -output-directory "$(dirname "$OUTPUT_FILE")" "$TEX_WRAPPER"
PDF_FILE="$(dirname "$OUTPUT_FILE")/results_table_wrapper.pdf"
convert -density 300 "$PDF_FILE" -quality 90 "$JPEG_OUTPUT_FILE"
echo "✅ JPG version written to $JPEG_OUTPUT_FILE"

# ─── 7. Cleanup temporary LaTeX files ─────────────────────────────────────────
echo "🧹 Cleaning up temporary LaTeX files"
rm -f "$(dirname "$OUTPUT_FILE")"/*.aux "$(dirname "$OUTPUT_FILE")"/*.log "$(dirname "$OUTPUT_FILE")"/*.out
rm -f "$(dirname "$OUTPUT_FILE")"/results_table_wrapper.tex

mv "$(dirname "$OUTPUT_FILE")"/results_table_wrapper.pdf "$(dirname "$OUTPUT_FILE")"/results_table.pdf

echo "✅ Done! Table written to $OUTPUT_FILE"
echo "✅ Markdown table written to $MD_OUTPUT_FILE"
