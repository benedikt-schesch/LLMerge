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
    "anthropic/claude-opus-4"
    "anthropic/claude-sonnet-4"
    "meta-llama/llama-4-maverick"
    "meta-llama/llama-3.3-70b-instruct"
    "google/gemini-2.5-pro-preview"
    "qwen/qwen3-235b-a22b"
    "x-ai/grok-3-beta"
    "qwen/qwq-32b"
    "o3"
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
  # Add model_name to the python script call
  if [[ "$model" == outputs/* ]]; then
    python3 eval.py --lora_weights "$model" --model_name "$model" > /dev/null 2>&1
  else
    python3 eval.py --model_name "$model" > /dev/null 2>&1
  fi
  echo "✅ Finished eval for $model";
'
echo "✅ All evaluations completed"


# ─── 3c. Concatenate all individual CSVs into one big file ──────────────────
COMBINED_CSV="tables/all_results_combined.csv"
echo "🤝 Concatenating all individual CSV results into $COMBINED_CSV..."

# Find all evaluation_results.csv files within the specified model directories
CSV_FILES=()
for model in "${MODELS[@]}"; do
    # Sanitize model name for path
    model_path="$model"
    if [[ "$model" == outputs/* ]]; then
        # For lora weights, the model name passed to eval.py is the same as the path
        model_path=$(basename "$(dirname "$model")")
    fi
    csv_file="$ROOT_DIR/$model/evaluation_results.csv"
    if [[ -f "$csv_file" ]]; then
        CSV_FILES+=("$csv_file")
    else
        echo "🤔 Note: Could not find CSV for model $model at $csv_file"
    fi
done

if [ ${#CSV_FILES[@]} -eq 0 ]; then
    echo "⚠️ No 'evaluation_results.csv' files were found to concatenate."
else
    # Use awk to efficiently merge the CSVs: print header from the first file,
    # then print the data rows from all other files.
    awk 'FNR==1 && NR!=1 {next} {print}' "${CSV_FILES[@]}" > "$COMBINED_CSV"
    echo "✅ Successfully created combined CSV at $COMBINED_CSV"
fi
echo # Blank line for readability


# ─── 4. Gather data before building tables ────────────────────────────────────
echo "📊 Gathering data from eval.log files in $ROOT_DIR"
# Use parallel arrays to store model data before printing
declare -a display_names
declare -a correct_scores
declare -a semantic_scores
declare -a raise_scores
declare -a invalid_scores

# Process only whitelisted models
for model in "${MODELS[@]}"; do
    logfile="$ROOT_DIR/$model/eval.log"
    if [[ -f "$logfile" ]]; then
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
        # Format model name
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
            display_model=$(echo "$display_model" | sed -r 's/([0-9]+(\.[0-9]+)?)b/\1B/g')
        fi

        # Store data in arrays, providing a default of 0 if a value is missing
        display_names+=("$display_model")
        correct_scores+=("${correct:-0}")
        semantic_scores+=("${semantic:-0}")
        raise_scores+=("${raise:-0}")
        # Calculate and store the "Invalid Java markdown" score
        invalid_score=$(echo "100 - ${valid:-0}" | bc)
        invalid_scores+=("${invalid_score}")
    else
        echo "⚠️ Logfile for model $model not found, skipping"
    fi
done

# Include best SFT model if available
SFT_MD="tables/results_table_sft.md"
if [[ -f "$SFT_MD" ]]; then
    echo "⚙️ Processing best SFT model from $SFT_MD"
    best_line=$(tail -n +3 "$SFT_MD" | sort -t '|' -k5 -nr | head -n1)
    if [[ -n "$best_line" ]]; then
      IFS='|' read -r _ epochs lr decay sched correct semantic raise valid _ <<< "$best_line"
      correct=$(echo "${correct//%/}" | xargs)
      semantic=$(echo "${semantic//%/}" | xargs)
      raise=$(echo "${raise//%/}" | xargs)
      valid=$(echo "${valid//%/}" | xargs)

      display_names+=("Best SFT model")
      correct_scores+=("${correct:-0}")
      semantic_scores+=("${semantic:-0}")
      raise_scores+=("${raise:-0}")
      # Calculate and store the "Invalid Java markdown" score for SFT
      invalid_score=$(echo "100 - ${valid:-0}" | bc)
      invalid_scores+=("${invalid_score}")
      echo "✅ Added Best SFT model data"
    fi
fi

# ─── 5. Find top scores per column and write tables ───────────────────────────

# Helper function to find the top three unique scores from an array of numbers
# Takes a sort flag (e.g., "-nr" for descending, "-n" for ascending) as the first argument
get_ranked_scores() {
    local sort_flag="$1"
    shift
    local -a scores=("$@")
    local -a unique_sorted=()
    if [ ${#scores[@]} -gt 0 ]; then
        unique_sorted=($(printf "%s\n" "${scores[@]}" | sort "$sort_flag" | uniq))
    fi
    # Output space-separated list of top 3 scores. Default to -1 (an unmatchable value).
    echo "${unique_sorted[0]:--1} ${unique_sorted[1]:--1} ${unique_sorted[2]:--1}"
}


# Helper function to format a table cell based on its rank
format_cell() {
    local value="$1"
    local s1="$2" local s2="$3" local s3="$4"
    local format_type="$5" # "latex" or "md"

    if [[ "$format_type" == "latex" ]]; then
        local suffix="\\%"
        if (( $(echo "$value == $s1" | bc -l) && $(echo "$s1 != -1" | bc -l) )); then
            echo "\\textcolor{ForestGreen}{${value}${suffix}}" # 1st place
        elif (( $(echo "$value == $s2" | bc -l) && $(echo "$s2 != -1" | bc -l) )); then
            echo "\\textcolor{RoyalBlue}{${value}${suffix}}" # 2nd place
        elif (( $(echo "$value == $s3" | bc -l) && $(echo "$s3 != -1" | bc -l) )); then
            echo "\\textcolor{BurntOrange}{${value}${suffix}}" # 3rd place
        else
            echo "${value}${suffix}"
        fi
    else # Markdown formatting for GitHub
        local output="${value}%"
        if (( $(echo "$value == $s1" | bc -l) && $(echo "$s1 != -1" | bc -l) )); then
            output+=" 🥇" # 1st place
        elif (( $(echo "$value == $s2" | bc -l) && $(echo "$s2 != -1" | bc -l) )); then
            output+=" 🥈" # 2nd place
        elif (( $(echo "$value == $s3" | bc -l) && $(echo "$s3 != -1" | bc -l) )); then
            output+=" 🥉" # 3rd place
        fi
        echo "$output"
    fi
}

# Find top three scores for EACH column independently
echo "🏆 Identifying top scores for each column..."
# For these columns, higher is better, so we sort descending (-nr)
read -r correct_1st correct_2nd correct_3rd < <(get_ranked_scores "-nr" "${correct_scores[@]}")
read -r semantic_1st semantic_2nd semantic_3rd < <(get_ranked_scores "-nr" "${semantic_scores[@]}")
read -r raise_1st raise_2nd raise_3rd < <(get_ranked_scores "-nr" "${raise_scores[@]}")
# For invalid markdown, lower is better, so we sort ascending (-n)
read -r invalid_1st invalid_2nd invalid_3rd < <(get_ranked_scores "-n" "${invalid_scores[@]}")


# Write LaTeX table header
cat << 'EOF' > "$OUTPUT_FILE"
\begin{table*}[ht]
\centering
\begin{tabular}{lrrrr}
\toprule
Model & Equivalent to developer & Code normalized equivalent to developer & Raises a conflict & Invalid Java markdown \\
\midrule
EOF

# Write Markdown table header
MD_OUTPUT_FILE="tables/results_table.md"
mkdir -p "$(dirname "$MD_OUTPUT_FILE")"
echo "| Model | Equivalent to developer | Code normalized equivalent to developer | Raises a conflict | Invalid Java markdown |" > "$MD_OUTPUT_FILE"
echo "| --- | ---: | ---: | ---: | ---: |" >> "$MD_OUTPUT_FILE"

# Iterate through collected data and write table rows with per-cell formatting
echo "📝 Building LaTeX and Markdown tables with per-column rankings..."
for i in "${!display_names[@]}"; do
    display_model="${display_names[i]}"

    # Format each cell for LaTeX output
    correct_tex=$(format_cell "${correct_scores[i]}" "$correct_1st" "$correct_2nd" "$correct_3rd" "latex")
    semantic_tex=$(format_cell "${semantic_scores[i]}" "$semantic_1st" "$semantic_2nd" "$semantic_3rd" "latex")
    raise_tex=$(format_cell "${raise_scores[i]}" "$raise_1st" "$raise_2nd" "$raise_3rd" "latex")
    invalid_tex=$(format_cell "${invalid_scores[i]}" "$invalid_1st" "$invalid_2nd" "$invalid_3rd" "latex")

    # Format each cell for Markdown output
    correct_md=$(format_cell "${correct_scores[i]}" "$correct_1st" "$correct_2nd" "$correct_3rd" "md")
    semantic_md=$(format_cell "${semantic_scores[i]}" "$semantic_1st" "$semantic_2nd" "$semantic_3rd" "md")
    raise_md=$(format_cell "${raise_scores[i]}" "$raise_1st" "$raise_2nd" "$raise_3rd" "md")
    invalid_md=$(format_cell "${invalid_scores[i]}" "$invalid_1st" "$invalid_2nd" "$invalid_3rd" "md")

    # Write the formatted rows to the output files
    echo "${display_model} & ${correct_tex} & ${semantic_tex} & ${raise_tex} & ${invalid_tex} \\\\" >> "$OUTPUT_FILE"
    echo "| ${display_model} | ${correct_md} | ${semantic_md} | ${raise_md} | ${invalid_md} |" >> "$MD_OUTPUT_FILE"
done
echo "✅ Finished building tables."

# ─── 6. Close out the table ───────────────────────────────────────────────────
cat << 'EOF' >> "$OUTPUT_FILE"
\bottomrule
\end{tabular}
\caption{Merge-resolution performance across models. Top three results in each column are highlighted by color: \textcolor{ForestGreen}{1st place}, \textcolor{RoyalBlue}{2nd place}, and \textcolor{BurntOrange}{3rd place}. For the 'Invalid Java markdown' column, lower scores are better.}
\end{table*}
EOF

# ─── 7. Generate JPEG version of the table ───────────────────────────────────
JPEG_OUTPUT_FILE="$(dirname "$OUTPUT_FILE")/results_table.jpg"
TEX_WRAPPER="$(dirname "$OUTPUT_FILE")/results_table_wrapper.tex"
# Added [dvipsnames]{xcolor} for more color options (ForestGreen, etc.)
cat << LATEX > "$TEX_WRAPPER"
\documentclass{article}
\usepackage[margin=5mm]{geometry}
\usepackage{booktabs}
\usepackage{pdflscape}
\usepackage[dvipsnames]{xcolor}
\pagestyle{empty}
\begin{document}
\begin{landscape}
\input{$OUTPUT_FILE}
\end{landscape}
\end{document}
LATEX
echo "🖨 Generating PDF version of the table for JPEG conversion"

# Print the command before executing it
echo "  ↳ Executing: pdflatex -output-directory \"$(dirname "$OUTPUT_FILE")\" \"$TEX_WRAPPER\""

pdflatex -output-directory "$(dirname "$OUTPUT_FILE")" "$TEX_WRAPPER" > /dev/null
PDF_FILE="$(dirname "$OUTPUT_FILE")/results_table_wrapper.pdf"
convert -density 300 "$PDF_FILE" -quality 90 "$JPEG_OUTPUT_FILE"
echo "✅ JPG version written to $JPEG_OUTPUT_FILE"

# ─── 8. Cleanup temporary LaTeX files ─────────────────────────────────────────
echo "🧹 Cleaning up temporary LaTeX files"
rm -f "$(dirname "$OUTPUT_FILE")"/*.aux "$(dirname "$OUTPUT_FILE")"/*.log "$(dirname "$OUTPUT_FILE")"/*.out
rm -f "$(dirname "$OUTPUT_FILE")"/results_table_wrapper.tex

mv "$(dirname "$OUTPUT_FILE")"/results_table_wrapper.pdf "$(dirname "$OUTPUT_FILE")"/results_table.pdf

echo "✅ Done! Table written to $OUTPUT_FILE"
echo "✅ Markdown table written to $MD_OUTPUT_FILE"
