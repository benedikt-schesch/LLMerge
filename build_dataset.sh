#!/bin/bash
# Usage: ./script.sh -g -m -b REPOS_DIR OUT_DIR [-keep_trivial_resolution]
#   -g : Run get_conflict and extract_conflict_blocks steps
#   -m : Run metrics_conflict_blocks step
#   -b : Run build_dataset step
#   -keep_trivial_resolution : Keep trivial resolutions in the dataset
#
# Example to run all steps:
#   ./script.sh -g -m -b /path/to/repos /path/to/output -keep_trivial_resolution

# Initialize flags
RUN_GET_EXTRACT=0
RUN_METRICS=0
RUN_BUILD=0
KEEP_FLAG=""
TEST_SIZE="0.2"

# Parse flags
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -g) RUN_GET_EXTRACT=1 ;;
        -m) RUN_METRICS=1 ;;
        -b) RUN_BUILD=1 ;;
        -keep_trivial_resolution) KEEP_FLAG="-keep_trivial_resolution" ;;
        --test_size)
            TEST_SIZE="$2"
            shift
            ;;
        --) shift; break ;; # Stop processing options
        -*)
            echo "Invalid option: $1" >&2
            exit 1
            ;;
        *)
            # Assume first non-flag argument is REPOS_DIR, second is OUT_DIR
            if [ -z "$REPOS_DIR" ]; then
                REPOS_DIR="$1"
            elif [ -z "$OUT_DIR" ]; then
                OUT_DIR="$1"
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
    shift
done

# Check if required arguments are provided
if [ -z "$REPOS_DIR" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: $0 -g -m -b REPOS_DIR OUT_DIR [-keep_trivial_resolution]" >&2
    exit 1
fi

# Execute steps based on flags
if [ $RUN_GET_EXTRACT -eq 1 ]; then
    rm -f run.log
    rm -rf .workdir

    if [ -d "$OUT_DIR/conflict_files" ]; then
        rm -r "$OUT_DIR/conflict_files"
    fi
    if [ -d "$OUT_DIR/conflict_blocks" ]; then
        rm -r "$OUT_DIR/conflict_blocks"
    fi

    echo "Running get_conflict_files.py..."
    python3 src/get_conflict_files.py --repos "$REPOS_DIR" --output_dir "$OUT_DIR"

    echo "Running extract_conflict_blocks.py..."
    python3 src/extract_conflict_blocks.py --input_dir "$OUT_DIR/conflict_files" --output_dir "$OUT_DIR/conflict_blocks"
fi

if [ $RUN_METRICS -eq 1 ]; then
    echo "Running metrics_conflict_blocks.py..."
    if [ -d "$OUT_DIR/filtered_dataset" ]; then
        rm -r "$OUT_DIR/filtered_dataset"
    fi
    python3 src/metrics_conflict_blocks.py \
        --input_dir "$OUT_DIR/conflict_blocks" \
        --filtered_output_dir "$OUT_DIR/filtered_dataset" \
        --csv_out "$OUT_DIR/conflict_metrics.csv" $KEEP_FLAG
fi

if [ $RUN_BUILD -eq 1 ]; then
    echo "Running build_dataset.py..."
    if [ -d "$OUT_DIR/dataset" ]; then
        rm -r "$OUT_DIR/dataset"
    fi
    rm -r "$OUT_DIR/dataset"
    python3 src/build_dataset.py \
        --conflict_blocks_dir "$OUT_DIR/filtered_dataset" \
        --output_dir "$OUT_DIR/dataset" \
        --test_size "$TEST_SIZE"
fi
