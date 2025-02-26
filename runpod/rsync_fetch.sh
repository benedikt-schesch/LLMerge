#!/bin/bash

# Excluded directories
EXCLUDES=(
  "unsloth_compiled_cache"
  ".mypy_cache"
  ".ruff_cache"
  "gil_free_venv"
  ".venv"
  ".workdir"
  "grpo_trainer_lora_model"
  "repos"
  "run.log"
  "uv.lock"
  "wandb"
)

# Default values
DEFAULT_SERVER="scheschb@godwit.cs.washington.edu"
DEFAULT_LOCAL_DIR="/Users/benediktschesch/Git/"
DEFAULT_REMOTE_DIR="/scratch/scheschb/LLMerge"

# Function to prompt user with default values
prompt_with_default() {
  local prompt_text="$1"
  local default_value="$2"
  read -p "$prompt_text [$default_value]: " input
  echo "${input:-$default_value}"
}

# Get user inputs with defaults
SERVER=$(prompt_with_default "Enter remote server and username" "$DEFAULT_SERVER")
LOCAL_DIR=$(prompt_with_default "Enter the local repository path" "$DEFAULT_LOCAL_DIR")
REMOTE_DIR=$(prompt_with_default "Enter the remote repository path" "$DEFAULT_REMOTE_DIR")

# Function to build the rsync exclude flags
build_exclude_flags() {
  local EXCLUDE_FLAGS=()
  for item in "${EXCLUDES[@]}"; do
    EXCLUDE_FLAGS+=(--exclude "$item")
  done
  echo "${EXCLUDE_FLAGS[@]}"
}

# Get exclude flags
EXCLUDE_FLAGS=$(build_exclude_flags)


echo "Syncing from remote ($SERVER:$REMOTE_DIR) to local ($LOCAL_DIR)..."
rsync -avz --delete $EXCLUDE_FLAGS "$SERVER:$REMOTE_DIR" "$LOCAL_DIR"
echo "Sync completed successfully."
