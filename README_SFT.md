# SFT Dataset Creation for LLMerge

This guide explains how to create a supervised fine-tuning (SFT) dataset for LLMerge using the DeepSeek R1 API. The process involves:

1. Generating examples using DeepSeek R1
2. Formatting the examples for SFT
3. Fine-tuning a model on the dataset

## Requirements

1. DeepSeek API key (set as environment variable `DEEPSEEK_API_KEY`)
2. Python packages listed in `requirements_dataset_building.txt`

## Step 1: Generate examples with DeepSeek R1 API

The script `src/deepseek_sft_data.py` queries the DeepSeek R1 API to resolve merge conflicts and evaluates the results.

```bash
# Set your API key
export DEEPSEEK_API_KEY="your_api_key_here"

# Run with default dataset (repos_50)
python src/deepseek_sft_data.py

# Use a specific dataset
python src/deepseek_sft_data.py --dataset merges/repos_reaper_100/dataset

# Limit the number of examples for testing
python src/deepseek_sft_data.py --limit 10
```

The script:
- Caches API responses to avoid redundant API calls
- Saves all examples to `outputs/deepseek_sft/example_*.txt`
- Generates a CSV summary at `outputs/deepseek_sft/results.csv`
- Logs metrics on how many examples were correctly resolved

## Step 2: Prepare SFT dataset

The script `src/prepare_sft_dataset.py` formats the DeepSeek responses for SFT training:

```bash
# Create a dataset from all examples
python src/prepare_sft_dataset.py

# Create a dataset with only correctly resolved examples
python src/prepare_sft_dataset.py --correct_only
```

The script:
- Formats responses into the proper thinking/answer format
- Extracts thinking sections when possible or creates basic ones
- Creates train/validation splits
- Saves the dataset to `outputs/sft_dataset/full` or `outputs/sft_dataset/correct_only`

## Step 3: Fine-tune the model with SFT

The script `sft_train.py` performs supervised fine-tuning on the prepared dataset:

```bash
# Train on the dataset with correctly resolved examples
python sft_train.py

# Train with custom parameters
python sft_train.py --dataset outputs/sft_dataset/full --epochs 5 --batch_size 2
```

The script:
- Fine-tunes the model specified in `src/variables.py` using LoRA
- Tracks training progress with Weights & Biases
- Saves the trained model to `outputs/sft_model` by default

## Step 4: Continue with GRPO training

After SFT training, you can continue with GRPO training using the `train.py` script. You'll need to:

1. Modify `train.py` to load the SFT model instead of starting from scratch
2. Run the GRPO training to further improve the model

## Directory Structure

```
outputs/
├── deepseek_cache/       # Cached API responses
├── deepseek_sft/         # DeepSeek API results
│   ├── example_*.txt     # Individual example files
│   └── results.csv       # Summary of results
├── sft_dataset/          # Formatted datasets
│   ├── full/             # All examples
│   └── correct_only/     # Only correctly resolved examples
└── sft_model/            # Trained SFT model
```

## Logs

- `deepseek_sft.log`: Logs from the DeepSeek API script
- `prepare_sft.log`: Logs from the dataset preparation script

## Note on API Usage

The DeepSeek R1 API has rate limits and usage costs. The script implements:
- Caching to avoid redundant API calls
- A 1-second delay between calls to be respectful of the API service
