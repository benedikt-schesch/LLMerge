# LLMerge

[![CI](https://github.com/benedikt-schesch/LLMerge/actions/workflows/ci.yml/badge.svg)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)]

A toolkit for training and evaluating Large Language Models to automatically resolve merge conflicts in code. 🤖

Evaluation results 🚀:

| Model | Correct merges | Semantic merges | Raising conflict | Valid Java markdown |
| --- | ---: | ---: | ---: | ---: |
| GPT 4.1 | 44.04% | 54.09% | 3.23% | 100.00% |
| Claude 3.7 Sonnet | 51.61% | 60.17% | 2.85% | 100.00% |
| Llama 4 Maverick | 26.18% | 32.63% | 31.76% | 99.75% |
| Llama 3.3 70B Instruct | 1.86% | 3.85% | 81.02% | 100.00% |
| Gemini 2.5 Pro Preview | 46.65% | 53.35% | 8.93% | 99.88% |
| Qwen3 235B A22B | 28.16% | 35.73% | 32.75% | 99.13% |
| Grok 3 Beta | 8.81% | 11.66% | 81.27% | 100.00% |
| QwQ 32B | 24.07% | 32.26% | 13.77% | 72.70% |
| o3 | 49.63% | 58.93% | 3.10% | 100.00% |
| Qwen3 14B | 12.90% | 16.63% | 69.48% | 99.88% |
| Qwen3 32B | 13.15% | 16.87% | 61.17% | 99.50% |
| Deepseek R1 Distill Qwen 1.5B | 0.00% | 0.12% | 0.00% | 77.42% |
| Deepseek R1 Distill Llama 8B | 3.35% | 7.57% | 14.76% | 94.17% |
| Deepseek R1 Distill Qwen 14B | 9.31% | 13.40% | 48.88% | 99.38% |
| Deepseek R1 Distill Qwen 32B | 22.83% | 30.40% | 30.65% | 99.01% |
| Deepseek R1 Distill Llama 70B | 25.81% | 33.00% | 29.40% | 98.88% |
| Deepseek R1 | 45.66% | 53.60% | 8.81% | 99.50% |
| Ours | 48.76% | 58.93% | 0.12% | 100.00% |
| Best SFT model |  17.99 % |  23.70 % |  42.56 % |  98.26 % |

## Table of Contents

- [Features ✨](#features)
- [Prerequisites 📋](#prerequisites)
- [Installation ⚙️](#installation)
- [Dataset Preparation 📊](#dataset-preparation)
- [Training 🚀](#training)
- [Evaluation 📊](#evaluation)
- [Project Structure](#project-structure)
- [License](#license)

## Features ✨

- 🤖 Train models to resolve merge conflicts using GRPO (Gradient Reward Policy Optimization)
- 📊 Comprehensive evaluation metrics for merge conflict resolution
- 🚀 Support for multiple LLMs including DeepSeek, Claude, GPT, and open-source models
- ⚡ Efficient training with LoRA and UnSloth optimization
- 📈 Detailed performance benchmarking and visualization

## Prerequisites 📋

- Python 3.8 or later
- Git
- CUDA-enabled GPU (optional, for training)
- Pre-built merge conflict datasets (see [Dataset Preparation](#dataset-preparation))

## Installation ⚙️

1. Clone the repository:

   ```bash
   git clone https://github.com/benedikt-schesch/LLMerge.git
   cd LLMerge
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install uv
   uv sync
   ```

> **Tip:** If you encounter CUDA issues, try:
> ```bash
> uv pip install -U transformers
> ```

## Dataset Preparation 📊

LLMerge requires pre-built merge conflict datasets for training and evaluation. These datasets should be created using [Merge-Bench-Builder](https://github.com/benedikt-schesch/Merge-Bench-Builder), which provides tools for extracting merge conflicts from Git repositories.

### Expected Dataset Structure

Datasets should be in HuggingFace format with the following structure:
```
merges/
└── dataset_name/
    └── dataset/
        ├── train/
        └── test/
```

### Creating Datasets

1. Clone and set up Merge-Bench-Builder:
   ```bash
   git clone https://github.com/benedikt-schesch/Merge-Bench-Builder.git
   cd Merge-Bench-Builder
   ```

2. Build your dataset (e.g., Java dataset with 1000 repositories):
   ```bash
   ./dataset_build_scripts/build_dataset_reaper_java_1000.sh -g -m -b
   ```

3. Copy or link the generated dataset to your LLMerge directory:
   ```bash
   cp -r merges/repos_reaper_1000 /path/to/LLMerge/merges/
   ```

### Formatting Existing Conflict Files

If you have existing `.conflict` and `.resolved_conflict` files, you can format them for training:

```bash
python src/build_dataset.py \
  --conflict_blocks_dir path/to/conflict/files \
  --output_dir merges/custom_dataset/dataset \
  --test_size 0.2
```

## Training 🚀

### GRPO Training

1. **Stage 1:** Initial training with higher learning rate

   ```bash
   python3 train.py --epochs 1500 --learning_rate 5e-5
   ```

2. **Stage 2:** Fine-tuning with lower learning rate

   ```bash
   python3 train.py --epochs 2000 --learning_rate 1e-5 --resume
   ```

### Supervised Fine-Tuning (SFT)

For supervised fine-tuning on specific datasets:

```bash
python3 sft_train.py --dataset_path merges/custom_dataset/dataset
```

## Evaluation 📊

### Evaluate a Single Model

```bash
python3 eval.py \
  --model_name unsloth/DeepSeek-R1-Distill-Qwen-14B \
  --dataset_path merges/repos_reaper_test/dataset \
  --split test
```

### Evaluate All Checkpoints

Run parallel evaluation of all saved checkpoints:

```bash
./src/scripts/eval_all_checkpoints.sh <n_processes>
```

### Generate Performance Tables

Build LaTeX tables with evaluation results:

```bash
./src/scripts/build_performance_table.sh
```

Results will be saved to `tables/results_table.tex`.

## Project Structure

```
.
├── src/
│   ├── build_dataset.py         # Format conflicts into training data
│   ├── model_inference.py       # Model inference utilities
│   ├── prepare_sft_dataset.py   # SFT data preparation
│   ├── deepseek_sft_data.py     # DeepSeek API integration
│   ├── utils.py                 # Utility functions
│   ├── variables.py             # Configuration variables
│   └── scripts/                 # Evaluation and analysis scripts
├── train.py                     # GRPO training script
├── sft_train.py                 # Supervised fine-tuning script
├── eval.py                      # Model evaluation script
├── plot_checkpoints.py          # Checkpoint visualization
├── tables/                      # Performance results
├── README.md
└── LICENSE
```

## Configuration

Key configuration variables in `src/variables.py`:
- `MODEL_NAME`: Base model for training
- `MAX_SEQUENCE_LENGTH`: Maximum token length
- `LORA_RANK`: LoRA rank for efficient fine-tuning
- `SYSTEM_PROMPT`: System prompt for the model
- `QUERY_PROMPT`: Prompt template for merge conflicts

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
