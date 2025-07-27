# LLMerge

[![CI](https://github.com/benedikt-schesch/LLMerge/actions/workflows/ci.yml/badge.svg)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)]

A toolkit for constructing and analyzing merge conflict datasets, and training models to automatically resolve merge conflicts in code. 🤖

## Evaluation Results 🚀

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
| **Ours** | **48.76%** | **58.93%** | **0.12%** | **100.00%** |

## Table of Contents

- [Features ✨](#features)
- [Prerequisites 📋](#prerequisites)
- [Installation ⚙️](#installation)
- [Usage](#usage)
  - [Dataset Construction 🗂️](#dataset-construction)
  - [Training 🚀](#training)
  - [Evaluation 📊](#evaluation)
- [Advanced Training Methods](#advanced-training-methods)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [GRPO Training](#grpo-training)
- [Project Structure](#project-structure)
- [License](#license)

## Features ✨

- 🛠️ Build customizable merge conflict datasets from Git history
- 📊 Compute conflict metrics and analyze resolution strategies
- 🤖 Train and evaluate models to resolve merge conflicts in Java code
- ⚙️ Support for multiple training approaches: GRPO, SFT, and distillation
- 🔄 API integration for DeepSeek R1 and OpenRouter models
- 📈 Comprehensive evaluation framework with multiple metrics

## Prerequisites 📋

- Python 3.8 or later
- Git
- CUDA-enabled GPU (optional, for training)
- API keys for DeepSeek or OpenRouter (optional, for API-based models)

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

## Usage

### Dataset Construction 🗂️

#### Small Test Run

```bash
./dataset_build_scripts/build_dataset_small.sh -g -m -b
```

#### Full Dataset (e.g., 1000 merges)

```bash
./dataset_build_scripts/build_dataset_reaper_1000.sh -g -m -b
```

#### Test Dataset

```bash
./dataset_build_scripts/build_dataset_reaper_test.sh -g -m -b
```

All scripts support:
- `-g`: Run get & extract steps
- `-m`: Compute metrics
- `-b`: Build final dataset
- `--test_size <fraction>`: Fraction reserved for testing (default: 0.2)
- `--max_num_merges <n>`: Max merges to include (default: 100)

### Training 🚀

#### GRPO Training (Default)

1. **Stage 1:** 1500 epochs, learning rate = 5e-5

   ```bash
   python3 train.py --epochs 1500 --learning_rate 5e-5
   ```

2. **Stage 2:** Resume training for 2000 epochs, learning rate = 1e-5

   ```bash
   python3 train.py --epochs 2000 --learning_rate 1e-5 --resume
   ```

#### Supervised Fine-Tuning (SFT)

Run experiments with different hyperparameters:

```bash
# Run all SFT experiments
./run_sft_experiments.sh

# Skip training and only evaluate existing models
./run_sft_experiments.sh --skip-training
```

### Evaluation 📊

#### Evaluate a single model

```bash
python3 eval.py --model_name "model_name_or_path" --dataset_path "merges/repos_reaper_test/dataset"
```

#### Evaluate API models

```bash
# DeepSeek R1
python3 eval.py --model_name "api/deepseek-r1"

# OpenRouter models
python3 eval.py --model_name "anthropic/claude-3.7-sonnet"
python3 eval.py --model_name "openai/gpt-4.1"
```

#### Build performance tables

```bash
./src/scripts/build_performance_table.sh
```

Results will be saved to:
- `tables/results_table.tex` (LaTeX format)
- `tables/results_table.md` (Markdown format)
- `tables/results_table.pdf` (PDF visualization)
- `tables/results_table.jpg` (Image format)

## Advanced Training Methods

### Supervised Fine-Tuning (SFT)

Create SFT datasets using DeepSeek R1 API:

1. **Generate examples:**
   ```bash
   export DEEPSEEK_API_KEY="your_api_key_here"
   python src/deepseek_sft_data.py --dataset merges/repos_reaper_100/dataset
   ```

2. **Prepare dataset:**
   ```bash
   python src/prepare_sft_dataset.py --correct_only
   ```

3. **Fine-tune:**
   ```bash
   python sft_train.py --dataset outputs/sft_dataset/correct_only --epochs 5
   ```

See [README_SFT.md](README_SFT.md) for detailed instructions.

### GRPO Training

GRPO (Gradient Reward Policy Optimization) is the default training method that uses reward signals to improve model performance on merge conflict resolution.

## Project Structure

```
.
├── dataset_build_scripts/     # Scripts for building datasets
│   ├── build_dataset_small.sh
│   ├── build_dataset_reaper_1000.sh
│   └── build_dataset_reaper_test.sh
├── src/                       # Source code
│   ├── build_dataset.py       # Dataset construction
│   ├── deepseek_sft_data.py   # DeepSeek API integration
│   ├── extract_conflict_blocks.py
│   ├── find_merges.py
│   ├── get_conflict_files.py
│   ├── metrics_conflict_blocks.py
│   ├── model_inference.py
│   ├── prepare_sft_dataset.py
│   ├── utils.py
│   ├── variables.py           # Configuration variables
│   └── scripts/               # Utility scripts
├── tables/                    # Generated result tables
├── train.py                   # GRPO training script
├── sft_train.py              # SFT training script
├── eval.py                   # Evaluation script
├── run_sft_experiments.sh    # SFT experiment runner
├── README.md
├── README_SFT.md             # SFT-specific documentation
└── LICENSE
```

## Key Metrics Explained

- **Correct merges**: Percentage of merges that exactly match the developer's resolution
- **Semantic merges**: Percentage of merges that are semantically equivalent (ignoring whitespace/formatting)
- **Raising conflict**: Percentage where the model correctly identifies unresolvable conflicts
- **Valid Java markdown**: Percentage of responses with properly formatted Java code blocks

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
