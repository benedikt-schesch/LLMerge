# LLMerge

![CI](https://github.com/benedikt-schesch/LLMerge/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)

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

- Python 3.12 or later
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

## Dataset Preparation 📊

### Dataset Construction 🗂️

#### Small Test Run

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
   cp -r merges/repos_reaper_java_train /path/to/LLMerge/merges/
   ```

### Dataset Format

LLMerge expects datasets to be in HuggingFace format with the conversation already formatted. The datasets should include:
- System prompts
- User queries with merge conflicts
- Expected resolutions

Datasets created by Merge-Bench-Builder will already be in the correct format.

### Training 🚀

#### GRPO Training (Default)

LLMerge supports three distinct training approaches for comprehensive comparison:

### 1. GRPO Training (Reinforcement Learning)

GRPO uses reward functions to train models with thinking-based reasoning:

```bash
python3 train.py --epochs 1500 --learning_rate 5e-5
```

### 2. Direct SFT (Imitation Learning)

Direct supervised fine-tuning on human-resolved conflicts without reasoning:

```bash
# For DeepSeek model with thinking (original approach)
python3 sft_train.py --dataset merges/repos_reaper_java_train/dataset

# For Qwen3-14B without thinking (new approach)
./run_direct_sft_experiments.sh
```

See [README_DIRECT_SFT.md](README_DIRECT_SFT.md) for details on the Qwen3 direct SFT approach.

### 3. Knowledge Distillation (API-based)

Train on outputs from DeepSeek R1 API (requires separate data preparation):

```bash
# First prepare distillation dataset
python3 src/deepseek_sft_data.py --dataset_path merges/repos_reaper_java_train/dataset

# Then train on distilled data
python3 sft_train.py --dataset merges/repos_reaper_java_train/dataset_sft --add_system_prompt
```

## Evaluation

Model evaluation is handled by the [Merge-Bench](https://github.com/benedikt-schesch/Merge-Bench) repository, which provides comprehensive evaluation across multiple programming languages including Java.

## Project Structure

```
.
├── src/
│   ├── model_inference.py       # Model inference utilities
│   ├── prepare_sft_dataset.py   # SFT data preparation
│   ├── deepseek_sft_data.py     # DeepSeek API integration
│   ├── utils.py                 # Utility functions
│   └── variables.py             # Configuration variables
├── train.py                     # GRPO training script
├── sft_train.py                 # Supervised fine-tuning script
├── README.md
├── README_SFT.md             # SFT-specific documentation
└── LICENSE
```

## Configuration

Key configuration variables in `src/variables.py`:
- `MAX_SEQUENCE_LENGTH`: Maximum token length
- `LORA_RANK`: LoRA rank for efficient fine-tuning
- `SYSTEM_PROMPT`: System prompt for the model
- `QUERY_PROMPT`: Prompt template for merge conflicts

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
