# LLMerge

[![CI](https://github.com/benedikt-schesch/LLMerge/actions/workflows/ci.yml/badge.svg)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)]

A toolkit for training Large Language Models to automatically resolve merge conflicts in Java code. 🤖

## Table of Contents

- [Features ✨](#features)
- [Prerequisites 📋](#prerequisites)
- [Installation ⚙️](#installation)
- [Dataset Preparation 📊](#dataset-preparation)
- [Training 🚀](#training)
- [Project Structure](#project-structure)
- [License](#license)

## Features ✨

- 🤖 Train models to resolve merge conflicts using GRPO (Gradient Reward Policy Optimization)
- 🎯 Supervised Fine-Tuning (SFT) with knowledge distillation from DeepSeek R1
- 🚀 Support for multiple base models including DeepSeek-R1-Distill-Qwen variants
- ⚡ Efficient training with LoRA and UnSloth optimization
- 📊 Two training approaches: GRPO for exploration and SFT for imitation learning

## Prerequisites 📋

- Python 3.12 or later
- Git
- CUDA-enabled GPU
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

### Dataset Format

LLMerge expects datasets to be in HuggingFace format with the conversation already formatted. The datasets should include:
- System prompts
- User queries with merge conflicts
- Expected resolutions

Datasets created by Merge-Bench-Builder will already be in the correct format.

## Training 🚀

### GRPO Training

```bash
python3 train.py --epochs 1500 --learning_rate 5e-5
```

### Supervised Fine-Tuning (SFT)

For supervised fine-tuning on specific datasets:

```bash
python3 sft_train.py --dataset_path merges/custom_dataset/dataset
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
