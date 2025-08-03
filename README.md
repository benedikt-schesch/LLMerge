# LLMergeJ

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/benedikt-schesch/Merge-Bench-Builder/actions/workflows/ci.yml/badge.svg)](https://github.com/benedikt-schesch/LLMerge/actions/workflows/ci.yml)

A specialized language model fine-tuned for merge conflict resolution in Java code. LLMergeJ is trained to understand and resolve complex merge conflicts with high accuracy. 🤖

## Table of Contents

- [Features ✨](#features)
- [Prerequisites 📋](#prerequisites)
- [Installation ⚙️](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [License](#license)

## Features ✨

- 🎯 Specialized fine-tuning for Java merge conflict resolution
- 🚀 Built on DeepSeek-R1-Distill-Qwen-14B architecture
- 📊 Achieves 48.8% exact match accuracy on Java merge conflicts
- ⚡ Efficient training with Unsloth optimization
- 🔄 Support for supervised fine-tuning (SFT) with custom datasets
- 📈 Comprehensive evaluation metrics and performance tracking

## Prerequisites 📋

- [uv](https://docs.astral.sh/uv/) - Python package manager
- [Unsloth prerequisites](https://github.com/unslothai/unsloth#installation) - For efficient training and inference

## Installation ⚙️

1. Clone the repository:

   ```bash
   git clone https://github.com/benedikt-schesch/LLMerge.git
   cd LLMergeJ
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

## Usage

### Training a Model

Train LLMergeJ on merge conflict data:

```bash
# Basic training
python train.py
```

### Supervised Fine-Tuning (SFT)

Run supervised fine-tuning experiments:

```bash
# Single SFT run
python sft_train.py --model_name "unsloth/deepseek-r1-distill-qwen-14b-bnb-4bit" --dataset_path "merges/repos_reaper_java_test/dataset/"

# Batch SFT experiments with hyperparameter grid
./run_sft_experiments.sh
```

### Training Parameters

Key training parameters you can customize:

- `--model_name`: Base model to fine-tune (default: "unsloth/deepseek-r1-distill-qwen-14b-bnb-4bit")
- `--dataset_path`: Path to training dataset
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--per_device_train_batch_size`: Batch size per device (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)

## Training

### Training Scripts

- `train.py`: Main training script with configurable parameters
- `sft_train.py`: Supervised fine-tuning with evaluation
- `run_sft_experiments.sh`: Batch training with hyperparameter grid search

## Project Structure

```
.
├── train.py                    # Main training script
├── sft_train.py               # Supervised fine-tuning script
├── run_sft_experiments.sh     # Batch training experiments
├── src/                       # Core source code
│   ├── utils.py              # Utility functions
│   └── variables.py          # Configuration variables
├── checkpoints/              # Model checkpoints (created during training)
├── pyproject.toml           # Project configuration
├── README.md                # This file
└── LICENSE                  # MIT License
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
