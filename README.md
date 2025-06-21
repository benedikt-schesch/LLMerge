# LLMerge

[![CI](https://github.com/benedikt-schesch/LLMerge/actions/workflows/ci.yml/badge.svg)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)]

A toolkit for constructing and analyzing merge conflict datasets, and training models to automatically resolve merge conflicts in code. 🤖

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
- [Usage](#usage)
- [Dataset Construction 🗂️](#dataset-construction)
- [Training 🚀](#training)
- [Evaluation 📊](#evaluation)
- [Project Structure](#project-structure)
- [License](#license)

## Features ✨

- 🛠️ Build customizable merge conflict datasets from Git history.
- 📊 Compute conflict metrics and analyze resolution strategies.
- 🤖 Train and evaluate models to resolve merge conflicts in Java code.
- ⚙️ Support full and test datasets with configurable size.

## Prerequisites

- Python 3.8 or later
- Git
- CUDA-enabled GPU (optional, for training)

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

### Small Test Run

```bash
./dataset_build_scripts/build_dataset_small.sh -g -m -b
```

### Full Dataset (e.g., 1000 merges)

```bash
./dataset_build_scripts/build_dataset_reaper_1000.sh -g -m -b
```

### Test Dataset

```bash
./dataset_build_scripts/build_dataset_reaper_test.sh -g -m -b
```

All scripts support:
- `-g`: Run get & extract steps
- `-m`: Compute metrics
- `-b`: Build final dataset
- `--test_size <fraction>`: Fraction reserved for testing (default: 0.2)
- `--max_num_merges <n>`: Max merges to include (default: 100)

## Training 🚀

1. **Stage 1:** 1500 epochs, learning rate = 5e-5

   ```bash
   python3 train.py --epochs 1500 --learning_rate 5e-5
   ```

2. **Stage 2:** Resume training for 2000 epochs, learning rate = 1e-5

   ```bash
   python3 train.py --epochs 2000 --learning_rate 1e-5 --resume
   ```

## Evaluation 📊

Evaluate all checkpoints in parallel:

```bash
./src/scripts/eval_all_checkpoints.sh <n_processes>
```

Build the performance table:

```bash
./src/scripts/build_performance_table.sh
```

Results will be saved to `tables/results_table.tex`.

## Project Structure

```
.
├── dataset_build_scripts/
│   ├── build_dataset_small.sh
│   ├── build_dataset_reaper_1000.sh
│   └── build_dataset_reaper_test.sh
├── src/
│   ├── get_conflict_files.py
│   ├── extract_conflict_blocks.py
│   ├── metrics_conflict_blocks.py
│   └── build_dataset.py
├── train.py
├── resolve_conflict.py
├── tables/
├── README.md
└── LICENSE
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
