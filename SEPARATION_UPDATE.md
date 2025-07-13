# LLMerge Separation Update

## Changes Made

### 1. Removed Dataset Building Components
- **Removed `src/build_dataset.py`** - Now only exists in Merge-Bench-Builder
- **Kept `SYSTEM_PROMPT` and `QUERY_PROMPT` in `variables.py`** - Still needed for inference

### 2. Removed Evaluation Components
- **Removed `eval.py`** - Evaluation now handled by Merge-Bench
- **Removed `plot_checkpoints.py`** - Visualization of evaluation results
- **Removed `src/scripts/` directory** - Contains evaluation scripts
- **Removed `tables/` directory** - Evaluation results

### 3. Updated README.md
- Removed evaluation results table
- Removed evaluation section
- Updated description to specify "Java code" focus
- Updated features to remove evaluation mentions
- Updated project structure
- Added reference to Merge-Bench for evaluation

## Clean Architecture Achieved

### Three Separate Repositories:

**1. Merge-Bench-Builder**
- Extracts conflicts from Git repositories
- Formats conflicts into HuggingFace datasets
- Supports multiple programming languages
- Owns `build_dataset.py`, `SYSTEM_PROMPT`, `QUERY_PROMPT`

**2. LLMerge** (Training Only)
- GRPO training for Java merge conflicts
- SFT with knowledge distillation from DeepSeek R1
- Model inference utilities
- No dataset building, no evaluation

**3. Merge-Bench** (Evaluation Only)
- Multi-language evaluation framework
- Evaluates both API and local models
- Generates performance tables
- Handles all benchmarking

## Workflow

```
Git Repos → Merge-Bench-Builder → HuggingFace Dataset → LLMerge → Trained Models → Merge-Bench → Results
```

## Benefits of Separation

1. **Clear Responsibilities**: Each repo has a single, well-defined purpose
2. **Language Flexibility**: Merge-Bench can evaluate all languages while LLMerge focuses on Java training
3. **Reusability**: Datasets and evaluation can be used independently
4. **Maintainability**: Changes to one component don't affect others
