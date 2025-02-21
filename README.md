# LLMerge

## Installation

Install uv following this [guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) or simply use this quick command:

```bash
pip install uv
```

Install all dependencies and activate the venv with the following command:

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Run small example

```bash
./build_dataset_small.sh
```

## Code Structure

```mermaid
graph TD
    find_merges.py["find_merges.py<br/><span style='font-size:12px;'>Get all possible merges for all the given repos</span>"] --> extract_conflict_files.py["extract_conflict_files.py<br/><span style='font-size:12px;'>Among all the merges extract all the conflicting files and their resolution</span>"] --> extract_conflict_blocks.py["extract_conflict_blocks.py<br/><span style='font-size:12px;'>Among all the conflicting files extract each conflict marker with necessary context and the resolution of each separately</span>"]
```
