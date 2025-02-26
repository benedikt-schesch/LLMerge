# LLMerge

![CI](https://github.com/benedikt-schesch/LLMerge/actions/workflows/ci.yml/badge.svg)

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

## Faster dataset construction with GIL-Free Python

The dataset building part relies on multithreading which is slowed down by the Python GIL.
To bypass the GIL use the latest GIL free Python version.
To install GIL free Python run:

```bash
uv venv gil_free_venv --python 3.12t
source gil_free_venv/bin/activate
uv pip install -r requirements_dataset_building.txt
```

With this environment you can *only* run the dataset creation scripts since the latest libraries don't properly support this version of python yet.
You can now run:

```bash
PYTHON_GIL=0 ./build_dataset_small.sh
```


## Usage

### Run small example

```bash
./build_dataset_small.sh
```

## Code Structure

```mermaid
graph TD
    get_conflict_files.py["extract_conflict_files.py<br/><span style='font-size:12px;'>Among all the merges extract all the conflicting files and their resolution</span>"] --> extract_conflict_blocks.py["extract_conflict_blocks.py<br/><span style='font-size:12px;'>Among all the conflicting files extract each conflict marker with necessary context and the resolution of each separately</span>"] --> metrics_conflict_blocks.py["metrics_conflict_blocks.py<br/><span style='font-size:12px;'>Computes different metrics for the dataset analysis and filtering</span>"]
```
