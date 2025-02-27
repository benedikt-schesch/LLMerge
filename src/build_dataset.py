# -*- coding: utf-8 -*-
"""Builds a dataset from the conflict blocks."""

import argparse
import os
from pathlib import Path
from typing import Dict, Union, List
from datasets import Dataset, DatasetDict
from rich.progress import track
from loguru import logger
from variables import SYSTEM_PROMPT, QUERY_PROMPT

logger.add("run.log")


def build_query(query: str) -> str:
    """Builds a query from the given conflict block."""
    return f"{QUERY_PROMPT}```java\n{query}\n```"


def load_conflict_dataset(directory: str) -> Dataset:
    """Loads the conflict dataset from the given directory."""
    conflict_files = sorted(Path(directory).glob("*.conflict"))
    queries, solutions = [], []

    for conflict_file in track(
        conflict_files, description="Processing conflict files..."
    ):
        resolved_file = conflict_file.with_name(
            conflict_file.stem + ".resolved_conflict"
        )
        queries.append(build_query(conflict_file.read_text(encoding="utf-8")))
        solutions.append(resolved_file.read_text(encoding="utf-8"))

    if not queries:
        raise ValueError("No valid conflict/solution pairs found.")
    return Dataset.from_dict({"question": queries, "answer": solutions})


def format_conversation(
    example: Dict[str, str],
) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """Formats the conversation for the chatbot model."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],
    }


def prepare_dataset(
    directory: str, test_size: float = 0.2, seed: int = 42
) -> DatasetDict:
    """Prepare the dataset for training and testing."""
    dataset = load_conflict_dataset(directory)
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    splits["train"] = splits["train"].map(format_conversation)
    splits["test"] = splits["test"].map(format_conversation)
    return DatasetDict({"train": splits["train"], "test": splits["test"]})


def main():
    """Main function for building the dataset."""
    parser = argparse.ArgumentParser(
        description="Prepare train/test dataset from conflict blocks."
    )
    parser.add_argument(
        "--conflict_blocks_dir", type=str, default="merges/repos_50/conflict_blocks"
    )
    parser.add_argument(
        "--output_dir", type=str, default="merges/repos_50/filtered_dataset"
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = prepare_dataset(args.conflict_blocks_dir, args.test_size, args.seed)
    logger.info(f"Train set size: {len(dataset['train'])}")
    logger.info(f"Test set size: {len(dataset['test'])}")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    logger.info(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
