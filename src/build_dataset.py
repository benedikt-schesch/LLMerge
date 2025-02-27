# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""
This script prepares a dataset for training a model on conflict resolution.
It loads conflict blocks from a directory, splits them into train/test sets,
and formats them as conversation examples.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

from datasets import Dataset, DatasetDict

from rich.progress import track
from loguru import logger

from variables import MAX_PROMPT_LENGTH, SYSTEM_PROMPT, QUERY_PROMPT

# Configure loguru to log to run.log
logger.add("run.log")


def build_query(conflict_query):
    """Builds a query from a conflict block."""
    return QUERY_PROMPT + "```java\n" + conflict_query + "\n```"


def load_conflict_dataset(  # pylint: disable=too-many-locals
    conflict_blocks_dir: str,
    metrics: pd.DataFrame,
    max_line_count: int = 20,
    keep_trivial_resolution: bool = True,
) -> Dataset:
    """
    Loads a dataset from a folder containing *.conflict and *.resolved_conflict files.
    Each example is a pair:
      - query: contents of the .conflict file
      - solution: contents of the .resolved_conflict file
    """
    conflict_dir = Path(conflict_blocks_dir)
    conflict_files = sorted(conflict_dir.glob("*.conflict"))

    queries = []
    solutions = []

    for conflict_file in track(
        conflict_files, description="Processing conflict files..."
    ):  # Wrapped in progress bar
        # The corresponding resolved file should have the same stem with .resolved_conflict extension.
        conflict_id = conflict_file.stem
        assert len(metrics[metrics["conflict_id"] == conflict_id]) == 1
        conflict_metrics = metrics[metrics["conflict_id"] == conflict_id].iloc[0]

        if (not keep_trivial_resolution) and conflict_metrics[
            "resolution_in_left_or_right"
        ]:
            logger.info(
                f"Skipping {conflict_file} because it has resolution in left or right."
            )
            continue

        resolved_file = conflict_file.with_name(conflict_id + ".resolved_conflict")
        if not resolved_file.exists():
            logger.info(
                f"Skipping {conflict_file} because corresponding resolved file not found."
            )
            continue

        conflict_query = conflict_file.read_text(encoding="utf-8")
        solution_text = resolved_file.read_text(encoding="utf-8")

        # Count number of lines in conflict query
        num_lines = len(conflict_query.split("\n"))
        if num_lines > max_line_count:
            logger.info(
                f"Skipping {conflict_file} because it has more than {max_line_count} lines."
            )
            continue
        query = build_query(conflict_query)
        if MAX_PROMPT_LENGTH < conflict_metrics["num_tokens_query"]:
            logger.info(
                f"Skipping {conflict_file} because it has more than {MAX_PROMPT_LENGTH} tokens."
            )
            continue
        queries.append(query)
        solutions.append(solution_text)

    if not queries:
        raise ValueError(
            "No valid conflict/solution pairs were found in the specified directory."
        )

    data = {"question": queries, "answer": solutions}
    dataset = Dataset.from_dict(data)
    return dataset


def make_conversation(
    example: Dict[str, str],
) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """
    Converts a single dataset example into a conversation format.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],
    }


def prepare_train_test_dataset(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    conflict_blocks_dir: str,
    metrics: pd.DataFrame,
    max_line_count: int = 20,
    test_size: float = 0.1,
    keep_trivial_resolution: bool = True,
    seed: int = 42,
):
    """
    Loads the conflict dataset, splits it into train and test splits,
    and applies the conversation formatting.
    """
    full_dataset = load_conflict_dataset(
        conflict_blocks_dir,
        max_line_count=max_line_count,
        metrics=metrics,
        keep_trivial_resolution=keep_trivial_resolution,
    )
    dataset_dict = full_dataset.train_test_split(test_size=test_size, seed=seed)

    # Map each split to the conversation format.
    dataset_dict["train"] = dataset_dict["train"].map(make_conversation)
    dataset_dict["test"] = dataset_dict["test"].map(make_conversation)

    return DatasetDict({"train": dataset_dict["train"], "test": dataset_dict["test"]})


def main():
    """Main function to prepare the dataset."""
    parser = argparse.ArgumentParser(
        description="Prepare train/test dataset from conflict blocks."
    )
    parser.add_argument(
        "--conflict_blocks_dir",
        type=str,
        default="merges/repos_50/conflict_blocks",
        help="Directory containing conflict blocks.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="merges/repos_50/conflict_metrics.csv",
        help="Directory to save the computed metrics.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset to use as test set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset split."
    )
    parser.add_argument(
        "--max_line_count",
        type=int,
        default=20,
        help="Maximum number of lines in a conflict block.",
    )
    parser.add_argument(
        "-keep_trivial_resolution",
        action="store_true",
        help="Filter out conflict blocks with trivial resolutions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="merges/repos_50/dataset",
        help="Directory to save the processed dataset.",
    )

    args = parser.parse_args()

    df_metrics = pd.read_csv(args.metrics)

    # Generate the dataset with a train/test split.
    dataset = prepare_train_test_dataset(
        args.conflict_blocks_dir,
        metrics=df_metrics,
        test_size=args.test_size,
        seed=args.seed,
        keep_trivial_resolution=args.keep_trivial_resolution,
        max_line_count=args.max_line_count,
    )
    logger.info(f"Train set size: {len(dataset['train'])}")
    logger.info(f"Test set size: {len(dataset['test'])}")

    # Create the output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the DatasetDict to disk.
    dataset.save_to_disk(args.output_dir)
    logger.info(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
