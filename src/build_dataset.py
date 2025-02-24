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

from datasets import Dataset, DatasetDict

# Define the system prompt used in conversation formatting.
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

QUERY_PROMPT = (
    "You are a semantic merge conflict resolution expert. Below is a snippet of code "
    "with surrounding context that includes a merge conflict.\n"
    "Return the entire snippet (including full context) in markdown code fences as provided, make sure you do not modify the context at all and preserve the spacing as is.\n"
    "Think in terms of intent and semantics that both sides of the merge are trying to achieve.\n"
    "If you are not sure on how to resolve the conflict or if the intent is ambiguous, please return the same snippet with the conflict.\n"
    "Here is the code snippet:\n"
)


def build_query(conflict_query):
    """Builds a query from a conflict block."""
    return QUERY_PROMPT + "```java\n" + conflict_query + "\n```"


# Now that we have split our training dataset, we need to validate our dataset (**Check if user/assistant conversation exist**) before moving to the next step.
def validate_dataset(dataset):
    """Perform basic validation checks on the dataset."""

    # Define the required fields for the dataset
    required_fields = ["problem", "prompt"]

    # Loop through the 'train' and 'test' splits of the dataset
    for split in ["train", "test"]:
        print(f"\nValidating {split} split:")

        # Retrieve column names from the dataset
        fields = dataset[split].column_names

        # Check if any required fields are missing
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"Warning: Missing fields: {missing}")  # Warn if fields are missing
        else:
            print("✓ All required fields present")  # Confirm all fields are present

        # Retrieve the first sample from the dataset split
        sample = dataset[split][0]

        # Extract the 'prompt' field, which contains a list of messages
        messages = sample["prompt"]

        # Validate the prompt format:
        # - It should contain at least two messages
        if (
            len(messages) >= 2
            and messages[0]["role"] == "system"
            and messages[1]["role"] == "user"
        ):
            print("✓ Prompt format is correct")  # Confirm correct format
        else:
            print("Warning: Incorrect prompt format")  # Warn if format is incorrect


def load_conflict_dataset(conflict_blocks_dir: str):
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

    for conflict_file in conflict_files:
        # The corresponding resolved file should have the same stem with .resolved_conflict extension.
        resolved_file = conflict_file.with_name(
            conflict_file.stem + ".resolved_conflict"
        )
        if not resolved_file.exists():
            print(
                f"Skipping {conflict_file} because corresponding resolved file not found."
            )
            continue

        conflict_query = conflict_file.read_text(encoding="utf-8")
        solution_text = resolved_file.read_text(encoding="utf-8")
        query = build_query(conflict_query)
        queries.append(query)
        solutions.append(solution_text)

    if not queries:
        raise ValueError(
            "No valid conflict/solution pairs were found in the specified directory."
        )

    data = {"problem": queries, "solution": solutions}
    dataset = Dataset.from_dict(data)
    return dataset


def make_conversation(example):
    """
    Converts a single dataset example into a conversation format.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def prepare_train_test_dataset(
    conflict_blocks_dir: str, test_size: float = 0.1, seed: int = 42
):
    """
    Loads the conflict dataset, splits it into train and test splits,
    and applies the conversation formatting.
    """
    full_dataset = load_conflict_dataset(conflict_blocks_dir)
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
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset to use as test set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset split."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="merges/repos_50/dataset",
        help="Directory to save the processed dataset.",
    )

    args = parser.parse_args()

    # Generate the dataset with a train/test split.
    dataset = prepare_train_test_dataset(
        args.conflict_blocks_dir, test_size=args.test_size, seed=args.seed
    )
    validate_dataset(dataset)
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")

    # Create the output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the DatasetDict to disk.
    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
