# -*- coding: utf-8 -*-
"""
Prepare SFT dataset from DeepSeek API responses.

This script:
1. Loads the results from the DeepSeek API queries
2. Formats the data into a format suitable for supervised fine-tuning
3. Creates a dataset with proper thinking/answer format
4. Saves the dataset to disk for training
"""

import csv
import re
from pathlib import Path
import argparse
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, DatasetDict


# Configure logger
logger.remove()
logger.add("prepare_sft.log", level="INFO")
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

# Directories and files
DEEPSEEK_OUTPUT_DIR = Path("outputs/deepseek_sft")
RESULTS_FILE = DEEPSEEK_OUTPUT_DIR / "results.csv"
SFT_OUTPUT_DIR = Path("outputs/sft_dataset")

# Pattern to match code blocks in markdown
CODE_BLOCK_RE = re.compile(r"```java\n(.*?)\n```", re.DOTALL)


def setup_directories():
    """Create necessary directories."""
    SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load results from the CSV file."""
    if not RESULTS_FILE.exists():
        logger.error(f"Results file not found: {RESULTS_FILE}")
        return []

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_code_from_markdown(text):
    """Extract Java code from markdown code blocks."""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def format_for_sft(example_id, prompt, response, expected_answer, is_correct):
    """Format the data for supervised fine-tuning."""
    # Extract thinking part if it exists
    thinking = ""
    answer = response

    # Check if response contains thinking/reasoning
    if "<think>" in response.lower() and "</think>" in response.lower():
        parts = re.split(r"(?i)</think>", response, 1)
        if len(parts) == 2:
            thinking = parts[0].replace("<think>", "", 1).strip()
            answer = parts[1].strip()
    elif (
        "let's think about" in response.lower()
        or "first, let me analyze" in response.lower()
    ):
        # Try to extract implicit thinking sections
        lines = response.split("\n")
        thinking_lines = []
        answer_lines = []
        in_thinking_section = True

        for line in lines:
            if in_thinking_section and (
                line.strip().startswith("```java")
                or line.strip().startswith("Here's the resolved code:")
                or line.strip().startswith("The resolved code:")
            ):
                in_thinking_section = False

            if in_thinking_section:
                thinking_lines.append(line)
            else:
                answer_lines.append(line)

        thinking = "\n".join(thinking_lines).strip()
        answer = "\n".join(answer_lines).strip()

    # If no thinking was found, create a basic one
    if not thinking:
        thinking = "Let me analyze this merge conflict."

    # Format final output with proper <think> tags
    formatted_output = f"<think>\n{thinking}\n</think>\n{answer}"

    # Return formatted example for SFT dataset
    return {
        "example_id": example_id,
        "prompt": prompt,
        "completion": formatted_output,
        "expected_answer": expected_answer,
        "is_correct": is_correct == "True",  # Convert string to boolean
    }


def create_sft_dataset(filter_correct_only=False):  # pylint: disable=too-many-locals
    """Create and save the SFT dataset."""
    setup_directories()

    # Load results
    results = load_results()
    logger.info(f"Loaded {len(results)} examples from results file")

    if filter_correct_only:
        results = [r for r in results if r["is_correct"] == "True"]
        logger.info(f"Filtered to {len(results)} correct examples")

    formatted_examples = []

    # Process each example
    for result in tqdm(results, desc="Formatting examples"):
        example_id = result["example_id"]
        example_file = DEEPSEEK_OUTPUT_DIR / result["output_file"]

        if not example_file.exists():
            logger.warning(f"Example file not found: {example_file}")
            continue

        # Load example file
        with open(example_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract prompt, response, and expected answer
        parts = content.split("\n\n")
        if len(parts) < 3:
            logger.warning(f"Invalid format in example file: {example_file}")
            continue

        prompt = parts[0].replace("PROMPT:", "", 1).strip()
        response = parts[1].replace("RESPONSE:", "", 1).strip()
        expected_answer = parts[2].replace("EXPECTED:", "", 1).strip()

        # Format for SFT
        formatted_example = format_for_sft(
            example_id, prompt, response, expected_answer, result["is_correct"]
        )

        formatted_examples.append(formatted_example)

    # Create dataset
    dataset = Dataset.from_list(formatted_examples)

    # Split into train and validation sets (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict(
        {"train": dataset["train"], "validation": dataset["test"]}
    )

    # Save dataset
    output_path = SFT_OUTPUT_DIR / ("correct_only" if filter_correct_only else "full")
    dataset_dict.save_to_disk(output_path)

    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Train set size: {len(dataset_dict['train'])}")
    logger.info(f"Validation set size: {len(dataset_dict['validation'])}")

    return dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare SFT dataset from DeepSeek API responses"
    )
    parser.add_argument(
        "--correct_only",
        action="store_true",
        help="Only include correctly resolved examples in the dataset",
    )
    args = parser.parse_args()

    create_sft_dataset(filter_correct_only=args.correct_only)
