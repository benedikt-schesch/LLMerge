#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for merge conflict resolution.

This script:
  1. Loads the model.
  2. Loads a dataset (the same one produced by the build_dataset.py script).
  3. Runs inference on each example (using the conversation prompt).
  4. Computes various accuracy metrics:
     - Percentage of samples that have correct Markdown formatting (```java ... ```)
     - Percentage of samples that have the correct code (expected solution appears in the answer)
     - Percentage of samples that have both correct code and correct markdown formatting
     - Percentage of samples that have correct merge conflict markers
     - Percentage of samples that have both correct merge conflict markers and markdown formatting
  5. Displays a progress bar with the current running metrics.
"""

import argparse
import re
import torch
from datasets import load_from_disk
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


def is_valid_conflict(extracted_code):
    """Check if the extracted code contains valid merge conflict markers."""
    # Find the indexes of the main conflict markers.
    start_idx = extracted_code.find("<<<<<<<")
    mid_idx = extracted_code.find("=======")
    end_idx = extracted_code.find(">>>>>>>")

    # All markers must be present.
    if start_idx == -1 or mid_idx == -1 or end_idx == -1:
        return False

    # Markers must appear in the correct order.
    if not start_idx < mid_idx < end_idx:
        return False

    # If a base commit marker is present, it must be between start and divider.
    base_idx = extracted_code.find("|||||||")
    if base_idx != -1:
        if not start_idx < base_idx < mid_idx:
            return False

    return True


def evaluate_accuracy(model, tokenizer, dataset, device):  # pylint: disable=too-many-locals
    """
    For each sample in the dataset:
      - Uses the prompt (a conversation list) to generate an answer.
      - Computes five metrics:
          1. Markdown formatting correctness (does the answer
                include a code block wrapped in ```java ... ```?)
          2. Correct code (does the answer include the expected solution text?)
          3. Both correct code and markdown formatting.
          4. Presence of merge conflict markers (i.e. "<<<<<<<", "=======" and ">>>>>>>")
          5. Both merge conflict markers and markdown formatting.
    Returns a dictionary with the overall percentages.
    Also uses rich's Progress bar to show the current running metrics.
    """
    total = len(dataset)

    # Regex to detect a Java code block with markdown formatting.
    pattern = r"```java\n([\s\S]+?)```"

    # Counters for metrics.
    count_correct_markdown = 0
    count_correct_code = 0
    count_merge_conflict = 0

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "({task.completed}/{task.total})",
        TextColumn("Metrics: {task.fields[metrics]}"),
        TimeElapsedColumn(),
    ]

    with Progress(*progress_columns) as progress:
        task_id = progress.add_task(
            "Evaluating...",
            total=total,
            metrics="MD: 0.00%, Code: 0.00%, Both: 0.00%, Merge: 0.00%, Merge+MD: 0.00%",
        )

        for i, sample in enumerate(dataset, start=1):
            prompt = sample["prompt"]
            inputs = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, device=device, return_tensors="pt"
            )

            outputs = model.generate(inputs, max_length=100000)
            outputs_without_prompt = outputs[0][len(inputs[0]) :]
            response = tokenizer.decode(
                outputs_without_prompt, skip_special_tokens=False
            )

            # Only keep the response after the </think> token.
            response = response.split("</think>")[1]

            # Check for correct markdown formatting.
            markdown_match = re.search(pattern, response)
            if markdown_match:
                count_correct_markdown += 1
                extracted_code = markdown_match.group(1)
                if extracted_code is not None and extracted_code == sample["solution"]:
                    count_correct_code += 1
                elif is_valid_conflict(extracted_code):
                    count_merge_conflict += 1
            else:
                extracted_code = None

            # Running percentages for display.
            running_md = (count_correct_markdown / i) * 100
            running_code = (count_correct_code / i) * 100
            running_merge = (count_merge_conflict / i) * 100

            metrics_str = (
                f"Markdown: {running_md:.2f}%, Code: {running_code:.2f}%, "
                f"Conflict: {running_merge:.2f}%"
            )
            progress.update(task_id, advance=1, metrics=metrics_str)

    # Compute final percentages.
    results = {
        "markdown_formatting": (
            (count_correct_markdown / total) * 100 if total > 0 else 0.0
        ),
        "correct_code": (count_correct_code / total) * 100 if total > 0 else 0.0,
        "merge_conflict": (count_merge_conflict / total) * 100 if total > 0 else 0.0,
    }

    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evaluate model on merge conflict resolution accuracy."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="merges/repos_50/dataset",
        help="Path to the dataset directory produced by build_dataset.py",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    model, tokenizer = None, None  # TODO: Load model and tokenizer here.

    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = load_from_disk(args.dataset_path)

    # Use the 'test' split if available, else evaluate on the full dataset.
    eval_dataset = dataset["test"] if "test" in dataset else dataset

    print(f"Evaluating on {len(eval_dataset)} samples...")
    results = evaluate_accuracy(model, tokenizer, eval_dataset, device)

    # Display final percentages.
    print("\nFinal Metrics:")
    print(
        f"Percentage with correct Markdown formatting: {results['markdown_formatting']:.2f}%"
    )
    print(f"Percentage with correct code: {results['correct_code']:.2f}%")
    print(
        "Percentage with both correct code and markdown formatting: "
        f"{results['code_and_markdown']:.2f}%"
    )
    print(
        f"Percentage with correct merge conflict markers: {results['merge_conflict_markers']:.2f}%"
    )
    print(
        "Percentage with both merge conflict markers and markdown formatting: "
        f"{results['merge_conflict_and_markdown']:.2f}%"
    )


if __name__ == "__main__":
    main()
