# -*- coding: utf-8 -*-
"""Generate SFT dataset using DeepSeek R1 API for conflict resolution.

This script:
1. Loads a dataset with merge conflicts
2. Queries the DeepSeek R1 API to resolve each conflict
3. Caches API responses to avoid redundant calls
4. Creates a dataset of (conflict, resolution, correct/incorrect) tuples
5. Outputs both the full responses and a CSV summary
"""

import csv
import argparse
import concurrent.futures
from typing import Tuple, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from loguru import logger
from utils import (
    extract_code_block,
    normalize_java_code,
    cached_query_deepseek_api,
)

# Configure logger
logger.remove()
logger.add("deepseek_sft.log", level="INFO")
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

OUTPUT_DIR = Path("deepseek_sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "results.csv"


def evaluate_resolution(
    prompt: str, resolution: str, expected_answer: str
) -> Tuple[bool, str]:
    """Evaluate if the resolution is correct."""
    # Extract code block from the resolution
    code_block = extract_code_block(resolution)

    if code_block is None:
        return False, "No code block found"

    # Check for exact match
    if code_block == expected_answer:
        return True, "exact_match"

    # Check for semantic match (ignoring comments/whitespace)
    if normalize_java_code(code_block) == normalize_java_code(expected_answer):
        return True, "semantic_match"

    # Check if conflict markers are still present
    ground_truth_conflict_markers = extract_code_block(prompt)
    if code_block == ground_truth_conflict_markers:
        return False, "conflict_preserved"

    return False, "incorrect_resolution"


def process_example(
    idx: int, example: Dict[str, str]
) -> Optional[Dict[str, str | bool | int]]:
    """Process a single example and evaluate the resolution."""
    prompt = example["question"]
    response = cached_query_deepseek_api(prompt)
    if response is None:
        raise ValueError("Response is None")
    resolution_text = response["result"]
    answer = example["answer"]
    output_file = OUTPUT_DIR / f"example_{idx}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"PROMPT:\n{prompt}\n\nRESPONSE:\n{resolution_text}\n\nEXPECTED:\n{answer}"
        )
    try:
        is_correct, match_type = evaluate_resolution(prompt, resolution_text, answer)
    except Exception as e:
        logger.error(f"Error processing response for example {idx}: {e}")
        return None
    logger.info(
        f"Example {idx}: {'Correct' if is_correct else 'Incorrect'} ({match_type})"
    )
    return {
        "example_id": idx,
        "is_correct": is_correct,
        "match_type": match_type,
        "output_file": output_file.name,
    }


def process_dataset(  # pylint: disable=too-many-locals
    dataset_path: Path,
    limit: Optional[int] = None,
    parallel_requests: int = 16,
    split: str = "test",
):
    """Process the dataset and generate SFT data."""
    # Load dataset
    dataset = load_from_disk(dataset_path)[split]
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Limit the number of examples if specified
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))  # type: ignore
        logger.info(f"Limited to {len(dataset)} examples")

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=parallel_requests
    ) as executor:
        future_to_idx = {
            executor.submit(process_example, idx, example): idx  # type: ignore
            for idx, example in enumerate(dataset)
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Processing conflicts",
        ):
            result = future.result()
            if result is not None:
                results.append(result)

    # Write results to CSV
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["example_id", "is_correct", "match_type", "output_file"]
        )
        writer.writeheader()
        writer.writerows(results)

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    percentage = (correct / total) * 100 if total > 0 else 0

    logger.info(f"\nProcessed {total} examples")
    logger.info(f"Correctly resolved: {correct} ({percentage:.2f}%)")

    # Breakdown by match type
    match_types = {}
    for r in results:
        match_type = r["match_type"]
        match_types[match_type] = match_types.get(match_type, 0) + 1

    for match_type, count in match_types.items():
        logger.info(f"{match_type}: {count} ({(count / total) * 100:.2f}%)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset using DeepSeek R1 API"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="merges/repos_reaper_1000/dataset",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the number of examples to process",
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=32,
        help="Number of parallel requests to the DeepSeek API",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to process (train or test)",
    )
    args = parser.parse_args()

    process_dataset(args.dataset, args.limit, args.parallel_requests, args.split)
