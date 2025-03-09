# -*- coding: utf-8 -*-
"""Generate SFT dataset using DeepSeek R1 API for conflict resolution.

This script:
1. Loads a dataset with merge conflicts
2. Queries the DeepSeek R1 API to resolve each conflict
3. Caches API responses to avoid redundant calls
4. Creates a dataset of (conflict, resolution, correct/incorrect) tuples
5. Outputs both the full responses and a CSV summary
"""

import os
import json
import csv
import time
import argparse
import hashlib
from typing import Tuple, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from loguru import logger
from openai import OpenAI
from utils import (
    extract_code_block,
    normalize_java_code,
)

# Configure logger
logger.remove()
logger.add("deepseek_sft.log", level="INFO")
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

DEEPSEEK_API_URL = "https://api.deepseek.com"
CACHE_DIR = Path("deepseek_cache")
OUTPUT_DIR = Path("deepseek_sft")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "results.csv"


def get_cache_key(prompt: str) -> str:
    """Generate a unique cache key for a prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()


def load_from_cache(cache_key: str) -> Optional[Dict[str, str]]:
    """Load response from cache if it exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(cache_key: str, response: Dict[str, str]) -> None:
    """Save response to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(response, file, indent=4)


def query_deepseek_api(prompt: str) -> Dict[str, str]:
    """Query the DeepSeek R1 API for conflict resolution."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        raise ValueError("DEEPSEEK_API_KEY key not set")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_API_URL)

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],  # type: ignore
        stream=False,
    )

    reasoning = response.choices[0].message.reasoning_content  # type: ignore
    result = response.choices[0].message.content

    if reasoning is None or result is None:
        raise ValueError("Response is missing reasoning or content")

    return {"prompt": prompt, "reasoning": reasoning, "result": result}


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


def process_dataset(  # pylint: disable=too-many-locals
    dataset_path: Path, limit: Optional[int] = None
):
    """Process the dataset and generate SFT data."""
    # Load dataset
    dataset = load_from_disk(dataset_path)["train"]
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Limit the number of examples if specified
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))  # type: ignore
        logger.info(f"Limited to {len(dataset)} examples")

    results = []

    # Process each example
    example: Dict[str, str]
    for idx, example in enumerate(tqdm(dataset, desc="Processing conflicts")):  # type: ignore
        # Prepare prompt with query template
        prompt = example["question"]
        cache_key = get_cache_key(prompt)

        # Check cache first
        cached_response = load_from_cache(cache_key)

        if cached_response:
            logger.info(f"Using cached response for example {idx}")
            reponse = cached_response
        else:
            logger.info(f"Querying DeepSeek API for example {idx}")
            reponse = query_deepseek_api(prompt)

            if reponse:
                save_to_cache(cache_key, reponse)
            else:
                logger.error(f"Failed to get response for example {idx}")
                continue

            # Rate limiting - be nice to the API
            time.sleep(1)

        # Extract resolution from response
        try:
            resolution_text = reponse["result"]
            answer = example["answer"]

            # Save the full response
            output_file = OUTPUT_DIR / f"example_{idx}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(
                    f"PROMPT:\n{prompt}\n\nRESPONSE:\n{resolution_text}\n\nEXPECTED:\n{answer}"
                )

            # Evaluate if resolution is correct
            is_correct, match_type = evaluate_resolution(
                prompt, resolution_text, answer
            )

            # Add to results
            results.append(
                {
                    "example_id": idx,
                    "is_correct": is_correct,
                    "match_type": match_type,
                    "output_file": output_file.name,
                }
            )

            logger.info(
                f"Example {idx}: {'Correct' if is_correct else 'Incorrect'} ({match_type})"
            )

        except (KeyError, IndexError) as e:
            logger.error(f"Error processing response for example {idx}: {e}")

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
        default="merges/repos_reapar_1000/dataset",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit the number of examples to process",
    )
    args = parser.parse_args()

    process_dataset(args.dataset, args.limit)
