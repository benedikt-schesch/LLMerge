# -*- coding: utf-8 -*-
"""Generate SFT dataset with exact matches only."""

import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from utils import extract_code_block, normalize_java_code, cached_query_deepseek_api
from variables import MAX_SEQUENCE_LENGTH, MODEL_NAME, SYSTEM_PROMPT

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def evaluate_resolution(resolution: str, expected_answer: str) -> str:
    """
    Return the evaluation result for the resolution.
    """
    code_block = extract_code_block(resolution)
    if code_block is None:
        return "No code block found"
    if code_block.strip() == expected_answer.strip():
        return "exact_match"
    if normalize_java_code(code_block) == normalize_java_code(expected_answer):
        return "semantic_match"
    return "incorrect_resolution"


def process_example(example: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Call the API for a single example and return the formatted text only if the resolution
    is an exact match.
    """
    prompt: str = example["question"]
    response: Optional[Dict[str, str]] = cached_query_deepseek_api(prompt)
    if response is None:
        return None
    resolution_text: str = response["result"]
    reasoning_text: str = response["reasoning"]
    answer: str = example["answer"]
    match_type = evaluate_resolution(resolution_text, answer)
    if match_type == "exact_match":
        resolution_full_text = (
            f"<think>\n{reasoning_text}</think>\n{resolution_text}\n\n"
        )
        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resolution_full_text},
            ]
        ]
        text = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in conversations
        ]
        return {"text": text[0]}
    return None


def process_dataset(
    dataset_path: Path,
    limit: Optional[int] = None,
    parallel_requests: int = 16,
    split: str = "test",
    save_dir: str = "my_sft_dataset",
) -> None:
    """
    Process the dataset, filter examples with exact match resolutions, build the final SFT dataset,
    and save it to disk.
    """
    ds = load_from_disk(dataset_path)[split]
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    results: List[Dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=parallel_requests
    ) as executor:
        futures = [executor.submit(process_example, example) for example in ds]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing examples",
        ):
            res = future.result()
            if res is not None:
                text = res["text"]
                # Filter out samples that exceed the max sequence length.
                tokens = tokenizer(text, truncation=False, add_special_tokens=False)
                if len(tokens["input_ids"]) <= MAX_SEQUENCE_LENGTH:
                    results.append(res)

    # Create and save the final dataset.
    final_sft_dataset = Dataset.from_list(results)
    final_sft_dataset.save_to_disk(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset with exact matches only."
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
        default=None,
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
        help="Dataset split to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="my_sft_dataset",
        help="Directory to save the final SFT dataset",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_path=Path(args.dataset),
        limit=args.limit,
        parallel_requests=args.parallel_requests,
        split=args.split,
        save_dir=args.output_dir,
    )
