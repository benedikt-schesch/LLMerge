#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replays a specific merge conflict, showing the original prompt, the ground
truth solution, and the outputs from all evaluated models for that conflict.
"""

import argparse
from pathlib import Path
from datasets import load_from_disk

# ANSI color codes for clearer terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

MODELS = (
    "openai/gpt-4.1",
    "anthropic/claude-3.7-sonnet",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-pro-preview",
    "qwen/qwen3-235b-a22b",
    "x-ai/grok-3-beta",
    "qwen/qwq-32b",
    "o3",
    "qwen/qwen3-14b",
    "qwen/qwen3-32b",
    "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "api/deepseek-r1",
    "outputs/unsloth/DeepSeek-R1-Distill-Qwen-14B/checkpoint-2000/",
)


def replay(index: int, dataset_path: str, eval_outputs_root: str, split: str):
    """
    Main function to load data and print the replay.

    Args:
        index: The integer index of the merge conflict to replay.
        dataset_path: Path to the cached Hugging Face dataset.
        eval_outputs_root: Root directory of the evaluation outputs.
        split: The dataset split to use (e.g., 'test').
    """
    # --- 1. Load the Prompt and Ground Truth from the Dataset ---
    try:
        dataset = load_from_disk(dataset_path)[split]
        if not 0 <= index < len(dataset):
            print(
                f"{RED}Error: Index {index} is out of bounds "
                f"for the dataset (size: {len(dataset)}).{RESET}"
            )
            return
        example = dataset[index]
    except FileNotFoundError:
        print(
            f"{RED}Error: Dataset not found at '{dataset_path}'. Please check the path.{RESET}"
        )
        return
    except Exception as e:
        print(f"{RED}An error occurred while loading the dataset: {e}{RESET}")
        return

    prompt = example.get("question", "Prompt not found in dataset.")
    ground_truth = example.get("answer", "Ground truth not found in dataset.")

    print(f"{BOLD}{BLUE}=== Replaying Conflict Index: {index} ==={RESET}\n")
    print(f"{BOLD}{GREEN}>>> PROMPT (Original Input){RESET}")
    print(prompt)
    print(f"\n{BOLD}{GREEN}>>> GROUND TRUTH (Expected Result){RESET}")
    print(ground_truth)

    # --- 2. Discover Models and Replay Their Outputs ---
    root_path = Path(eval_outputs_root)

    print(f"\n{BOLD}{BLUE}=== Model Outputs ==={RESET}")
    for model in MODELS:
        # # For nested LoRA paths, get the more descriptive name
        # if model_path.parent.name != split and model_path.parent.parent.name != "test":
        #      model_name = f"{model_path.parent.name}/{model_path.name}"

        print(f"\n{BOLD}{YELLOW}--- OUTPUT FOR: {model} ---{RESET}")
        print(model)
        output_file = root_path / model / f"example_{index}.txt"

        try:
            full_completion = output_file.read_text(encoding="utf-8")

            # Use the same logic as eval.py to extract the actual completion
            if "<｜Assistant｜>" in full_completion:
                completion = full_completion.split("<｜Assistant｜>", 1)[1]
            elif "<|im_start|>assistant" in full_completion:
                completion = full_completion.split("<|im_start|>assistant", 1)[1]
            else:
                # Fallback for API models or simple outputs
                completion = full_completion

            print(completion.strip())

        except FileNotFoundError:
            print(f"[{RED}Output file not found for this model {output_file}.{RESET}]")
        except Exception as e:
            print(f"[{RED}Could not read or parse output file: {e}{RESET}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay a specific merge conflict, showing prompt, "
        "ground truth, and all model outputs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="The integer index of the conflict to replay.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="merges/repos_reaper_test/dataset",
        help="Path to the cached dataset directory used in training/evaluation.",
    )
    parser.add_argument(
        "--eval_outputs_root",
        type=str,
        default="eval_outputs/repos_reaper_test/test",
        help="Root directory where evaluation outputs for all models are stored.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'train', 'test').",
    )

    args = parser.parse_args()
    replay(args.index, args.dataset_path, args.eval_outputs_root, args.split)
