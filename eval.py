#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for merge outputs.
Loads the same dataset as in training and computes:
  - % with valid thinking format
  - % with valid Java markdown formatting
  - % that correctly raise the merge conflict (i.e. preserve the original conflict)
  - % that are correctly resolved
"""

import argparse
import os
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger
import unsloth
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import torch
from datasets import load_from_disk

# Set HF_HOME to avoid re-downloading models
os.environ["HF_HOME"] = "/m-coriander/coriander/scheschb/.cache/"

from train import (
    merged_conflict_reward,
    format_reward,
    java_markdown_reward,
)
from src.variables import (
    MAX_SEQUENCE_LENGTH,
    MAX_OUTPUT_LENGTH,
    MODEL_NAME,
    SYSTEM_PROMPT,
)
from src.utils import cached_query_deepseek_api, cached_query_openrouter

# Define remote/api model identification
API_MODEL_NAMES = {"api/deepseek-r1", "o3"}
API_MODEL_PREFIXES = (
    "openai",
    "anthropic",
    "qwen",
    "meta",
    "google",
    "x-ai",
    "deepseek",
)


def is_api_model(model_name: str) -> bool:
    """Check if the model is an API model."""
    return model_name in API_MODEL_NAMES or any(
        model_name.startswith(prefix) for prefix in API_MODEL_PREFIXES
    )


open("eval.log", "w", encoding="utf-8").close()  # pylint: disable=consider-using-with
logger.add("eval.log", backtrace=True, diagnose=True)


def model_inference(
    example, model, tokenizer, text_streamer, add_system_prompt=False, no_thinking=False
):
    """Perform model inference."""
    if model == "api/deepseek-r1":
        # Bypass local model and call deepseek_sft_data's query_deepseek_api
        response = cached_query_deepseek_api(example["question"])
        if response is None:
            return ""
        reasoning = response["reasoning"]
        result = response["result"]
        # Combine them into a single string that the rest of the eval script expects
        full_completion = f"<think>\n{reasoning}</think>\n{result}"
        return full_completion
    if isinstance(model, str) and is_api_model(model):
        # Bypass local model and call openrouter's query_openrouter
        response = cached_query_openrouter(example["question"], model)
        if response is None:
            return ""
        result = response["result"]
        reasoning = getattr(result, "reasoning", "No reasoning found")
        # Combine them into a single string that the rest of the eval script expects
        full_completion = f"<think>\n{reasoning}</think>\n{result}"
        return full_completion

    # Prepare the prompt based on the dataset structure
    if "prompt" in example and isinstance(example["prompt"], list):
        # Use the prompt as-is if it's already in chat format
        prompt = example["prompt"].copy()  # Make a copy to avoid modifying original

        # In no_thinking mode, remove system prompt and any thinking-related content
        if no_thinking:
            # Remove system prompt if present
            if prompt and prompt[0].get("role") == "system":
                prompt = prompt[1:]

            # For assistant messages, remove thinking tags if present
            for msg in prompt:
                if msg.get("role") == "assistant" and "<think>" in msg.get(
                    "content", ""
                ):
                    content = msg["content"]
                    # Remove everything between <think> and </think> tags
                    import re

                    content = re.sub(
                        r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
                    )
                    msg["content"] = content.strip()
        elif add_system_prompt:
            # Add system prompt if requested and not already present
            if not (prompt and prompt[0].get("role") == "system"):
                prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt
    else:
        # Create chat format from question
        prompt = [{"role": "user", "content": example["question"]}]
        if add_system_prompt and not no_thinking:
            prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt

    # Generate a completion for the given prompt.
    # Always disable thinking mode for Qwen models
    template_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
        "enable_thinking": False,  # Always disabled for non-reasoning mode
    }

    inputs = tokenizer.apply_chat_template(prompt, **template_kwargs).to(model.device)  # type: ignore

    # Generate with a max number of new tokens.
    # Use Qwen3 recommended parameters for non-thinking mode
    output_tokens = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=MAX_OUTPUT_LENGTH,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        use_cache=True,
    )
    # Get the full completion before truncation.
    full_completion = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
    return full_completion


def get_model(
    model_name, load_in_4bit: bool = True, lora_weights: Optional[str] = None
):
    """Load the model and tokenizer."""
    # Load the model and tokenizer (using same parameters as in training)
    if model_name == "api/deepseek-r1":
        return "api/deepseek-r1", None, None
    if is_api_model(model_name):
        return model_name, None, None

    if "unsloth" in model_name or "output" in model_name or "checkpoint" in model_name:
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            load_in_4bit=load_in_4bit,
        )

        # Set up chat template for Qwen3 (same as in training)
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen-3",
        )

        # Enable inference mode for 2x faster inference
        unsloth.FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-outside-toplevel

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if lora_weights:
        model.load_adapter(lora_weights)

    print(f"Device: {model.device}")
    text_streamer = TextStreamer(
        tokenizer, skip_prompt=True
    )  # Skip prompt for cleaner output
    return model, tokenizer, text_streamer


def main():  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    """Main function for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluation script for merge outputs.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Model name to load",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="merges/repos_reaper_test/dataset",
        help="Path to the dataset on disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs",
        help="Directory to store evaluation outputs",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to the LoRA weights",
    )
    parser.add_argument(
        "--load_in_4bit",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Load model in 4bit mode",
    )
    parser.add_argument(
        "--no_thinking",
        action="store_true",
        help="Model was trained without thinking mode (for direct SFT)",
    )
    parser.add_argument(
        "--add_system_prompt",
        action="store_true",
        help="Add system prompt to evaluation (if model was trained with it)",
    )
    args = parser.parse_args()

    # Load the dataset (using the same training data)
    # Try to load with split first, if that fails, load the entire dataset
    try:
        dataset = load_from_disk(args.dataset_path)[args.split]
    except KeyError:
        # If the dataset doesn't have splits, load it directly
        dataset = load_from_disk(args.dataset_path)
        # If it's a DatasetDict with a single key, use that
        if hasattr(dataset, "keys") and len(dataset.keys()) == 1:
            dataset = dataset[list(dataset.keys())[0]]

    logger.info("Starting evaluation...")
    logger.info(f"Loaded {len(dataset)} examples.")

    model_name = args.model_name
    load_in_4bit = args.load_in_4bit
    lora_weights = args.lora_weights

    torch.set_grad_enabled(False)
    output_dir = Path(args.output_dir)

    # Extract dataset name from dataset_path, assuming format 'merges/{dataset_name}/dataset'
    parts = args.dataset_path.split("/")
    dataset_name = parts[1] if len(parts) > 2 else "default"
    output_dir = output_dir / dataset_name / args.split
    if lora_weights:
        output_dir = output_dir / lora_weights
    elif load_in_4bit:
        output_dir = output_dir / f"{model_name}-loaded-4bit"
    else:
        output_dir = output_dir / model_name
    # Set up file to store full outputs before truncation.
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(output_dir / "eval.log", backtrace=True, diagnose=True)

    # Lazy model loading: initialize as None.
    model, tokenizer, text_streamer = None, None, None

    total = 0
    count_thinking = 0
    count_java_md = 0
    count_conflict_preserved = 0
    count_resolved_perfectly = 0
    count_resolved_semantically = 0

    # Loop over the examples in the dataset.
    # Pre-generate full completions in parallel for remote models
    if is_api_model(model_name):
        if model is None:
            model, tokenizer, text_streamer = get_model(
                model_name, load_in_4bit, lora_weights
            )

        def _gen(gen_args):
            idx, example = gen_args
            output_file_path = output_dir / f"example_{idx}.txt"
            if not output_file_path.exists():
                logger.info(f"Parallel processing example {idx}...")
                full = model_inference(
                    example,
                    model,
                    tokenizer,
                    text_streamer,
                    args.add_system_prompt,
                    args.no_thinking,
                )
                output_file_path.write_text(full, encoding="utf-8")

        with ThreadPoolExecutor(max_workers=32) as executor:
            # Show progress bar for parallel pre-generation
            list(
                tqdm(
                    executor.map(_gen, enumerate(dataset)),
                    total=len(dataset),
                    desc="Pre-generating full completions",
                )
            )
    pbar = tqdm(dataset)
    for idx, example in enumerate(pbar):
        total += 1

        output_file_path = output_dir / f"example_{idx}.txt"
        print(f"Output file path: {output_file_path}")
        if output_file_path.exists():
            logger.info(f"Loading example {idx} from file...")
            with open(output_file_path, "r", encoding="utf-8") as f:
                full_completion = f.read()
        else:
            logger.info(f"Processing example {idx}...")
            # Load the model lazily if not already loaded.
            if model is None:
                model, tokenizer, text_streamer = get_model(
                    model_name, load_in_4bit, lora_weights
                )
            full_completion = model_inference(
                example,
                model,
                tokenizer,
                text_streamer,
                args.add_system_prompt,
                args.no_thinking,
            )
            # Write the full completion to file.
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(full_completion)
        if "<｜Assistant｜>" in full_completion:
            completion = full_completion.split("<｜Assistant｜>", 1)[1]
        elif "<|im_start|>assistant" in full_completion:
            completion = full_completion.split("<|im_start|>assistant", 1)[1]
        elif is_api_model(model_name):
            completion = full_completion
        else:
            raise ValueError("Could not find completion in full output.")

        # Wrap prompt text into the expected structure.
        completions = [[{"content": completion}]]
        prompts = [[{"content": example["question"]}]]  # type: ignore
        answers = [example["answer"]]  # type: ignore

        # Evaluate the thinking format (skip for no_thinking models)
        if not args.no_thinking and format_reward(completions, log_wandb=False)[0] > 0:
            count_thinking += 1

        # Evaluate the Java markdown formatting.
        if java_markdown_reward(completions, log_wandb=False)[0] > 0:
            count_java_md += 1

        reward = merged_conflict_reward(prompts, completions, answers, log_wandb=False)[
            0
        ]

        # If the model raises a conflict
        if reward == 0.1:
            count_conflict_preserved += 1

        # If the model resolves the conflict semantically
        if reward >= 0.5:
            logger.info(f"Semantically resolved {idx}.")
            count_resolved_semantically += 1

        # If the model resolves the conflict perfectly
        if reward == 1.0:
            logger.info(f"Resolved {idx}.")
            count_resolved_perfectly += 1

        # Update progress bar with current percentages.
        pbar.set_postfix(
            {
                "Correct": f"{100 * count_resolved_perfectly / total:.2f}%",
                "Semantic Correct": f"{100 * count_resolved_semantically / total:.2f}%",
            }
        )

    # Compute final percentages.
    pct_thinking = 100 * count_thinking / total if total > 0 else 0
    pct_java_md = 100 * count_java_md / total if total > 0 else 0
    pct_conflict = 100 * count_conflict_preserved / total if total > 0 else 0
    pct_resolved = 100 * count_resolved_perfectly / total if total > 0 else 0
    pct_resolved_semantic = (
        100 * count_resolved_semantically / total if total > 0 else 0
    )

    logger.success("Evaluation Results:")
    logger.success(f"Total merges evaluated: {total}")
    if not args.no_thinking:
        logger.success(f"Percentage with valid thinking format: {pct_thinking:.2f}%")
    logger.success(f"Percentage with valid Java markdown format: {pct_java_md:.2f}%")
    logger.success(f"Percentage correctly raising merge conflict: {pct_conflict:.2f}%")
    logger.success(
        f"Percentage semantically correctly resolved merges: {pct_resolved_semantic:.2f}%"
    )
    logger.success(f"Percentage correctly resolved merges: {pct_resolved:.2f}%")


if __name__ == "__main__":
    main()
