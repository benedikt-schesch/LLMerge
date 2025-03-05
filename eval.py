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

# Import variables and functions from your training script.
from datasets import load_from_disk
from rich.progress import track
import torch
from train import (
    MAX_SEQ_LENGTH,
    MAX_PROMPT_LENGTH,
    SYSTEM_PROMPT,
    LORA_RANK,
    extract_code_block,
    compute_conflict_reward,
    compute_goal_file_reward,
    has_conflict_markers,
    format_reward,
    java_markdown_reward,
)


def main():  # pylint: disable=too-many-locals
    """Main function for evaluation script."""
    # Load the dataset (using the same training data)
    dataset = load_from_disk("merges/repos_50/dataset")["train"]

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load the model and tokenizer (using same parameters as in training)
    if "unsloth" in model_name:
        from unsloth import FastLanguageModel  # pylint: disable=import-outside-toplevel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH + MAX_PROMPT_LENGTH + len(SYSTEM_PROMPT),
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=LORA_RANK,
        )
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-outside-toplevel

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    total = 0
    count_thinking = 0
    count_java_md = 0
    count_conflict_preserved = 0
    count_resolved = 0

    # Loop over the examples in the dataset.
    for example in track(dataset):
        total += 1
        prompt_text = example["prompt"][-1]["content"]  # type: ignore

        # Generate a completion for the given prompt.
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_text, add_generation_prompt=True, tokenize=False
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)  # type: ignore
        # Generate with a max number of new tokens.
        output_tokens = model.generate(**inputs, max_new_tokens=MAX_SEQ_LENGTH)
        completion = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Wrap prompt text into the expected structure.
        wrapped_prompt = [[{"content": prompt_text}]]

        # Evaluate the thinking format.
        if format_reward(wrapped_prompt)[0] > 0:
            count_thinking += 1

        # Evaluate the Java markdown formatting.
        if java_markdown_reward(wrapped_prompt)[0] > 0:
            count_java_md += 1

        code_block = extract_code_block(completion)
        if code_block is None:
            continue

        if (
            has_conflict_markers(code_block)
            and compute_conflict_reward(wrapped_prompt, code_block) == 1.0
        ):
            count_conflict_preserved += 1
        elif compute_goal_file_reward(wrapped_prompt, code_block) == 1.0:
            count_resolved += 1

    # Compute percentages.
    pct_thinking = 100 * count_thinking / total if total > 0 else 0
    pct_java_md = 100 * count_java_md / total if total > 0 else 0
    pct_conflict = 100 * count_conflict_preserved / total if total > 0 else 0
    pct_resolved = 100 * count_resolved / total if total > 0 else 0

    print("Evaluation Results:")
    print(f"Total merges evaluated: {total}")
    print(f"Percentage with valid thinking format: {pct_thinking:.2f}%")
    print(f"Percentage with valid Java markdown format: {pct_java_md:.2f}%")
    print(f"Percentage correctly raising merge conflict: {pct_conflict:.2f}%")
    print(f"Percentage correctly resolved merges: {pct_resolved:.2f}%")


if __name__ == "__main__":
    main()
