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
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import unsloth
from transformers import TextStreamer
import torch
from datasets import load_from_disk
from train import (
    MAX_SEQ_LENGTH,
    MAX_PROMPT_LENGTH,
    SYSTEM_PROMPT,
    extract_code_block,
    compute_conflict_reward,
    compute_goal_file_reward,
    has_conflict_markers,
    format_reward,
    java_markdown_reward,
)

open("eval.log", "w", encoding="utf-8").close()  # pylint: disable=consider-using-with
logger.add("eval.log", backtrace=True, diagnose=True)


def main():  # pylint: disable=too-many-locals, too-many-statements
    """Main function for evaluation script."""
    # Load the dataset (using the same training data)
    dataset = load_from_disk("merges/repos_50/dataset")["train"]

    logger.info("Starting evaluation...")
    logger.info(f"Loaded {len(dataset)} examples.")

    model_name = "unsloth/QwQ-32B"

    torch.set_grad_enabled(False)
    output_dir = Path("eval_ouputs")

    # Load the model and tokenizer (using same parameters as in training)
    if "unsloth" in model_name:
        load_in_4bit = True
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH + MAX_PROMPT_LENGTH + len(SYSTEM_PROMPT),
            load_in_4bit=load_in_4bit,
        )
        if load_in_4bit:
            output_dir = output_dir / f"{model_name}-loaded-4bit"
        else:
            output_dir = output_dir / model_name
        unsloth.FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-outside-toplevel

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        output_dir = output_dir / model_name

    print(f"Device: {model.device}")

    # Set up file to store full outputs before truncation.
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(output_dir / "eval.log", backtrace=True, diagnose=True)

    total = 0
    count_thinking = 0
    count_java_md = 0
    count_conflict_preserved = 0
    count_resolved_perfectly = 0
    count_resolved_semantically = 0
    text_streamer = TextStreamer(tokenizer)  # type: ignore

    # Loop over the examples in the dataset.
    for idx, example in enumerate(tqdm(dataset)):
        total += 1

        # Generate a completion for the given prompt.
        inputs = tokenizer.apply_chat_template(
            example["prompt"],  # type: ignore
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)  # type: ignore

        # Generate with a max number of new tokens.
        output_tokens = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=MAX_SEQ_LENGTH,
            use_cache=True,
        )
        # Get the full completion before truncation.
        full_completion = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
        # Write the full completion to file.
        output_file = output_dir / f"example_{idx}.txt"
        with open(output_file, "w", encoding="utf-8") as output_file:
            output_file.write(full_completion)

        if "<｜Assistant｜>" in full_completion:
            completion = full_completion.split("<｜Assistant｜>", 1)[1]
        elif "<|im_start|>assistant" in full_completion:
            completion = full_completion.split("<|im_start|>assistant", 1)[1]
        else:
            raise ValueError("Could not find completion in full output.")

        # Wrap prompt text into the expected structure.
        wrapped_completions = [[{"content": completion}]]
        wrapped_prompts = [[{"content": example["question"]}]]  # type: ignore

        # Evaluate the thinking format.
        if format_reward(wrapped_completions)[0] > 0:
            count_thinking += 1

        # Evaluate the Java markdown formatting.
        if java_markdown_reward(wrapped_completions)[0] > 0:
            count_java_md += 1

        code_block = extract_code_block(completion)
        if code_block is None:
            continue

        if (
            has_conflict_markers(code_block)
            and compute_conflict_reward(wrapped_prompts, code_block) == 1.0
        ):
            count_conflict_preserved += 1
        elif (
            compute_goal_file_reward(
                wrapped_prompts, code_block, correct_answer_multiplier=1
            )
            == 1.0
        ):
            logger.info(f"Example {idx} resolved perfectly.")
            count_resolved_perfectly += 1
            count_resolved_semantically += 1
        else:
            # Create seantic code blocks i.e. remove all whitespace and newlines and tabs.
            semantic_code_block = " ".join(code_block.split())
            semantic_resolution = " ".join(example["answer"].split())  # type: ignore
            if semantic_code_block == semantic_resolution:
                logger.info(f"Example {idx} semantically resolved.")
                count_resolved_semantically += 1

    # Compute percentages.
    pct_thinking = 100 * count_thinking / total if total > 0 else 0
    pct_java_md = 100 * count_java_md / total if total > 0 else 0
    pct_conflict = 100 * count_conflict_preserved / total if total > 0 else 0
    pct_resolved = 100 * count_resolved_perfectly / total if total > 0 else 0
    pct_resolved_semantic = (
        100 * count_resolved_semantically / total if total > 0 else 0
    )

    logger.success("Evaluation Results:")
    logger.success(f"Total merges evaluated: {total}")
    logger.success(f"Percentage with valid thinking format: {pct_thinking:.2f}%")
    logger.success(f"Percentage with valid Java markdown format: {pct_java_md:.2f}%")
    logger.success(f"Percentage correctly raising merge conflict: {pct_conflict:.2f}%")
    logger.success(f"Percentage correctly resolved merges: {pct_resolved:.2f}%")
    logger.success(
        f"Percentage semantically correctly resolved merges: {pct_resolved_semantic:.2f}%"
    )


if __name__ == "__main__":
    main()
