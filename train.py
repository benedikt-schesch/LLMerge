# -*- coding: utf-8 -*-
"""UnSloth - GRPO Training Script"""
# pylint: disable=unused-argument

import os
import re
import math
import argparse
from typing import List, Dict
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
import wandb
from src.variables import (
    MAX_SEQUENCE_LENGTH,
    MODEL_NAME,
    LORA_RANK,
    MAX_OUTPUT_LENGTH,
    MAX_PROMPT_LENGTH,
    SYSTEM_PROMPT,
)
from src.utils import extract_code_block, normalize_java_code


os.environ["WANDB_PROJECT"] = "LLMerge"

CORRECT_ANSWER_MULTIPLIER = math.sqrt(2)
JAVA_MARKDOWN_PATTERN = r"```java\n(.*?)\n```"
THINKING_PATTERN = r"^(?:[\s\S]*?)\n</think>\n(?:[\s\S]*)$"
CONFLICT_MARKERS = ["<<<<<<<", "=======", "|||||||", ">>>>>>>"]


# def log_responses(
#     prompts: List[List[Dict[str, str]]], responses: List[str], answer: List[str]
# ) -> None:
#     """Log the responses for debugging"""
#     q = prompts[0][-1]["content"]
#     debug_file = "debug.txt"
#     if os.path.exists(debug_file):
#         with open(debug_file, "r", encoding="utf-8") as f:
#             existing_entries = f.read().count("Question:")
#     else:
#         existing_entries = 0
#     entry_number = existing_entries + 1

#     with open(debug_file, "a", encoding="utf-8") as f:
#         f.write(
#             f"\n\nEntry #{entry_number}\nQuestion:\n{q}\nExpected Answer:\n{answer[0]}\n\n"
#         )
#         for idx, r in enumerate(responses):
#             f.write(f"Response {idx}:\n{r}\n\n")

# ------------------------------------------
# 1) Pre-compile your regex patterns
# ------------------------------------------
THINKING_RE = re.compile(r"^(?:[\s\S]*?)\n</think>\n(?:[\s\S]*)$", re.DOTALL)

# For normalizing Java code
BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/")
LINE_COMMENT_RE = re.compile(r"//.*")
WHITESPACE_RE = re.compile(r"\s+")

CONFLICT_MARKERS = ["<<<<<<<", "=======", "|||||||", ">>>>>>>"]

# Pre-compile patterns
JAVA_MARKDOWN_RE = re.compile(r"```java\n(.*?)\n```", re.DOTALL)
THINKING_RE = re.compile(r"^(?:[\s\S]*?)\n</think>\n(?:[\s\S]*)$", re.DOTALL)


def extract_answer(text: str) -> str:
    """
    Extracts the answer portion from the response (after </think>).
    If there's no </think>, just returns the original text.
    """
    parts = text.split("</think>", 1)
    return parts[-1] if len(parts) > 1 else parts[0]


def has_conflict_markers(text: str) -> bool:
    """Check if the text contains any conflict markers (e.g., '<<<<<<<')."""
    return any(marker in text for marker in CONFLICT_MARKERS)


# ------------------------------------------------------------------
# Reward Functions (using list comprehensions where possible)
# ------------------------------------------------------------------


def format_reward(
    completions: List[List[Dict[str, str]]],
    log_wandb: bool = True,
    **kwargs,
) -> List[float]:
    """
    Reward = 0.5 if the completion matches the 'thinking' pattern.
    Otherwise 0.0.
    """
    rewards = [0.5 if THINKING_RE.match(c[0]["content"]) else 0.0 for c in completions]
    if log_wandb:
        wandb.log({"format_reward": rewards})
    print("Format Reward:", rewards)
    return rewards


def java_markdown_reward(
    completions: List[List[Dict[str, str]]],
    log_wandb: bool = True,
    **kwargs,
) -> List[float]:
    """
    Reward = 1.0 if the *answer block* (after </think>)
    contains a Java code block (```java ... ```).
    Otherwise 0.0.
    """
    rewards = [
        1.0 if JAVA_MARKDOWN_RE.search(extract_answer(c[0]["content"])) else 0.0
        for c in completions
    ]
    if log_wandb:
        wandb.log({"java_markdown_reward": rewards})
    print("Java Markdown Reward:", rewards)
    return rewards


def merged_conflict_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    log_wandb: bool = True,
    **kwargs,
) -> List[float]:
    """
    Merged reward function with the following logic:
      - 1.0 if the completion's code block exactly matches the correct resolution
      - 0.5 if it's only semantically the same (ignoring comments/whitespace)
      - 0.1 if it matches the prompt's code block (i.e. raises a conflict)
      - 0.0 otherwise
    """
    # Extract the "goal" code block (the one in the prompt's last message)
    goal_code_block = extract_code_block(prompts[0][-1]["content"])

    # Print the responses for debugging
    # print("-" * 20, f"\nResponse:\n{completions[0][0]['content']}")

    rewards = [
        (
            0.0
            if (cb := extract_code_block(extract_answer(c[0]["content"]))) is None
            else 1.0
            if cb == answer[idx].strip()  # exact match
            else 0.5
            if normalize_java_code(cb)
            == normalize_java_code(answer[idx].strip())  # semantic match
            else 0.1
            if cb == goal_code_block  # same as prompt => conflict
            else 0.0
        )
        for idx, c in enumerate(completions)
    ]
    if log_wandb:
        wandb.log({"merged_conflict_reward": rewards})
    print("Merged Conflict Reward:", rewards)
    print("Output lengths:", [len(c[0]["content"]) for c in completions])
    return rewards


if __name__ == "__main__":
    PatchFastRL("GRPO", FastLanguageModel)

    parser = argparse.ArgumentParser(description="UnSloth - GRPO Training Script")
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    args = parser.parse_args()
    model_name = args.model_name
    learning_rate = args.learning_rate
    epochs = args.epochs
    resume = args.resume

    print("Loading dataset...")

    dataset = load_from_disk("merges/repos_reaper_java_train/dataset")

    def add_system_prompt(example):
        """Add system prompt to the conversation."""
        if "prompt" in example and isinstance(example["prompt"], list):
            # Check if system prompt already exists
            if not (example["prompt"] and example["prompt"][0].get("role") == "system"):
                # Add system prompt at the beginning
                example["prompt"] = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ] + example["prompt"]
        return example

    # Apply system prompt to the dataset
    print("Adding system prompts to dataset...")
    dataset = dataset.map(add_system_prompt)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning # type: ignore
        random_state=3407,
    )

    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference!
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=1.0,
        warmup_steps=30,
        lr_scheduler_type="constant_with_warmup",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Increase to 4 for smoother training
        num_generations=16,  # Decrease if out of memory
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_OUTPUT_LENGTH,
        temperature=0.9,
        num_train_epochs=epochs,
        max_steps=2500,
        save_steps=100,
        max_grad_norm=0.2,
        report_to="wandb",
        output_dir=f"checkpoints/{MODEL_NAME}",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[  # type: ignore
            format_reward,
            merged_conflict_reward,
        ],
        args=training_args,
        train_dataset=dataset["train"],  # type: ignore
    )
    trainer.train(resume_from_checkpoint=resume)
