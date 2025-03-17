# -*- coding: utf-8 -*-
"""UnSloth - GRPO Training Script"""
# pylint: disable=unused-argument

import os
import re
import math
from typing import List, Dict
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import wandb
from src.variables import (
    LORA_RANK,
    MAX_OUTPUT_LENGTH,
    MAX_PROMPT_LENGTH,
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
    print("Loading dataset...")

    dataset = load_from_disk("merges/repos_reaper_1000/dataset")

    MODEL_NAME = "outputs/sft_model/final_model_16bit"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True)

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)

    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference!
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        warmup_steps=15,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Increase to 4 for smoother training
        num_generations=16,  # Decrease if out of memory
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_OUTPUT_LENGTH,
        temperature=0.7,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=300,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs",
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
    trainer.train()
    model.save_pretrained("outputs/sft_model/final_model_16bit")
