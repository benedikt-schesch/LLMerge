# -*- coding: utf-8 -*-
"""UnSloth - GRPO Training Script"""
# pylint: disable=unused-argument

import os
import re
import math
from typing import Optional, List, Dict
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
from src.variables import (
    MODEL,
    MAX_SEQ_LENGTH,
    LORA_RANK,
    MAX_PROMPT_LENGTH,
    SYSTEM_PROMPT,
)

os.environ["WANDB_PROJECT"] = "LLMerge"

CORRECT_ANSWER_MULTIPLIER = math.sqrt(2)
JAVA_MARKDOWN_PATTERN = r"```java\n(.*?)\n```"
THINKING_PATTERN = r"^(?:[\s\S]*?)\n</think>\n(?:[\s\S]*)$"
CONFLICT_MARKERS = ["<<<<<<<", "=======", "|||||||", ">>>>>>>"]


# Load and prep dataset
def extract_answer(text: str) -> str:
    """Extracts the answer block from the new formatted response."""
    return text.split("</think>")[-1]


def format_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(THINKING_PATTERN, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def java_markdown_reward(completions, **kwargs) -> list[float]:
    """
    Checks if the answer block (extracted via extract_xml_answer) contains Java markdown formatting.
    This version is 'strong' because it only considers the content within the answer block.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = [
        1.0 if re.search(JAVA_MARKDOWN_PATTERN, extract_answer(r), re.DOTALL) else 0.0
        for r in responses
    ]
    return rewards


def extract_code_block(text: str) -> Optional[str]:
    """
    Extracts a code block from a markdown-formatted text.
    If no markdown code block is found, returns None
    """
    match = re.search(JAVA_MARKDOWN_PATTERN, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def log_responses(prompts, responses, answer):
    """Log the responses for debugging"""
    q = prompts[0][-1]["content"]
    debug_file = "debug.txt"
    if os.path.exists(debug_file):
        with open(debug_file, "r", encoding="utf-8") as f:
            existing_entries = f.read().count("Question:")
    else:
        existing_entries = 0
    entry_number = existing_entries + 1

    with open(debug_file, "a", encoding="utf-8") as f:
        f.write(
            f"\n\nEntry #{entry_number}\nQuestion:\n{q}\nExpected Answer:\n{answer[0]}\n\n"
        )
        for idx, r in enumerate(responses):
            f.write(f"Response {idx}:\n{r}\n\n")


def has_conflict_markers(text: str) -> bool:
    """Check if the text contains any conflict markers."""
    return any(marker in text for marker in CONFLICT_MARKERS)


def compute_conflict_reward(
    prompts: List[List[Dict[str, str]]], code_block: str
) -> float:
    """
    Computes reward as the similarity ratio (acting as a normalized edit distance signal)
    between the answer block from the response and a reference.
    """
    goal_code_block = extract_code_block(prompts[0][-1]["content"])
    assert goal_code_block is not None, "Code block not found in prompt"
    sim = code_block == goal_code_block
    return sim


def compute_goal_file_reward(
    prompts: List[List[Dict[str, str]]],
    code_block: str,
    correct_answer_multiplier: float = CORRECT_ANSWER_MULTIPLIER,
) -> float:
    """
    Computes reward as the similarity ratio (acting as a normalized edit distance signal)
    between the answer block from the response and a reference.
    """
    goal_code_block = extract_code_block(prompts[0][-1]["content"])
    assert goal_code_block is not None, "Code block not found in prompt"
    sim = code_block == goal_code_block
    return correct_answer_multiplier * sim if sim == 1.0 else sim


def correctness_reward_func(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    correct_answer_multiplier: float = CORRECT_ANSWER_MULTIPLIER,
    **kwargs,
) -> list[float]:
    """
    Computes reward as the similarity ratio (acting as a normalized edit distance signal)
    between the answer block from the response and a reference.

    - If the answer block contains any conflict markers (e.g., <<<<<<<, =======, >>>>>>>),
      the reference is the code block extracted from the input prompt.
    - Otherwise, the reference is the expected answer (provided via `answer`).
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]

    # Print the first response for debugging
    print("-" * 20, f"\nResponse:\n{responses[0]}")

    log_responses(prompts, responses, answer)

    rewards = []
    for response in extracted_responses:
        code_block = extract_code_block(response)
        if code_block is None:
            rewards.append(0.0)
        elif has_conflict_markers(code_block):
            rewards.append(compute_conflict_reward(prompts, code_block))
        else:
            rewards.append(
                compute_goal_file_reward(prompts, code_block, correct_answer_multiplier)
            )

    # Square the rewards to amplify the signal
    rewards = [r**2 for r in rewards]
    return rewards


def normalize_java_code(code: str) -> str:
    """
    Normalizes Java code by removing block comments, line comments, and extra whitespace.
    This helps to focus on the semantics rather than formatting details.
    """
    # Remove block comments (/* ... */)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    # Remove line comments (//...)
    code = re.sub(r"//.*", "", code)
    # Replace multiple whitespace characters with a single space
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def soft_correctness_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    **kwargs,
) -> list[float]:
    """
    Computes a soft reward based on semantic similarity between the Java code in the response and the
    goal code block. Instead of checking for strict equality, it normalizes both pieces of code
    (removing comments and extra whitespace) before comparing, which helps to capture Java semantics.
    """
    # Extract the content responses and the answer block from each response
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]

    # Log responses for debugging purposes
    log_responses(prompts, responses, answer)

    rewards = []
    for response in extracted_responses:
        code_block = extract_code_block(response)
        if code_block is None:
            rewards.append(0.0)
        else:
            goal_code_block = extract_code_block(prompts[0][-1]["content"])
            if goal_code_block is None:
                raise ValueError("Code block not found in prompt")
            # Normalize both code blocks to remove non-semantic differences
            normalized_code = normalize_java_code(code_block)
            normalized_goal = normalize_java_code(goal_code_block)
            rewards.append(normalized_code == normalized_goal)
    return rewards


if __name__ == "__main__":
    PatchFastRL("GRPO", FastLanguageModel)

    print("Loading dataset...")

    dataset = load_from_disk("merges/repos_reaper_1000/dataset")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LENGTH + MAX_PROMPT_LENGTH + len(SYSTEM_PROMPT),
        load_in_4bit=False,  # False for LoRA 16bit
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
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0,
        warmup_ratio=0,
        warmup_steps=20,
        lr_scheduler_type="constant_with_warmup",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=14,  # Decrease if out of memory
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_SEQ_LENGTH,
        temperature=0.9,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=500,
        save_steps=100,
        max_grad_norm=0.2,
        report_to="wandb",
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[  # type: ignore
            format_reward,
            correctness_reward_func,
            soft_correctness_reward,
        ],
        args=training_args,
        train_dataset=dataset["train"],  # type: ignore
    )
    trainer.train()
