# -*- coding: utf-8 -*-
"""UnSloth - GRPO Training Script"""
# pylint: disable=unused-argument

import os
import re
from difflib import SequenceMatcher
from typing import Optional
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
from src.variables import MODEL, MAX_SEQ_LENGTH, LORA_RANK, MAX_PROMPT_LENGTH, SYSTEM_PROMPT

os.environ["WANDB_PROJECT"] = "LLMerge"

PatchFastRL("GRPO", FastLanguageModel)

print("Loading dataset...")

dataset = load_from_disk("repos_reaper_1000/dataset")

CORRECT_REWARD_STRENGTH = 1
JAVA_MARKDOWN_PATTERN = r"```java\n(.*?)\n```"


# Load and prep dataset
def extract_answer(text: str) -> str:
    """Extracts the answer block from the new formatted response."""
    return text.split("</think>")[-1]


def format_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^(?:[\s\S]*?)\n</think>\n(?:[\s\S]*)$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
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


def similarity(s1: str, s2: str) -> float:
    """Return similarity ratio between two strings (1.0 means identical)."""
    return SequenceMatcher(None, s1, s2).ratio()


def extract_code_block(text: str) -> Optional[str]:
    """
    Extracts a code block from a markdown-formatted text.
    If no markdown code block is found, returns None
    """
    match = re.search(JAVA_MARKDOWN_PATTERN, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def correctness_reward_func(  # pylint: disable=too-many-locals
    prompts, completions, answer, **kwargs
) -> list[float]:
    """
    Computes reward as the similarity ratio (acting as a normalized edit distance signal)
    between the answer block from the response and a reference.

    - If the answer block contains any conflict markers (e.g., <<<<<<<, =======, >>>>>>>),
      the reference is the code block extracted from the input prompt.
    - Otherwise, the reference is the expected answer (provided via `answer`).
    """
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_answer(r) for r in responses]

    # Print the first response for debugging
    print("-" * 20, f"\nResponse:\n{responses[0]}")

    # Log details to debug file
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

    rewards = []
    for r in extracted_responses:
        r = extract_code_block(r)
        if r is None:
            rewards.append(0.0)
        elif any(
            marker in r for marker in ["<<<<<<<", "=======", "|||||||", ">>>>>>>"]
        ):
            # If conflict markers are present, compare the answer to the input prompt's code block.
            code_block = extract_code_block(q)
            assert code_block is not None, "Code block not found in prompt"
            sim = similarity(r, code_block)
            rewards.append(sim)
        else:
            # Otherwise, compare to the expected resolved result.
            sim = similarity(r, answer[0])
            if sim == 1.0:
                rewards.append(CORRECT_REWARD_STRENGTH * sim)
            else:
                rewards.append(sim)
    return rewards


if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LENGTH + MAX_PROMPT_LENGTH + len(SYSTEM_PROMPT),
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
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=8,  # Decrease if out of memory
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_SEQ_LENGTH,
        temperature=0.6,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=1000,
        save_steps=200,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs",
        run_name="testing",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[  # type: ignore
            format_reward,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset["train"],  # type: ignore
    )
    trainer.train()
