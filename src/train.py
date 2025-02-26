# -*- coding: utf-8 -*-
"""UnSloth - GRPO Training Script"""
# pylint: disable=unused-argument

import os
import re
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
from variables import MODEL, MAX_SEQ_LENGTH, LORA_RANK, MAX_PROMPT_LENGTH

PatchFastRL("GRPO", FastLanguageModel)

os.environ["WANDB_PROJECT"] = "LLMerge"


dataset = load_from_disk("merges/repos_50/dataset")


# Load and prep dataset
def extract_xml_answer(text: str) -> str:
    """Extracts the answer block from the XML-formatted response."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


JAVA_MARKDOWN_PATTERN = r"```java\s*\n.*?```"


def java_markdown_weak_reward_func(completions, **kwargs) -> list[float]:
    """
    Checks if the entire solution (i.e. the complete response) contains Java markdown formatting.
    This version is 'weak' as it does not restrict the check to the answer block.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        # Search for a java code block anywhere in the response.
        if re.search(JAVA_MARKDOWN_PATTERN, r, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def java_markdown_strong_reward_func(completions, **kwargs) -> list[float]:
    """
    Checks if the answer block (extracted via extract_xml_answer) contains Java markdown formatting.
    This version is 'strong' because it only considers the content within the answer block.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        answer_block = extract_xml_answer(r)
        if re.search(JAVA_MARKDOWN_PATTERN, answer_block, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Modified correctness reward function:
    Checks if the expected answer is contained within the answer block.
    Automatically keeps track of the number of entries in the debug file.
    """
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    print(
        "-" * 20,
        f"\nResponse:\n{responses[0]}",
    )

    # Count existing entries in the debug file
    debug_file = "debug.txt"
    if os.path.exists(debug_file):
        with open(debug_file, "r", encoding="utf-8") as f:
            existing_entries = f.read().count("Question:")
    else:
        existing_entries = 0

    entry_number = existing_entries + 1

    # Append to debug file with entry number
    with open(debug_file, "a", encoding="utf-8") as f:
        f.write(
            f"\n\nEntry #{entry_number}\nQuestion:\n{q}\nAnswer:\n{answer[0]}\n\n\n"
        )
        for idx, r in enumerate(responses):
            f.write(f"\nResponse {idx}:\n{r}\n\n\n")

    # Reward if the expected answer is a substring of the extracted answer block.
    return [2.0 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
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
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
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
        temperature=0.7,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs",
        run_name="testing",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            java_markdown_weak_reward_func,
            java_markdown_strong_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset["train"],
    )
    trainer.train()
