# -*- coding: utf-8 -*-
"""
Supervised Fine-Tuning (SFT) Script for merge conflict resolution.

This script:
1. Loads the prepared SFT dataset
2. Fine-tunes the base model using LoRA
3. Saves the trained model for later GRPO training
"""

import os
import argparse
from pathlib import Path
from datasets import load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer

from src.variables import (
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH_SFT,
    LORA_RANK,
    SYSTEM_PROMPT,
)

# Set WANDB project
os.environ["WANDB_PROJECT"] = "LLMerge-SFT"


def train_sft(
    dataset_path: Path,
    train_args,
    output_dir: Path = Path("outputs"),
):
    """Train a model using Supervised Fine-Tuning."""
    # Load dataset
    output_dir = (
        output_dir
        / MODEL_NAME
        / (
            f"{train_args.run_name}_"
            f"lr{train_args.lr}_"
            f"epochs{train_args.epochs}_"
            f"wd{train_args.weight_decay}_"
            f"{train_args.lr_scheduler_type}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # Add system prompt if requested
    if train_args.add_system_prompt:

        def add_system_prompt(example):
            """Add system prompt to the conversation."""
            if "prompt" in example and isinstance(example["prompt"], list):
                # Check if system prompt already exists
                if not (
                    example["prompt"] and example["prompt"][0].get("role") == "system"
                ):
                    # Add system prompt at the beginning
                    example["prompt"] = [
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ] + example["prompt"]
            return example

        print("Adding system prompts to dataset...")
        dataset = dataset.map(add_system_prompt)

    # Initialize model
    print(f"Loading model {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQUENCE_LENGTH_SFT,
        load_in_4bit=True,
        max_lora_rank=LORA_RANK,
    )

    # Set up LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=train_args.epochs,
        learning_rate=train_args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=train_args.weight_decay,
        lr_scheduler_type=train_args.lr_scheduler_type,
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
    )

    # Initialize SFT Trainer
    trainer = SFTTrainer(  # pylint: disable=unexpected-keyword-arg
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQUENCE_LENGTH_SFT,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
    )

    # Start training
    print("Starting SFT training...")
    trainer.train()

    # Save model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    model.save_pretrained_merged(
        output_dir / "final_model_16bit",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for merge conflict resolution"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="merges/repos_reaper_1000/dataset_sft",
        help="Path to the SFT dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning Rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Epochs to train for",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight Decay",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="LR Scheduler Type",
    )
    parser.add_argument(
        "--add_system_prompt",
        action="store_true",
        help="Add system prompt to dataset (for thinking-based training)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="sft_model",
        help="Name prefix for the training run (e.g., 'distill_model', 'sft_model')",
    )
    args = parser.parse_args()

    train_sft(
        dataset_path=args.dataset,
        train_args=args,
        output_dir=Path(args.output_dir),
    )
