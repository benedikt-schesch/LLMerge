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
from datasets import load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer

from src.variables import (
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
    LORA_RANK,
)

# Set WANDB project
os.environ["WANDB_PROJECT"] = "LLMerge-SFT"


def train_sft(
    dataset_path,
    output_dir="outputs/sft_model",
):
    """Train a model using Supervised Fine-Tuning."""
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print(f"Train set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")

    # Initialize model
    print(f"Loading model {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.85,
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
    training_args = (
        TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )

    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
    )

    # Start training
    print("Starting SFT training...")
    trainer.train()

    # Save model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for merge conflict resolution"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="outputs/sft_dataset/correct_only",
        help="Path to the SFT dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sft_model",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    args = parser.parse_args()

    train_sft(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )
