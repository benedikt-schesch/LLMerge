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
    MODEL,
    MAX_SEQ_LENGTH,
    LORA_RANK,
    MAX_PROMPT_LENGTH,
    SYSTEM_PROMPT,
)

# Set WANDB project
os.environ["WANDB_PROJECT"] = "LLMerge-SFT"


def train_sft(
    dataset_path,
    output_dir="outputs/sft_model",
    epochs=1,
    batch_size=1,
    gradient_accumulation_steps=4,
):
    """Train a model using Supervised Fine-Tuning."""
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print(f"Train set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")

    # Initialize model
    print(f"Loading model {MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LENGTH + MAX_PROMPT_LENGTH + len(SYSTEM_PROMPT),
        load_in_4bit=True,
        fast_inference=False,  # Disable during training
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
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="wandb",
    )

    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="completion",
        packing=False,
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
