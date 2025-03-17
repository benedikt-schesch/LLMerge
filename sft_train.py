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
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

from src.variables import (
    MODEL_NAME,
)

# Set WANDB project
os.environ["WANDB_PROJECT"] = "LLMerge-SFT"


def train_sft(
    dataset_path: Path,
    output_dir: Path = Path("outputs"),
):
    """Train a model using Supervised Fine-Tuning."""
    # Load dataset
    output_dir = output_dir / MODEL_NAME / "sft_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=str(output_dir),
        report_to="wandb",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_token"],
        task_type="CAUSAL_LM",
    )

    # Initialize SFT Trainer
    trainer = SFTTrainer(
        MODEL_NAME,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
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
        default="merges/repos_reaper_1000/dataset_sft",
        help="Path to the SFT dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Directory to save the trained model",
    )
    args = parser.parse_args()

    train_sft(
        dataset_path=args.dataset,
        output_dir=Path(args.output_dir),
    )
