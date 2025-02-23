# -*- coding: utf-8 -*-
"""Training script"""
# Import necessary libraries
import os

# Import PyTorch and Hugging Face Transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import dataset utilities

# Import libraries from TRL (Transformers Reinforcement Learning)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-GRPO-training"  # For saving our trained model

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer with chat template
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, padding_side="right"
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabulary size: {len(tokenizer)}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


# Initialize base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
)

print(f"Model parameters: {model.num_parameters():,}")
