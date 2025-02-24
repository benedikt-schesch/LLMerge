# -*- coding: utf-8 -*-
"""Script to train a Qwen model using GRPO"""
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Import PyTorch and Hugging Face Transformers
import torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
# Import libraries from TRL (Transformers Reinforcement Learning)
from trl import GRPOConfig, GRPOTrainer

# Import dataset utilities
from build_dataset import validate_dataset

# Parameters
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-GRPO-training"
DATASET_PATH = "merges/repos_50/dataset"  # Path produced by build_dataset.py
# Device
device = torch.device("cpu")
print(f"Using device: {device}")


def get_model(device):
    """Get the model and tokenizer for training."""
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

    # Move model to the appropriate device
    model.to(device)

    return model, tokenizer


# Test basic inference
def test_model_inference(user_input: str | List[Dict[str, str]]):
    """Test basic model inference with the loaded model and tokenizer."""
    if isinstance(user_input, str):
        messages = [
            {"role": "system", "content": "You are Qwen, a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
    else:
        messages = user_input

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, max_new_tokens=100, do_sample=True, temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


model, tokenizer = get_model(device)

# Test the model
test_input = "how are you?"
response = test_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Model Response: {response}")

dataset = load_from_disk(DATASET_PATH)

# Validate dataset
validate_dataset(dataset)

print(
    f"Loaded dataset: {len(dataset['train'])} training samples and {len(dataset['test'])} test samples."
)

# Give an example prompt and test the model
sample = dataset["train"][0]
prompt = sample["prompt"]
print(f"Example prompt: {prompt}")
response = test_model_inference(prompt)
print(f"Model response: {response}")

print("==== START TRAINING ====")
# Reward Functions


# Accuracy Reward
def accuracy_reward(completions, **kwargs):
    """
    Reward function to check if the model's response is mathematically
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """

    # Extract responses
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    solutions = kwargs.get("solution")  # Get solutions from kwargs

    if solutions is None:
        return [0.5] * len(completions)  # Return neutral reward if no solution

    return [0.5] * len(completions)  # Return neutral reward


# In this function, we check whether the model response is **equivalent** to the correct answer. Instead of comparing raw text, we:


# Format Reward
# Implement Format Reward Function
def format_reward(completions, **kwargs):
    """
    Reward function to check if the completion has the correct format:
    <think>...</think> <answer>...</answer>.
    """
    # Define the regex pattern for the desired format
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    # Extract the content from each completion
    completion_contents = [completion[0]["content"] for completion in completions]

    # Check if each completion matches the pattern
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]

    # Reward 1.0 for correct format, 0.0 otherwise
    return [1.0 if match else 0.0 for match in matches]


# In this function:


# Reasoning Steps Reward
def reasoning_steps_reward(completions, **kwargs):
    r"""
    Reward function to encourage clear step-by-step reasoning.
    It looks for patterns like "Step 1:", numbered lists, bullet points,
    and transition words.
    """
    # Regex pattern to find indicators of reasoning steps
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Count the number of reasoning step indicators in each completion
    matches = [
        len(re.findall(pattern, content, re.MULTILINE))
        for content in completion_contents
    ]

    # Reward is proportional to the number of reasoning steps, maxing out at 1.0
    # We're using a "magic number" 3 here - encourage at least 3 steps for full reward
    return [min(1.0, count / 3) for count in matches]


# Cosine Scaled Reward


# Implement Cosine Scaled Reward Function
def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """

    def cosine_scaled_reward(completions, solution, accuracy_rewards, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
            gen_len = len(content)  # Length of the generated answer
            progress = gen_len / max_len  # How far we are to max length
            cosine = math.cos(progress * math.pi)  # Cosine value based on progress

            if (
                acc_reward > 0.5
            ):  # Assuming accuracy_reward gives ~1.0 for correct answers
                min_value = min_value_correct
                max_value = max_value_correct
            else:  # Incorrect answer
                min_value = max_value_wrong  # Note the swap!
                max_value = min_value_wrong

            # Cosine scaling formula!
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards

    return cosine_scaled_reward


# `get_cosine_scaled_reward(...)` generates a reward function for training, customizing scaling with parameters like min_value_wrong/max_value_wrong (penalty range for incorrect answers) and min_value_correct/max_value_correct (reward range for correct ones). max_len sets the maximum length for scaling.


# Repetition Penalty Reward
def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """Helper function to generate n-grams from text."""
        words = text.lower().split()  # Lowercase and split into words
        return zip(*[words[i:] for i in range(ngram_size)])  # Create n-grams

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":  # No penalty for empty completions
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:  # No penalty for short completions
                rewards.append(0.0)
                continue

            ngrams = set()  # Use a set to store unique n-grams
            total = 0
            for ng in zipngram(completion, ngram_size):  # Generate n-grams
                ngrams.add(ng)  # Add n-gram to the set (duplicates are ignored)
                total += 1  # Count total n-grams

            # Calculate scaling factor: more repetition -> higher scaling
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty  # Apply penalty based on scaling
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


# Our `get_repetition_penalty_reward(...)` creates a reward function to penalize repetition, with parameters like ngram_size (default 3, for trigrams) and max_penalty (a negative value, e.g., -0.1).


# Training Configurations for R1 Zero
@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward for cosine scaling for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for cosine scaling for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward for cosine scaling for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for cosine scaling for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for cosine scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={
            "help": "Maximum (negative) penalty for for repetition penalty reward"
        },
    )


# Our `@dataclass` decorator makes it easy to create a class for storing data. WhileGRPOScriptArguments class holds reward settings.

# Define TrainingArguments from transformers
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # Output directory for checkpoints and logs
    overwrite_output_dir=True,
    num_train_epochs=1,  # Total number of training epochs
    per_device_train_batch_size=8,  # Batch size per device during training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    learning_rate=5e-5,  # Initial learning rate for AdamW optimizer
    warmup_ratio=0.1,  # Linear warmup over warmup_ratio fraction of training steps
    weight_decay=0.01,  # Apply weight decay to all layers except bias and LayerNorm weights
    logging_steps=10,  # Log every X updates steps
    evaluation_strategy="steps",  # Evaluate every `eval_steps`
    eval_steps=50,  # Evaluation and logging steps
    save_strategy="steps",  # Save checkpoint every `save_steps`
    save_steps=50,  # Save checkpoint every X updates steps
    save_total_limit=2,  # Limit the total amount of checkpoints. Deletes the older checkpoints.
    dataloader_num_workers=2,  # Number of subprocesses to use for data loading
    seed=42,  # Random seed for reproducibility
    bf16=True,  # Use mixed precision BFP16 training
    push_to_hub=False,  # Whether to push the final model to Hugging Face Hub
    gradient_checkpointing=True,  # Enable gradient checkpointing
    report_to="none",  # Reporting to no one
    no_cuda=(
        True if device == torch.device("cpu") else False
    ),  # Disable CUDA if using CPU
)


# Finally, we need to have a ModelConfig. This is where we put settings that are specific to the **model itself**, like which pre-trained model to use, what data type to use (like bfloat16), and whether to trust remote code or not and so.
@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """

    model_name_or_path: str = field(
        default=MODEL_NAME,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default `torch_dtype` and load the model under this dtype."
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code when loading model and tokenizer."},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": "Attention implementation to use. 'flash_attention_2' or None"
        },
    )


# Our **ModelConfig** class holds key settings, including model_name_or_path, which defaults to **Qwen 0.5B Instruct**. We use torch_dtype="bfloat16" for efficiency and set trust_remote_code=True for safe remote loading. Additionally, attn_implementation="flash_attention_2" is enabled for potentially faster training if supported.


# Instantiate configuration objects
script_args = GRPOScriptArguments()
model_args = ModelConfig()


# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,  # Assuming accuracy_reward is defined in previous steps
        "format": format_reward,  # Assuming format_reward is defined in previous steps
        "reasoning_steps": reasoning_steps_reward,  # Assuming reasoning_steps_reward is defined
        "cosine": get_cosine_scaled_reward(  # Assuming get_cosine_scaled_reward is defined
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(  # Assuming get_repetition_penalty_reward is defined
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list


# Our callback function which will track loss and other important info.


logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """
    A simple callback for logging training information at specific steps.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % args.logging_steps == 0:
            logger.info(
                f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}"
            )


def get_callbacks(training_args, model_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()]  # Instantiate our LoggingCallback
    return callbacks


# Finally, initializing these function.

# Get reward functions and callbacks
reward_functions = get_reward_functions(script_args)
callbacks = get_callbacks(training_args, model_args, script_args)


# ## GRPO Training Loop


# Create GRPOConfig from TrainingArguments
grpo_config = GRPOConfig(
    **training_args.to_dict(),  # Convert TrainingArguments to dictionary and unpack
    **{
        # REMOVED model_init_kwargs here
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
    },
)

grpo_trainer = GRPOTrainer(
    model=model,  # Our initialized Qwen model
    reward_funcs=reward_functions,  # List of reward functions from previous step
    args=grpo_config,  # GRPOConfig (created from TrainingArguments)
    train_dataset=dataset["train"],  # Training dataset
    eval_dataset=dataset["test"],  # Evaluation dataset
    callbacks=callbacks,  # List of callbacks
)


# We can now start the **Training Loop**! This is as simple as calling the train() method on our grpo_trainer.


# Start the GRPO Training Loop
train_result = grpo_trainer.train()


# When you run this cell, you should see the training process begin.


# ## Saving Tiny R1 Zero LLM


# Define the path to your trained model (same as OUTPUT_DIR)
TRAINED_MODEL_PATH = "data/Qwen-GRPO-training"

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

# Save the trained model
grpo_trainer.save_model(TRAINED_MODEL_PATH)

print(f"GRPO Trained model saved to {TRAINED_MODEL_PATH}")
