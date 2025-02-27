# -*- coding: utf-8 -*-
"""Module for generating responses using a model from Hugging Face's model hub."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from variables import SYSTEM_PROMPT


def load_model(model_name: str):
    """Loads the model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    return model, tokenizer


def generate_response(model_name: str, prompt: str) -> str:
    """Generates a response using the chat template and DeepSeek's system prompt."""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    model, tokenizer = load_model(model_name)
    formatted_prompt = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)  # type: ignore
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    MODEL_NAME = "unsloth/deepSeek-r1-distill-qwen-1.5b"
    QUESTION = "What is 1+1?"

    response = generate_response(MODEL_NAME, QUESTION)

    print("Response:", response)
