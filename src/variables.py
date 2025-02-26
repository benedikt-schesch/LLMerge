# -*- coding: utf-8 -*-
"""This file contains the global variables used in the project."""

MODEL = "unsloth/deepSeek-r1-distill-qwen-1.5b"  # Model to use for generation
MAX_SEQ_LENGTH = 2048  # Maximum number of tokens of the entire sequence
MAX_PROMPT_LENGTH = 256  # Maximum number of tokens in the prompt
LORA_RANK = 64  # Larger rank = smarter, but slower
