# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""This file contains the global variables used in the project."""

MODEL_NAME = "unsloth/Qwen3-14B"  # Model to use for generation
MAX_OUTPUT_LENGTH = 2048  # Maximum number of tokens of the entire sequence
MAX_PROMPT_LENGTH = 512  # Maximum number of tokens in the prompt
LORA_RANK = 128  # Larger rank = smarter, but slower

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed within <think> </think> followed by the answer, i.e., "
    "<think> reasoning process here </think> answer here"
)

MAX_SEQUENCE_LENGTH = MAX_OUTPUT_LENGTH + MAX_PROMPT_LENGTH + len(SYSTEM_PROMPT)
MAX_SEQUENCE_LENGTH_SFT = 4 * MAX_OUTPUT_LENGTH + MAX_PROMPT_LENGTH + len(SYSTEM_PROMPT)

QUERY_PROMPT = (
    "You are a semantic merge conflict resolution expert. Below is a snippet of code "
    "with surrounding context that includes a merge conflict.\n"
    "Return the entire snippet (including full context) in markdown code syntax as provided, make sure you do not modify the context at all and preserve the spacing as is.\n"
    "Think in terms of intent and semantics that both sides of the merge are trying to achieve.\n"
    "If you are not sure on how to resolve the conflict or if the intent is ambiguous, please return the same snippet with the conflict.\n"
    "Here is the code snippet:\n"
)
