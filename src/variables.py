# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""This file contains the global variables used in the project."""

MAX_NUM_MERGES = 100

MODEL = "unsloth/deepSeek-r1-distill-qwen-7b"  # Model to use for generation
MAX_SEQ_LENGTH = 2048  # Maximum number of tokens of the entire sequence
MAX_PROMPT_LENGTH = 256  # Maximum number of tokens in the prompt
LORA_RANK = 64  # Larger rank = smarter, but slower

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed within <think> </think> followed by the answer, i.e., "
    "<think> reasoning process here </think> answer here"
)

QUERY_PROMPT = (
    "You are a semantic merge conflict resolution expert. Below is a snippet of code "
    "with surrounding context that includes a merge conflict.\n"
    "Return the entire snippet (including full context) in markdown code fences as provided, make sure you do not modify the context at all and preserve the spacing as is.\n"
    "Think in terms of intent and semantics that both sides of the merge are trying to achieve.\n"
    "If you are not sure on how to resolve the conflict or if the intent is ambiguous, please return the same snippet with the conflict.\n"
    "Here is the code snippet:\n"
)
