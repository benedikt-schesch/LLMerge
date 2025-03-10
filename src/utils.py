# -*- coding: utf-8 -*-
"""Utility functions for the project."""

from typing import Optional
import re
import os

JAVA_MARKDOWN_RE = re.compile(r"```java\n(.*?)\n```", re.DOTALL)

# For normalizing Java code
BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/")
LINE_COMMENT_RE = re.compile(r"//.*")
WHITESPACE_RE = re.compile(r"\s+")

CONFLICT_MARKERS = ["<<<<<<<", "=======", "|||||||", ">>>>>>>"]


def get_num_workers(n_threads: Optional[int] = None) -> int:
    """Get the number of workers for parallel processing."""
    if n_threads is not None and n_threads > 0:
        return n_threads
    os_cpu_count = os.cpu_count()
    if os_cpu_count is None:
        return 1
    return os_cpu_count - 1


def normalize_java_code(code: str) -> str:
    """
    Normalizes Java code by removing block comments, line comments,
    and extra whitespace (so we focus on core semantics).
    """
    code = BLOCK_COMMENT_RE.sub("", code)
    code = LINE_COMMENT_RE.sub("", code)
    code = WHITESPACE_RE.sub(" ", code)
    return code.strip()


def extract_code_block(text: str) -> Optional[str]:
    """
    Extracts the code block from a markdown-formatted text:
       ```java
       ... some code ...
       ```
    Returns None if there's no Java code block.
    """
    match = JAVA_MARKDOWN_RE.search(text)
    return match.group(1).strip() if match else None
