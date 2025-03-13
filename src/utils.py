# -*- coding: utf-8 -*-
"""Utility functions for the project."""

from typing import Optional, Dict
import re
import os
import hashlib
import json
from pathlib import Path
import time
from openai import OpenAI
from loguru import logger

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


DEEPSEEK_API_URL = "https://api.deepseek.com"
CACHE_DIR = Path("deepseek_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(prompt: str) -> str:
    """Generate a unique cache key for a prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()


def load_from_cache(cache_key: str) -> Optional[Dict[str, str]]:
    """Load response from cache if it exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(cache_key: str, response: Dict[str, str]) -> None:
    """Save response to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(response, file, indent=4)


def query_deepseek_api(prompt: str) -> Optional[Dict[str, str]]:
    """Query the DeepSeek R1 API for conflict resolution with retries."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        raise ValueError("DEEPSEEK_API_KEY key not set")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_API_URL)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],  # type: ignore
                stream=False,
            )

            reasoning = response.choices[0].message.reasoning_content  # type: ignore
            result = response.choices[0].message.content

            if reasoning is None or result is None:
                raise ValueError("Response is missing reasoning or content")

            return {"prompt": prompt, "reasoning": reasoning, "result": result}

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)  # Short delay before retry
            else:
                raise  # Raise exception after 3 failed attempts
    raise ValueError("Failed to query DeepSeek API after 3 attempts")


def cached_query_deepseek_api(prompt: str) -> Optional[Dict[str, str]]:
    """Query the DeepSeek R1 API with caching."""
    cache_key = get_cache_key(prompt)
    cached_response = load_from_cache(cache_key)
    if cached_response:
        logger.info(f"Using cached response for prompt: {prompt}")
        return cached_response
    response = query_deepseek_api(prompt)
    if response:
        save_to_cache(cache_key, response)
    return response
