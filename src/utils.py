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
CACHE_DIR = Path("query_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(prompt: str) -> str:
    """Generate a unique cache key for a prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()


def load_from_cache(cache_key: str, model_name: str) -> Optional[Dict[str, str]]:
    """Load response from cache if it exists."""
    cache_file = CACHE_DIR / model_name / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(cache_key: str, response: Dict[str, str], model_name: str) -> None:
    """Save response to cache."""
    (CACHE_DIR / model_name).mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / model_name / f"{cache_key}.json"
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
    cached_response = load_from_cache(cache_key, "deepseek_cache")
    if cached_response:
        logger.info(f"Using cached response for prompt: {prompt}")
        return cached_response
    response = query_deepseek_api(prompt)
    if response:
        save_to_cache(cache_key, response, "deepseek_cache")
    return response


def cached_query_openrouter(prompt: str, model: str) -> Optional[Dict[str, str]]:
    """
    Query the specified OpenRouter model with caching.

    :param prompt: the user prompt to send
    :param model: the OpenRouter model name (e.g. "gpt-4o", "claude-2.0")
    :returns: dict with keys "prompt", "result" (and "reasoning" if available)
    """
    # Build a cache key that includes the model name
    cache_key = get_cache_key(prompt)
    # Try load from cache
    cached = load_from_cache(cache_key, model)
    if cached:
        logger.info(f"Using cached response for model={model}")
        return cached

    # Get API credentials / endpoint
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        raise ValueError("OPENROUTER_API_KEY key not set")
    if model == "o3":
        # O3 has restricted access
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    # Attempt the call with up to 3 retries
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            print(resp)
            # Extract content; reasoning may or may not be present
            content = resp.choices[0].message.content
            reasoning = getattr(resp.choices[0].message, "reasoning_content", None)

            result = {"prompt": prompt, "result": content}
            if reasoning is not None:
                result["reasoning"] = reasoning

            # Cache and return
            save_to_cache(cache_key, result, model)
            return result

        except Exception as e:
            logger.error(f"[OpenRouter] attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                raise

    # Shouldn't get here
    raise ValueError("Failed to query OpenRouter API after 3 attempts")
