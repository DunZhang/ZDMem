"""
LLM wrapper with retry logic and JSON parsing.
"""

import json
import logging
from typing import Any

import litellm
from json_repair import repair_json

from .config import get_embedding_model

logger = logging.getLogger(__name__)


def call_llm(
    prompt: str,
    model: str,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> str:
    """
    Call LLM with retry logic and JSON mode.

    Args:
        prompt: The prompt to send.
        model: Model identifier.
        max_retries: Number of retry attempts.
        temperature: Sampling temperature.

    Returns:
        Raw response content from LLM.

    Raises:
        RuntimeError: If all retries fail.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            logger.debug("LLM response: %s", content)
            return content
        except Exception as e:
            last_error = e
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt + 1, max_retries, e
            )
            if attempt == max_retries - 1:
                raise RuntimeError(f"LLM call failed after {max_retries} retries") from last_error
    raise RuntimeError("LLM call failed after all retries")


def parse_json_response(response: str) -> Any:
    """
    Parse LLM JSON response, handling code blocks and malformed JSON.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed JSON object.
    """
    text = response.strip()

    # Remove markdown code block markers
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to repair malformed JSON
        repaired = repair_json(text)
        return json.loads(repaired)


def call_llm_json(
    prompt: str,
    model: str,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> Any:
    """
    Call LLM and parse JSON response.

    Args:
        prompt: The prompt to send.
        model: Model identifier.
        max_retries: Number of retry attempts.
        temperature: Sampling temperature.

    Returns:
        Parsed JSON object from LLM response.
    """
    response = call_llm(prompt, model, max_retries, temperature)
    return parse_json_response(response)


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """
    Get embedding vector for text.

    Args:
        text: Text to embed.
        model: Embedding model identifier, defaults to EMBEDDING_MODEL env var.

    Returns:
        Embedding vector as list of floats.
    """
    model = model or get_embedding_model()
    response = litellm.embedding(model=model, input=[text])
    return response.data[0]["embedding"]


def get_embeddings_batch(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """
    Get embedding vectors for multiple texts in a single API call.

    Args:
        texts: List of texts to embed.
        model: Embedding model identifier, defaults to EMBEDDING_MODEL env var.

    Returns:
        List of embedding vectors, in same order as input texts.
    """
    if not texts:
        return []

    model = model or get_embedding_model()
    response = litellm.embedding(model=model, input=texts)

    # Sort by index to ensure correct order
    sorted_data = sorted(response.data, key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]
