"""
Entity extraction from text.
"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import get_model
from .llm import call_llm_json

logger = logging.getLogger(__name__)

# Setup Jinja2 environment
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))


def extract_entities(
    text: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    Extract entities from text.

    Supports two types of input:
    1. Long text (conversations, notes) - comprehensive extraction
    2. Short text (questions, queries) - focus on query intent

    The LLM adapts based on text characteristics.

    Args:
        text: Input text.
        model: LLM model, defaults to ENTITY_EXTRACT_MODEL or DEFAULT_MODEL.

    Returns:
        Deduplicated list of entities.
    """
    model = get_model(model, "ENTITY_EXTRACT_MODEL")
    template = jinja_env.get_template("extract_entities.j2")
    prompt = template.render(text=text)

    logger.debug("Extracting entities from text: %s...", text[:100])

    result = call_llm_json(prompt, model)
    entities = result.get("entities", [])

    # Ensure uniqueness while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)

    logger.info("Extracted %d entities", len(unique_entities))
    return unique_entities
