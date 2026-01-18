"""
Search string generation for memory retrieval.
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


def generate_search_strings(
    text: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    Generate multiple search strings from user text.

    Key points:
    - Must cover all topics and information points in the text
    - Prefer redundancy over missing information
    - Does NOT include the original text (caller adds it as fallback)

    Args:
        text: User input text.
        model: LLM model, defaults to SEARCH_STRING_GEN_MODEL or DEFAULT_MODEL.

    Returns:
        List of search strings.
    """
    model = get_model(model, "SEARCH_STRING_GEN_MODEL")
    template = jinja_env.get_template("generate_search_strings.j2")
    prompt = template.render(text=text)

    logger.debug("Generating search strings for text: %s...", text[:100])

    result = call_llm_json(prompt, model)
    search_strings = result.get("search_strings", [])

    logger.info("Generated %d search strings", len(search_strings))
    return search_strings
