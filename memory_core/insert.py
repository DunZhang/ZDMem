"""
Memory insertion operations.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import get_model
from .llm import call_llm_json
from .models import Memory, MemoryAction, MemoryOperation, MemoryReference

logger = logging.getLogger(__name__)

# Setup Jinja2 environment
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))


def add_line_numbers(text: str) -> str:
    """
    Add line number prefixes to each line of text.

    Args:
        text: Input text.

    Returns:
        Text with line numbers, e.g., "[1]: first line\\n[2]: second line"
    """
    lines = text.split('\n')
    return '\n'.join(f"[{i+1}]: {line}" for i, line in enumerate(lines))


def insert_memories(
    text: str,
    text_id: str,
    user_id: str,
    existing_memories: list[Memory],
    *,
    model: str | None = None,
) -> list[MemoryOperation]:
    """
    Extract new memories from text, avoiding duplicates with existing memories.

    Args:
        text: User input text.
        text_id: Text identifier for reference tracking.
        user_id: User identifier.
        existing_memories: Already existing memories (for deduplication).
        model: LLM model, defaults to MEMORY_INSERT_MODEL or DEFAULT_MODEL.

    Returns:
        List of CREATE operations.
    """
    model = get_model(model, "MEMORY_INSERT_MODEL")

    # Add line numbers to text for reference tracking
    numbered_text = add_line_numbers(text)

    template = jinja_env.get_template("insert_memories.j2")
    prompt = template.render(
        text=numbered_text,
        text_id=text_id,
        existing_memories=existing_memories,
    )

    logger.debug("Extracting new memories from text: %s...", text[:100])

    result = call_llm_json(prompt, model)
    new_memories = result.get("memories", [])

    create_ops: list[MemoryOperation] = []

    for mem_data in new_memories:
        memory = _create_memory_from_dict(mem_data, text_id, user_id)
        create_ops.append(MemoryOperation(
            action=MemoryAction.CREATE,
            created_memory=memory,
        ))

    logger.info("Created %d new memory operations", len(create_ops))
    return create_ops


def _create_memory_from_dict(
    data: dict,
    text_id: str,
    user_id: str,
) -> Memory:
    """
    Create Memory instance from LLM output dict.

    Handles:
    - occurred_string to occurred_at conversion
    - Reference building from lines

    Args:
        data: Dict from LLM with content, keywords, occurred_string, lines.
        text_id: Text identifier.
        user_id: User identifier.

    Returns:
        New Memory instance (without id, caller assigns it).
    """
    content = data.get("content", "")
    keywords = data.get("keywords", [])
    occurred_string = data.get("occurred_string")
    lines = data.get("lines", [])

    # Build reference
    references = []
    if lines:
        references.append(MemoryReference(text_id=text_id, lines=lines))

    # Convert occurred_string to occurred_at
    occurred_at = _parse_occurred_string(occurred_string) if occurred_string else None

    return Memory(
        user_id=user_id,
        content=content,
        keywords=keywords,
        occurred_string=occurred_string,
        occurred_at=occurred_at,
        references=references,
    )


def _parse_occurred_string(occurred_string: str) -> str | None:
    """
    Parse partial time string to full ISO 8601 format.

    Supports formats:
    - "2024" -> "2024-01-01T00:00:00+00:00"
    - "2024-12" -> "2024-12-01T00:00:00+00:00"
    - "2024-12-25" -> "2024-12-25T00:00:00+00:00"
    - "2024-12-25T03" -> "2024-12-25T03:00:00+00:00"
    - "2024-12-25T03:30" -> "2024-12-25T03:30:00+00:00"

    Args:
        occurred_string: Partial time string.

    Returns:
        ISO 8601 formatted string, or None if parsing fails.
    """
    if not occurred_string:
        return None

    try:
        # Try various formats from most specific to least
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%dT%H",
            "%Y-%m-%d",
            "%Y-%m",
            "%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(occurred_string, fmt)
                # Set timezone to UTC
                dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                continue

        logger.warning("Could not parse occurred_string: %s", occurred_string)
        return None

    except Exception as e:
        logger.warning("Error parsing occurred_string '%s': %s", occurred_string, e)
        return None
