"""
Memory update operations.
"""

import logging
from dataclasses import replace
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import get_model
from .llm import call_llm_json
from .models import Memory, MemoryAction, MemoryOperation

logger = logging.getLogger(__name__)

# Setup Jinja2 environment
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))


def update_memories_batch(
    text: str,
    memories: list[Memory],
    *,
    model: str | None = None,
) -> tuple[list[MemoryOperation], list[MemoryOperation]]:
    """
    Process a batch of memories for updates and deletions.

    Args:
        text: User input text.
        memories: List of memories to check for updates/deletions.
        model: LLM model, defaults to MEMORY_UPDATE_MODEL or DEFAULT_MODEL.

    Returns:
        Tuple of (update_operations, delete_operations).
    """
    if not memories:
        return [], []

    model = get_model(model, "MEMORY_UPDATE_MODEL")
    template = jinja_env.get_template("update_memories.j2")
    prompt = template.render(text=text, memories=memories)

    logger.debug("Processing %d memories for updates", len(memories))

    result = call_llm_json(prompt, model)
    operations = result.get("operations", [])

    update_ops: list[MemoryOperation] = []
    delete_ops: list[MemoryOperation] = []

    for op in operations:
        if "update_idx" in op:
            idx = op["update_idx"]
            update_dict = op.get("update_dict", {})
            if idx is not None and isinstance(idx, int) and 0 <= idx < len(memories):
                original = memories[idx]
                updated = apply_update(original, update_dict)
                update_ops.append(MemoryOperation(
                    action=MemoryAction.UPDATE,
                    memory_id=original.id,
                    update_dict=update_dict,
                    updated_memory=updated,
                ))
                logger.debug("Update operation for memory %s: %s", original.id, update_dict)
            else:
                logger.warning("Invalid update_idx %d, skipping", idx)

        elif "delete_idx" in op:
            idx = op["delete_idx"]
            if idx is not None and isinstance(idx, int) and 0 <= idx < len(memories):
                delete_ops.append(MemoryOperation(
                    action=MemoryAction.DELETE,
                    memory_id=memories[idx].id,
                ))
                logger.debug("Delete operation for memory %s", memories[idx].id)
            else:
                logger.warning("Invalid delete_idx %d, skipping", idx)

    logger.info("Batch result: %d updates, %d deletes", len(update_ops), len(delete_ops))
    return update_ops, delete_ops


def apply_update(memory: Memory, update_dict: dict) -> Memory:
    """
    Apply update_dict to memory, returning new Memory instance.

    Only updates allowed fields: content, keywords, occurred_string.

    Args:
        memory: Original memory.
        update_dict: Fields to update.

    Returns:
        New Memory instance with updates applied.
    """
    allowed_fields = {"content", "keywords", "occurred_string"}
    kwargs = {k: v for k, v in update_dict.items() if k in allowed_fields}

    if not kwargs:
        return memory

    return replace(memory, **kwargs)
