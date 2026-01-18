"""
Synonym generation for entities.
"""

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import get_model, get_synonym_batch_size, get_synonym_max_retries
from .llm import call_llm_json

logger = logging.getLogger(__name__)

# Setup Jinja2 environment
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))


def generate_synonyms(
    entities: list[str],
    *,
    batch_size: int | None = None,
    max_retries: int | None = None,
    model: str | None = None,
) -> dict[str, list[str]]:
    """
    Batch generate synonyms for entities.

    Note: This function only generates synonyms, it does NOT store them.
    Caller should:
    1. Decide when to call (periodically, when new entities reach threshold)
    2. Call manager.save_synonyms to store results
    3. Use synonyms in search_by_entities for expansion

    Process:
    1. Split entities into batches
    2. Call LLM for each batch
    3. Retry for missed entities
    4. Merge all batch results

    Args:
        entities: List of entities.
        batch_size: Entities per batch, defaults to SYNONYM_BATCH_SIZE (20).
        max_retries: Retry attempts for missed entities, defaults to SYNONYM_MAX_RETRIES (2).
        model: LLM model, defaults to SYNONYM_GEN_MODEL or DEFAULT_MODEL.

    Returns:
        Synonym mapping: {entity: [synonym1, synonym2, ...]}
        Entities without synonyms have empty arrays.
    """
    if not entities:
        return {}

    batch_size = batch_size or get_synonym_batch_size()
    max_retries = max_retries or get_synonym_max_retries()
    model = get_model(model, "SYNONYM_GEN_MODEL")

    logger.info("Generating synonyms for %d entities (batch_size=%d)", len(entities), batch_size)

    all_synonyms: dict[str, list[str]] = {}
    remaining_entities = list(entities)

    for retry in range(max_retries + 1):
        if not remaining_entities:
            break

        if retry > 0:
            logger.info("Retry %d for %d missed entities", retry, len(remaining_entities))

        # Process in batches
        for i in range(0, len(remaining_entities), batch_size):
            batch = remaining_entities[i:i + batch_size]
            batch_result = _generate_synonyms_batch(batch, model)
            all_synonyms.update(batch_result)

        # Find missed entities
        remaining_entities = [e for e in entities if e not in all_synonyms]

    # Ensure all entities have an entry (empty list if no synonyms)
    for entity in entities:
        if entity not in all_synonyms:
            all_synonyms[entity] = []
            logger.warning("Entity '%s' missed after all retries", entity)

    logger.info("Generated synonyms for %d entities", len(all_synonyms))
    return all_synonyms


def _generate_synonyms_batch(
    entities: list[str],
    model: str,
) -> dict[str, list[str]]:
    """
    Generate synonyms for a batch of entities.

    Args:
        entities: Batch of entities.
        model: LLM model.

    Returns:
        Synonym mapping for this batch.
    """
    template = jinja_env.get_template("generate_synonyms.j2")
    entities_json = json.dumps(entities, ensure_ascii=False)
    prompt = template.render(entities=entities_json)

    logger.debug("Generating synonyms for batch: %s", entities)

    result = call_llm_json(prompt, model)
    synonyms = result.get("synonyms", {})

    return synonyms
