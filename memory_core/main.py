"""
Main entry point for memory processing.
"""

import logging

from .config import get_search_top_k, get_update_batch_size
from .entity import extract_entities
from .insert import insert_memories
from .models import Memory, MemoryOperationResult
from .protocol import MemoryDatabaseManager
from .search import generate_search_strings
from .update import update_memories_batch

logger = logging.getLogger(__name__)


def process_memories(
    text: str,
    text_id: str,
    user_id: str,
    manager: MemoryDatabaseManager,
    *,
    search_top_k: int | None = None,
    update_batch_size: int | None = None,
    search_string_gen_model: str | None = None,
    memory_update_model: str | None = None,
    memory_insert_model: str | None = None,
    entity_extract_model: str | None = None,
) -> MemoryOperationResult:
    """
    Process text to create, update, and delete memories.

    Flow:
    1. Generate search strings from text
    2. Hybrid retrieval for each search string (vector + entity)
    3. Merge and sort results by occurred_at
    4. Batch update/delete existing memories
    5. Build context from updated memories
    6. Insert new memories

    Args:
        text: Input text (conversation, notes, etc.).
        text_id: Unique text identifier for source tracking.
        user_id: User identifier.
        manager: Memory database manager implementation.
        search_top_k: Results per search string, defaults to SEARCH_TOP_K (10).
        update_batch_size: Memories per update batch, defaults to UPDATE_BATCH_SIZE (10).
        search_string_gen_model: Model for search string generation.
        memory_update_model: Model for memory updates.
        memory_insert_model: Model for memory insertion.
        entity_extract_model: Model for entity extraction.

    Returns:
        MemoryOperationResult with created, updated, and deleted operations.
    """
    search_top_k = search_top_k or get_search_top_k()
    update_batch_size = update_batch_size or get_update_batch_size()

    logger.info("Processing memories for text_id=%s, user_id=%s", text_id, user_id)

    # Step 1: Generate search strings
    logger.debug("Step 1: Generating search strings")
    search_strings = generate_search_strings(text, model=search_string_gen_model)
    search_strings.append(text)  # Add original text as fallback
    logger.info("Generated %d search strings (including original)", len(search_strings))

    # Step 2-3: Hybrid retrieval and merge
    logger.debug("Step 2-3: Hybrid retrieval")
    related_memories = _retrieve_related_memories(
        user_id=user_id,
        search_strings=search_strings,
        top_k=search_top_k,
        manager=manager,
        entity_extract_model=entity_extract_model,
    )
    logger.info("Retrieved %d related memories", len(related_memories))

    # Step 4: Batch update/delete
    logger.debug("Step 4: Batch update/delete")
    all_update_ops = []
    all_delete_ops = []
    updated_ids = set()
    deleted_ids = set()
    id_to_updated_memory = {}

    for i in range(0, len(related_memories), update_batch_size):
        batch = related_memories[i:i + update_batch_size]
        update_ops, delete_ops = update_memories_batch(
            text, batch, model=memory_update_model
        )

        for op in update_ops:
            all_update_ops.append(op)
            updated_ids.add(op.memory_id)
            if op.updated_memory:
                id_to_updated_memory[op.memory_id] = op.updated_memory

        for op in delete_ops:
            all_delete_ops.append(op)
            deleted_ids.add(op.memory_id)

    logger.info("Update step: %d updates, %d deletes", len(all_update_ops), len(all_delete_ops))

    # Step 5: Build context for insertion
    logger.debug("Step 5: Building insertion context")
    context_memories = []
    for memory in related_memories:
        if memory.id in deleted_ids:
            continue  # Skip deleted
        if memory.id in updated_ids:
            # Use updated version
            context_memories.append(id_to_updated_memory[memory.id])
        else:
            context_memories.append(memory)

    # Step 6: Insert new memories
    logger.debug("Step 6: Inserting new memories")
    create_ops = insert_memories(
        text=text,
        text_id=text_id,
        user_id=user_id,
        existing_memories=context_memories,
        model=memory_insert_model,
    )
    logger.info("Created %d new memories", len(create_ops))

    result = MemoryOperationResult(
        created=create_ops,
        updated=all_update_ops,
        deleted=all_delete_ops,
    )

    logger.info(
        "Processing complete: %d created, %d updated, %d deleted",
        len(result.created), len(result.updated), len(result.deleted)
    )

    return result


def _retrieve_related_memories(
    user_id: str,
    search_strings: list[str],
    top_k: int,
    manager: MemoryDatabaseManager,
    entity_extract_model: str | None,
) -> list[Memory]:
    """
    Retrieve related memories using hybrid search for each search string.

    For process_memories, we want comprehensive recall without RRF fusion.
    Different from hybrid_search which is for chat-time query.

    Args:
        user_id: User identifier.
        search_strings: List of search strings.
        top_k: Results per search.
        manager: Memory database manager.
        entity_extract_model: Model for entity extraction.

    Returns:
        Deduplicated list of memories sorted by occurred_at ascending.
    """
    seen_ids = set()
    all_memories: list[Memory] = []

    for search_string in search_strings:
        # Vector search
        vector_results = manager.search_by_vector(user_id, search_string, top_k)
        for scored in vector_results:
            if scored.memory.id not in seen_ids:
                seen_ids.add(scored.memory.id)
                all_memories.append(scored.memory)

        # Entity search
        entities = extract_entities(search_string, model=entity_extract_model)
        if entities:
            entity_results = manager.search_by_entities(user_id, entities, top_k)
            for scored in entity_results:
                if scored.memory.id not in seen_ids:
                    seen_ids.add(scored.memory.id)
                    all_memories.append(scored.memory)

    # Sort by occurred_at ascending (earlier first), None at the end
    def sort_key(m: Memory) -> tuple:
        if m.occurred_at is None:
            return (1, "")  # None goes last
        return (0, m.occurred_at)

    all_memories.sort(key=sort_key)

    return all_memories
