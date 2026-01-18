"""
Local file-based implementation of MemoryDatabaseManager.

Suitable for small-scale usage and testing.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import get_memory_data_dir, get_rrf_k
from .entity import extract_entities
from .llm import get_embedding
from .models import Memory, ScoredMemory

logger = logging.getLogger(__name__)


class LocalFileMemoryManager:
    """
    File-based MemoryDatabaseManager implementation.

    Storage structure:
        {data_dir}/{user_id}/
            memories.json    - List of Memory dicts
            embeddings.json  - {memory_id: [float, ...]}
            synonyms.json    - {canonical: [synonym, ...]}
    """

    def __init__(self, data_dir: str | None = None):
        """
        Initialize the manager.

        Args:
            data_dir: Root data directory, defaults to MEMORY_DATA_DIR env var.
        """
        self.data_dir = Path(data_dir or get_memory_data_dir())
        logger.info("LocalFileMemoryManager initialized with data_dir=%s", self.data_dir)

    def _get_user_dir(self, user_id: str) -> Path:
        """Get user-specific directory, creating if needed."""
        user_dir = self.data_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_memories_path(self, user_id: str) -> Path:
        return self._get_user_dir(user_id) / "memories.json"

    def _get_embeddings_path(self, user_id: str) -> Path:
        return self._get_user_dir(user_id) / "embeddings.json"

    def _get_synonyms_path(self, user_id: str) -> Path:
        return self._get_user_dir(user_id) / "synonyms.json"

    # ========== File I/O ==========

    def _load_memories(self, user_id: str) -> list[Memory]:
        """Load all memories for user."""
        path = self._get_memories_path(user_id)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Memory.from_dict(d) for d in data]

    def _save_memories(self, user_id: str, memories: list[Memory]) -> None:
        """Save all memories for user."""
        path = self._get_memories_path(user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in memories], f, ensure_ascii=False, indent=2)

    def _load_embeddings(self, user_id: str) -> dict[str, list[float]]:
        """Load all embeddings for user."""
        path = self._get_embeddings_path(user_id)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_embeddings(self, user_id: str, embeddings: dict[str, list[float]]) -> None:
        """Save all embeddings for user."""
        path = self._get_embeddings_path(user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)

    # ========== Protocol Methods ==========

    def search_by_vector(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """Vector similarity search using cosine similarity."""
        memories = self._load_memories(user_id)
        embeddings = self._load_embeddings(user_id)

        if not memories or not embeddings:
            return []

        # Get query embedding
        query_embedding = np.array(get_embedding(query))

        # Calculate cosine similarity for each memory
        scores = []
        for memory in memories:
            if memory.id not in embeddings:
                continue
            mem_embedding = np.array(embeddings[memory.id])
            # Cosine similarity
            similarity = np.dot(query_embedding, mem_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(mem_embedding) + 1e-10
            )
            scores.append(ScoredMemory(memory=memory, score=float(similarity)))

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]

    def search_by_entities(
        self,
        user_id: str,
        entities: list[str],
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """Keyword-based search with synonym expansion."""
        if not entities:
            return []

        memories = self._load_memories(user_id)
        if not memories:
            return []

        # Get synonyms and build reverse map (synonym -> canonical)
        synonyms = self.get_synonyms(user_id)
        synonym_to_canonical: dict[str, str] = {}
        for canonical, syns in synonyms.items():
            synonym_to_canonical[canonical.lower()] = canonical.lower()
            for syn in syns:
                synonym_to_canonical[syn.lower()] = canonical.lower()

        # Normalize query keywords
        query_keywords_normalized = set()
        for keyword in entities:
            keyword_lower = keyword.lower()
            # Map to canonical if possible
            canonical = synonym_to_canonical.get(keyword_lower, keyword_lower)
            query_keywords_normalized.add(canonical)

        # Score each memory by keyword match
        scores = []
        for memory in memories:
            # Normalize memory keywords
            mem_keywords_normalized = set()
            for keyword in memory.keywords:
                keyword_lower = keyword.lower()
                canonical = synonym_to_canonical.get(keyword_lower, keyword_lower)
                mem_keywords_normalized.add(canonical)

            # Count matches
            matches = len(query_keywords_normalized & mem_keywords_normalized)
            if matches > 0:
                score = matches / len(query_keywords_normalized)
                scores.append(ScoredMemory(memory=memory, score=score))

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]

    def get_all_keywords(self, user_id: str) -> list[str]:
        """Get all unique keywords from user's memories."""
        memories = self._load_memories(user_id)
        all_keywords = set()
        for memory in memories:
            all_keywords.update(memory.keywords)
        return list(all_keywords)

    def get_synonyms(self, user_id: str) -> dict[str, list[str]]:
        """Get synonym mapping for user."""
        path = self._get_synonyms_path(user_id)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def hybrid_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """Hybrid search with RRF fusion."""
        rrf_k = get_rrf_k()

        # Vector search
        vector_results = self.search_by_vector(user_id, query, top_k * 2)

        # Entity search
        entities = extract_entities(query)
        entity_results = self.search_by_entities(user_id, entities, top_k * 2) if entities else []

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        id_to_memory: dict[str, Memory] = {}

        for rank, scored in enumerate(vector_results):
            mid = scored.memory.id
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (rrf_k + rank + 1)
            id_to_memory[mid] = scored.memory

        for rank, scored in enumerate(entity_results):
            mid = scored.memory.id
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (rrf_k + rank + 1)
            id_to_memory[mid] = scored.memory

        # Sort by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = [
            ScoredMemory(memory=id_to_memory[mid], score=rrf_scores[mid])
            for mid in sorted_ids[:top_k]
        ]

        return results

    # ========== CRUD Methods ==========

    def add_memory(self, user_id: str, memory: Memory) -> Memory:
        """
        Add a new memory.

        - Assigns UUID if id is empty
        - Sets created_at and updated_at timestamps
        - Generates and stores embedding

        Returns:
            Memory with assigned id and timestamps.
        """
        memories = self._load_memories(user_id)
        embeddings = self._load_embeddings(user_id)

        # Assign ID if empty
        if not memory.id:
            memory.id = str(uuid.uuid4())

        # Set timestamps
        now = datetime.now(timezone.utc).isoformat()
        memory.created_at = now
        memory.updated_at = now

        # Generate embedding
        embedding = get_embedding(memory.to_str_for_dense_retrieval())
        embeddings[memory.id] = embedding

        # Save
        memories.append(memory)
        self._save_memories(user_id, memories)
        self._save_embeddings(user_id, embeddings)

        logger.info("Added memory %s for user %s", memory.id, user_id)
        return memory

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        update_dict: dict,
    ) -> Memory | None:
        """
        Update an existing memory.

        - Regenerates embedding if content changed
        - Updates updated_at timestamp

        Returns:
            Updated memory, or None if not found.
        """
        memories = self._load_memories(user_id)
        embeddings = self._load_embeddings(user_id)

        # Find memory
        memory_idx = None
        for i, m in enumerate(memories):
            if m.id == memory_id:
                memory_idx = i
                break

        if memory_idx is None:
            logger.warning("Memory %s not found for user %s", memory_id, user_id)
            return None

        memory = memories[memory_idx]
        old_content = memory.content

        # Apply updates
        allowed_fields = {"content", "keywords", "occurred_string", "occurred_at"}
        for key, value in update_dict.items():
            if key in allowed_fields:
                setattr(memory, key, value)

        # Update timestamp
        memory.updated_at = datetime.now(timezone.utc).isoformat()

        # Regenerate embedding if content changed
        if memory.content != old_content:
            embedding = get_embedding(memory.to_str_for_dense_retrieval())
            embeddings[memory_id] = embedding
            self._save_embeddings(user_id, embeddings)

        # Save
        memories[memory_idx] = memory
        self._save_memories(user_id, memories)

        logger.info("Updated memory %s for user %s", memory_id, user_id)
        return memory

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        Delete a memory.

        Returns:
            True if deleted, False if not found.
        """
        memories = self._load_memories(user_id)
        embeddings = self._load_embeddings(user_id)

        # Find and remove
        new_memories = [m for m in memories if m.id != memory_id]
        if len(new_memories) == len(memories):
            logger.warning("Memory %s not found for user %s", memory_id, user_id)
            return False

        # Remove embedding
        if memory_id in embeddings:
            del embeddings[memory_id]

        # Save
        self._save_memories(user_id, new_memories)
        self._save_embeddings(user_id, embeddings)

        logger.info("Deleted memory %s for user %s", memory_id, user_id)
        return True

    def get_memory(self, user_id: str, memory_id: str) -> Memory | None:
        """Get a single memory by ID."""
        memories = self._load_memories(user_id)
        for m in memories:
            if m.id == memory_id:
                return m
        return None

    def save_synonyms(self, user_id: str, synonyms: dict[str, list[str]]) -> None:
        """Save synonym mapping for user."""
        path = self._get_synonyms_path(user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(synonyms, f, ensure_ascii=False, indent=2)
        logger.info("Saved synonyms for user %s", user_id)

    def get_all_memories(self, user_id: str) -> list[Memory]:
        """Get all memories for user."""
        return self._load_memories(user_id)
