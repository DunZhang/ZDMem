"""
Protocol definition for memory database manager.

Callers implement this interface to connect their own storage backend.
"""

from typing import Protocol

from .models import ScoredMemory


class MemoryDatabaseManager(Protocol):
    """
    Protocol for memory storage and retrieval.

    Core module depends on this abstract interface, not concrete implementations.
    Callers implement this protocol to use their own database technology.
    """

    def search_by_vector(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """
        Vector similarity search.

        Args:
            user_id: User identifier.
            query: Query text to encode and search.
            top_k: Number of results to return.

        Returns:
            List of ScoredMemory sorted by similarity descending.
        """
        ...

    def search_by_entities(
        self,
        user_id: str,
        entities: list[str],
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """
        Entity-based exact matching search.

        Implementors should:
        1. Use get_synonyms to get the synonym table
        2. Expand input entities with synonyms before matching

        Args:
            user_id: User identifier.
            entities: List of entities to match.
            top_k: Number of results to return.

        Returns:
            List of ScoredMemory sorted by relevance descending.
            Score can be based on number of matched entities, weights, etc.
        """
        ...

    def get_all_entities(self, user_id: str) -> list[str]:
        """
        Get all unique entities from the memory store.

        Args:
            user_id: User identifier.

        Returns:
            Deduplicated list of all entities.
        """
        ...

    def get_synonyms(self, user_id: str) -> dict[str, list[str]]:
        """
        Get synonym mapping table.

        Args:
            user_id: User identifier.

        Returns:
            Mapping from canonical entity to list of synonyms.
            Example: {"老婆": ["妻子", "太太", "媳妇"]}
        """
        ...

    def hybrid_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """
        Hybrid search: vector + entity retrieval with RRF fusion.

        Used for chat-time memory retrieval. Different from process_memories retrieval:
        - This method uses RRF fusion and truncates to top_k
        - process_memories retrieval aims for comprehensive recall without RRF

        Args:
            user_id: User identifier.
            query: Query text.
            top_k: Number of results to return.

        Returns:
            RRF-fused ScoredMemory list sorted by score descending.

        Implementation notes:
            1. Call search_by_vector for vector results
            2. Extract entities from query, call search_by_entities
            3. Fuse with RRF: score = 1/(k+rank1) + 1/(k+rank2)
               k from RRF_K env var, default 60
        """
        ...
