"""
Data structures for the memory system.
"""

from dataclasses import dataclass, field
from enum import StrEnum
import json


class MemoryAction(StrEnum):
    """Memory operation type."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class MemoryReference:
    """Memory source reference."""
    text_id: str
    lines: list[int]  # Line numbers (1-based) from source text


@dataclass
class Memory:
    """Memory entity."""
    user_id: str
    content: str
    id: str = ""  # Empty for new memories, assigned by caller
    keywords: list[str] = field(default_factory=list)  # Keywords for search and indexing
    occurred_string: str | None = None  # Partial time string, e.g., "2024-12-25", "2024-12"
    occurred_at: str | None = None  # ISO 8601 format
    created_at: str = ""  # ISO 8601 format
    updated_at: str = ""  # ISO 8601 format
    references: list[MemoryReference] = field(default_factory=list)

    def to_str_for_dense_retrieval(self) -> str:
        """
        Return string for vector encoding retrieval.

        Returns:
            The content field.
        """
        return self.content

    def to_str_for_update(self) -> str:
        """
        Return JSON string for update judgment in prompts.

        Only includes content, keywords, occurred_string fields.

        Returns:
            JSON formatted string.
        """
        return json.dumps({
            "content": self.content,
            "keywords": self.keywords,
            "occurred_string": self.occurred_string,
        }, ensure_ascii=False, indent=2)

    def to_str_for_insert(self) -> str:
        """
        Return JSON string for insertion deduplication in prompts.

        Only includes content and occurred_string fields.

        Returns:
            JSON formatted string.
        """
        return json.dumps({
            "content": self.content,
            "occurred_string": self.occurred_string,
        }, ensure_ascii=False, indent=2)

    def get_keywords(self) -> list[str]:
        """
        Return all keywords related to this memory.

        Returns:
            Copy of the keywords list.
        """
        return self.keywords.copy()

    def to_dict(self) -> dict:
        """
        Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation of the memory.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "keywords": self.keywords,
            "occurred_string": self.occurred_string,
            "occurred_at": self.occurred_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "references": [
                {"text_id": ref.text_id, "lines": ref.lines}
                for ref in self.references
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """
        Deserialize from dictionary.

        Supports backward compatibility with old format (entities, category, spans).

        Args:
            data: Dictionary representation of a memory.

        Returns:
            Memory instance.
        """
        # Parse references with backward compatibility
        references = []
        for ref in data.get("references", []):
            text_id = ref["text_id"]
            # Support both new 'lines' and old 'spans' format
            if "lines" in ref:
                lines = ref["lines"]
            elif "spans" in ref:
                # Convert spans to line numbers (use first element of each span as line)
                # This is a fallback for old data
                lines = [s[0] for s in ref["spans"]]
            else:
                lines = []
            references.append(MemoryReference(text_id=text_id, lines=lines))

        # Support both 'keywords' and old 'entities' field
        keywords = data.get("keywords", data.get("entities", []))

        return cls(
            id=data.get("id", ""),
            user_id=data["user_id"],
            content=data["content"],
            keywords=keywords,
            occurred_string=data.get("occurred_string"),
            occurred_at=data.get("occurred_at"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            references=references,
        )


@dataclass
class ScoredMemory:
    """Memory with retrieval score."""
    memory: Memory
    score: float


@dataclass
class MemoryOperation:
    """
    Unified representation for CREATE, UPDATE, DELETE operations.

    For CREATE: created_memory is set, memory_id is empty
    For UPDATE: memory_id, update_dict, and updated_memory are set
    For DELETE: only memory_id is set
    """
    action: MemoryAction
    memory_id: str = ""  # Target memory ID, empty for CREATE
    # Used for CREATE
    created_memory: Memory | None = None
    # Used for UPDATE
    update_dict: dict | None = None  # Only fields that changed
    updated_memory: Memory | None = None  # Complete memory after update


@dataclass
class MemoryOperationResult:
    """Result of memory processing."""
    created: list[MemoryOperation]  # action=CREATE operations
    updated: list[MemoryOperation]  # action=UPDATE operations
    deleted: list[MemoryOperation]  # action=DELETE operations
