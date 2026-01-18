"""
ZDMem - LLM Memory System

Core module for managing user memories with hybrid search (vector + entity-based).
"""

from .models import (
    Memory,
    MemoryAction,
    MemoryOperation,
    MemoryOperationResult,
    MemoryReference,
    ScoredMemory,
)
from .protocol import MemoryDatabaseManager
from .main import process_memories
from .entity import extract_entities
from .search import generate_search_strings
from .synonym import generate_synonyms

__all__ = [
    # Data structures
    "Memory",
    "MemoryAction",
    "MemoryOperation",
    "MemoryOperationResult",
    "MemoryReference",
    "ScoredMemory",
    # Protocol
    "MemoryDatabaseManager",
    # Core functions
    "process_memories",
    "extract_entities",
    "generate_search_strings",
    "generate_synonyms",
]
