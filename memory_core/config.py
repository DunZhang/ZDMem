"""
Configuration management for the memory system.

Loads settings from environment variables with sensible defaults.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


@lru_cache(maxsize=1)
def get_default_model() -> str:
    """Get the default LLM model."""
    return os.getenv("DEFAULT_MODEL", "openrouter/google/gemini-3-flash-preview")


def get_model(param_value: str | None, env_var: str) -> str:
    """
    Select model with priority: param > env_var > DEFAULT_MODEL.

    Args:
        param_value: Model specified as function parameter.
        env_var: Environment variable name for this function's model.

    Returns:
        Selected model identifier.
    """
    if param_value:
        return param_value
    env_value = os.getenv(env_var)
    if env_value:
        return env_value
    return get_default_model()


def get_int_config(env_var: str, default: int) -> int:
    """
    Get integer configuration from environment variable.

    Args:
        env_var: Environment variable name.
        default: Default value if not set.

    Returns:
        Integer configuration value.
    """
    value = os.getenv(env_var)
    return int(value) if value else default


def get_search_top_k() -> int:
    """Get number of results per search string."""
    return get_int_config("SEARCH_TOP_K", 10)


def get_update_batch_size() -> int:
    """Get batch size for memory updates."""
    return get_int_config("UPDATE_BATCH_SIZE", 10)


def get_synonym_batch_size() -> int:
    """Get batch size for synonym generation."""
    return get_int_config("SYNONYM_BATCH_SIZE", 20)


def get_synonym_max_retries() -> int:
    """Get max retries for synonym generation."""
    return get_int_config("SYNONYM_MAX_RETRIES", 2)


def get_rrf_k() -> int:
    """Get RRF fusion parameter k."""
    return get_int_config("RRF_K", 60)


def get_embedding_model() -> str:
    """Get embedding model identifier."""
    return os.getenv("EMBEDDING_MODEL", "voyage/voyage-4")


def get_memory_data_dir() -> str:
    """Get data storage directory path."""
    return os.getenv("MEMORY_DATA_DIR", "./data")
