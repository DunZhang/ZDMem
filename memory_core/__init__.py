"""Memory Core - LLM 记忆系统核心模块"""

from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

from .models import Memory, ScoredMemory
from .protocol import MemoryDatabaseManager
from .local_manager import LocalFileMemoryManager
from .extract import extract_memories

__all__ = [
    "Memory",
    "ScoredMemory",
    "MemoryDatabaseManager",
    "LocalFileMemoryManager",
    "extract_memories",
]
