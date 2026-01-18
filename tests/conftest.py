"""
Pytest fixtures for memory system tests.
"""

import tempfile
from pathlib import Path

import pytest

from memory_core.local_manager import LocalFileMemoryManager
from memory_core.models import Memory, MemoryReference


# Sample test data from design document
SAMPLE_TEXT = """2024年12月20日，我和老婆小美去深圳南山区的海底捞吃火锅，她特别喜欢番茄锅底。
第二天我们去了华强北买了一台 iPhone 16 Pro，花了 8999 元。
小美说她下周要去上海出差，大概待一周左右。
对了，我最近在学 Python 的 FastAPI 框架，感觉比 Flask 好用多了。"""

SAMPLE_TEXT_ID = "test-text-001"
SAMPLE_USER_ID = "test-user-001"


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return SAMPLE_TEXT


@pytest.fixture
def sample_text_id():
    """Sample text ID."""
    return SAMPLE_TEXT_ID


@pytest.fixture
def sample_user_id():
    """Sample user ID."""
    return SAMPLE_USER_ID


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def manager(temp_data_dir):
    """LocalFileMemoryManager with temporary data directory."""
    return LocalFileMemoryManager(data_dir=temp_data_dir)


@pytest.fixture
def sample_memory(sample_user_id):
    """Sample memory for testing."""
    return Memory(
        id="mem-001",
        user_id=sample_user_id,
        content="老婆小美喜欢番茄锅底的火锅",
        entities=["小美", "番茄锅底", "火锅"],
        category="美食",
        occurred_string="2024-12-20",
        occurred_at="2024-12-20T00:00:00+00:00",
        created_at="2024-12-20T10:00:00+00:00",
        updated_at="2024-12-20T10:00:00+00:00",
        references=[
            MemoryReference(text_id="text-001", spans=[(0, 30)])
        ],
    )


@pytest.fixture
def sample_memories(sample_user_id):
    """Multiple sample memories for testing."""
    return [
        Memory(
            id="mem-001",
            user_id=sample_user_id,
            content="老婆小美喜欢番茄锅底的火锅",
            entities=["小美", "番茄锅底", "火锅"],
            category="美食",
            occurred_string="2024-12-20",
            occurred_at="2024-12-20T00:00:00+00:00",
        ),
        Memory(
            id="mem-002",
            user_id=sample_user_id,
            content="在华强北买了 iPhone 16 Pro，花了 8999 元",
            entities=["华强北", "iPhone 16 Pro"],
            category="购物",
            occurred_string="2024-12-21",
            occurred_at="2024-12-21T00:00:00+00:00",
        ),
        Memory(
            id="mem-003",
            user_id=sample_user_id,
            content="小美下周要去上海出差，大概待一周",
            entities=["小美", "上海"],
            category="出行",
            occurred_string="2024-12",
        ),
        Memory(
            id="mem-004",
            user_id=sample_user_id,
            content="正在学习 Python 的 FastAPI 框架，比 Flask 好用",
            entities=["Python", "FastAPI", "Flask"],
            category="编程",
        ),
    ]
