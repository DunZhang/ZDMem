"""
Tests for memory update operations.
"""

import pytest

from memory_core.models import Memory, MemoryAction
from memory_core.update import apply_update, update_memories_batch


class TestApplyUpdate:
    """Tests for apply_update function."""

    def test_update_content(self, sample_memory):
        """Test updating content field."""
        update_dict = {"content": "新的内容"}
        result = apply_update(sample_memory, update_dict)

        assert result.content == "新的内容"
        # Other fields should remain unchanged
        assert result.entities == sample_memory.entities
        assert result.category == sample_memory.category

    def test_update_multiple_fields(self, sample_memory):
        """Test updating multiple fields."""
        update_dict = {
            "content": "更新的内容",
            "entities": ["新实体1", "新实体2"],
            "category": "新分类",
        }
        result = apply_update(sample_memory, update_dict)

        assert result.content == "更新的内容"
        assert result.entities == ["新实体1", "新实体2"]
        assert result.category == "新分类"

    def test_update_ignores_disallowed_fields(self, sample_memory):
        """Test that disallowed fields are ignored."""
        update_dict = {
            "content": "新内容",
            "id": "new-id",  # Should be ignored
            "created_at": "2099-01-01",  # Should be ignored
        }
        result = apply_update(sample_memory, update_dict)

        assert result.content == "新内容"
        assert result.id == sample_memory.id  # Unchanged
        assert result.created_at == sample_memory.created_at  # Unchanged

    def test_empty_update_dict(self, sample_memory):
        """Test with empty update dict."""
        result = apply_update(sample_memory, {})
        assert result == sample_memory

    def test_update_occurred_string(self, sample_memory):
        """Test updating occurred_string."""
        update_dict = {"occurred_string": "2025-01"}
        result = apply_update(sample_memory, update_dict)

        assert result.occurred_string == "2025-01"


class TestUpdateMemoriesBatch:
    """Tests for update_memories_batch function (real LLM calls)."""

    def test_batch_with_no_changes_needed(self, sample_memories):
        """Test batch where no changes are needed."""
        text = "今天天气真好，适合出去走走"  # Unrelated text
        update_ops, delete_ops = update_memories_batch(text, sample_memories)

        # Both lists should be valid (may be empty)
        assert isinstance(update_ops, list)
        assert isinstance(delete_ops, list)

    def test_batch_with_update(self, sample_memories):
        """Test batch with potential update."""
        # Text that corrects existing memory
        text = "小美其实更喜欢清汤锅底，不是番茄锅底"
        update_ops, delete_ops = update_memories_batch(text, sample_memories)

        # LLM may or may not decide to update
        assert isinstance(update_ops, list)
        assert isinstance(delete_ops, list)

        # If there are updates, check structure
        for op in update_ops:
            assert op.action == MemoryAction.UPDATE
            assert op.memory_id != ""
            assert op.update_dict is not None
            assert op.updated_memory is not None

    def test_batch_with_deletion_signal(self, sample_memories):
        """Test batch with deletion signal."""
        # Text that invalidates existing memory
        text = "我之前说的 iPhone 购买信息是错误的，请忘记它"
        update_ops, delete_ops = update_memories_batch(text, sample_memories)

        # LLM may or may not decide to delete
        assert isinstance(update_ops, list)
        assert isinstance(delete_ops, list)

        # If there are deletions, check structure
        for op in delete_ops:
            assert op.action == MemoryAction.DELETE
            assert op.memory_id != ""

    def test_empty_memories_list(self):
        """Test with empty memories list."""
        text = "一些文本"
        update_ops, delete_ops = update_memories_batch(text, [])

        assert update_ops == []
        assert delete_ops == []
