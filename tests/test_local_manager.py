"""
Tests for LocalFileMemoryManager.
"""

import pytest

from memory_core.local_manager import LocalFileMemoryManager
from memory_core.models import Memory


class TestLocalFileMemoryManagerCRUD:
    """Tests for CRUD operations."""

    def test_add_memory(self, manager, sample_user_id):
        """Test adding a memory."""
        memory = Memory(
            user_id=sample_user_id,
            content="测试记忆内容",
            entities=["测试"],
            category="测试",
        )

        result = manager.add_memory(sample_user_id, memory)

        assert result.id != ""  # Should have assigned ID
        assert result.created_at != ""
        assert result.updated_at != ""
        assert result.content == "测试记忆内容"

    def test_get_memory(self, manager, sample_user_id):
        """Test getting a memory by ID."""
        memory = Memory(
            user_id=sample_user_id,
            content="测试获取记忆",
        )
        added = manager.add_memory(sample_user_id, memory)

        result = manager.get_memory(sample_user_id, added.id)

        assert result is not None
        assert result.id == added.id
        assert result.content == "测试获取记忆"

    def test_get_memory_not_found(self, manager, sample_user_id):
        """Test getting non-existent memory."""
        result = manager.get_memory(sample_user_id, "non-existent-id")
        assert result is None

    def test_update_memory(self, manager, sample_user_id):
        """Test updating a memory."""
        memory = Memory(
            user_id=sample_user_id,
            content="原始内容",
            entities=["原始"],
        )
        added = manager.add_memory(sample_user_id, memory)

        result = manager.update_memory(
            sample_user_id,
            added.id,
            {"content": "更新后的内容", "entities": ["更新"]},
        )

        assert result is not None
        assert result.content == "更新后的内容"
        assert result.entities == ["更新"]
        assert result.updated_at != added.updated_at

    def test_update_memory_not_found(self, manager, sample_user_id):
        """Test updating non-existent memory."""
        result = manager.update_memory(
            sample_user_id,
            "non-existent-id",
            {"content": "新内容"},
        )
        assert result is None

    def test_delete_memory(self, manager, sample_user_id):
        """Test deleting a memory."""
        memory = Memory(
            user_id=sample_user_id,
            content="待删除的记忆",
        )
        added = manager.add_memory(sample_user_id, memory)

        result = manager.delete_memory(sample_user_id, added.id)
        assert result is True

        # Should not be found after deletion
        assert manager.get_memory(sample_user_id, added.id) is None

    def test_delete_memory_not_found(self, manager, sample_user_id):
        """Test deleting non-existent memory."""
        result = manager.delete_memory(sample_user_id, "non-existent-id")
        assert result is False

    def test_get_all_memories(self, manager, sample_user_id):
        """Test getting all memories."""
        # Add multiple memories
        for i in range(3):
            memory = Memory(
                user_id=sample_user_id,
                content=f"记忆 {i}",
            )
            manager.add_memory(sample_user_id, memory)

        memories = manager.get_all_memories(sample_user_id)
        assert len(memories) == 3


class TestLocalFileMemoryManagerSearch:
    """Tests for search operations."""

    def test_search_by_vector(self, manager, sample_user_id):
        """Test vector similarity search."""
        # Add memories
        memory1 = Memory(
            user_id=sample_user_id,
            content="我喜欢吃川菜，特别是麻婆豆腐",
            entities=["川菜", "麻婆豆腐"],
        )
        memory2 = Memory(
            user_id=sample_user_id,
            content="今天学习了 Python 编程",
            entities=["Python", "编程"],
        )
        manager.add_memory(sample_user_id, memory1)
        manager.add_memory(sample_user_id, memory2)

        # Search for food-related content
        results = manager.search_by_vector(sample_user_id, "川菜和火锅", top_k=2)

        assert len(results) > 0
        # Food-related memory should rank higher
        assert "川菜" in results[0].memory.content or "麻婆" in results[0].memory.content

    def test_search_by_entities(self, manager, sample_user_id):
        """Test entity-based search."""
        memory1 = Memory(
            user_id=sample_user_id,
            content="和小美一起吃火锅",
            entities=["小美", "火锅"],
        )
        memory2 = Memory(
            user_id=sample_user_id,
            content="小美喜欢番茄锅底",
            entities=["小美", "番茄锅底"],
        )
        manager.add_memory(sample_user_id, memory1)
        manager.add_memory(sample_user_id, memory2)

        results = manager.search_by_entities(sample_user_id, ["小美"], top_k=10)

        assert len(results) == 2
        for r in results:
            assert "小美" in r.memory.entities

    def test_search_by_entities_with_synonyms(self, manager, sample_user_id):
        """Test entity search with synonym expansion."""
        # Add memory with "老婆"
        memory = Memory(
            user_id=sample_user_id,
            content="老婆喜欢吃火锅",
            entities=["老婆", "火锅"],
        )
        manager.add_memory(sample_user_id, memory)

        # Save synonyms
        manager.save_synonyms(sample_user_id, {
            "老婆": ["妻子", "太太", "媳妇"],
        })

        # Search with synonym "妻子"
        results = manager.search_by_entities(sample_user_id, ["妻子"], top_k=10)

        assert len(results) == 1
        assert results[0].memory.content == "老婆喜欢吃火锅"

    def test_get_all_entities(self, manager, sample_user_id):
        """Test getting all entities."""
        memory1 = Memory(
            user_id=sample_user_id,
            content="记忆1",
            entities=["实体A", "实体B"],
        )
        memory2 = Memory(
            user_id=sample_user_id,
            content="记忆2",
            entities=["实体B", "实体C"],
        )
        manager.add_memory(sample_user_id, memory1)
        manager.add_memory(sample_user_id, memory2)

        entities = manager.get_all_entities(sample_user_id)

        assert set(entities) == {"实体A", "实体B", "实体C"}

    def test_hybrid_search(self, manager, sample_user_id):
        """Test hybrid search with RRF fusion."""
        memory1 = Memory(
            user_id=sample_user_id,
            content="小美喜欢吃番茄锅底的火锅",
            entities=["小美", "番茄锅底", "火锅"],
        )
        memory2 = Memory(
            user_id=sample_user_id,
            content="今天去了海底捞吃火锅",
            entities=["海底捞", "火锅"],
        )
        manager.add_memory(sample_user_id, memory1)
        manager.add_memory(sample_user_id, memory2)

        results = manager.hybrid_search(sample_user_id, "小美喜欢什么火锅", top_k=2)

        assert len(results) > 0
        # Results should be sorted by RRF score


class TestLocalFileMemoryManagerSynonyms:
    """Tests for synonym operations."""

    def test_save_and_get_synonyms(self, manager, sample_user_id):
        """Test saving and retrieving synonyms."""
        synonyms = {
            "老婆": ["妻子", "太太"],
            "手机": ["电话", "移动电话"],
        }
        manager.save_synonyms(sample_user_id, synonyms)

        result = manager.get_synonyms(sample_user_id)

        assert result == synonyms

    def test_get_synonyms_empty(self, manager, sample_user_id):
        """Test getting synonyms when none exist."""
        result = manager.get_synonyms(sample_user_id)
        assert result == {}
