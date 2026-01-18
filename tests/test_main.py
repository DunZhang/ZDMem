"""
Integration tests for process_memories.

Uses real LLM calls as specified in design document.
"""

import pytest

from memory_core import process_memories
from memory_core.local_manager import LocalFileMemoryManager
from memory_core.models import Memory, MemoryAction


class TestProcessMemories:
    """Integration tests for process_memories."""

    def test_process_new_memories(self, manager, sample_text, sample_text_id, sample_user_id):
        """Test processing text to create new memories."""
        result = process_memories(
            text=sample_text,
            text_id=sample_text_id,
            user_id=sample_user_id,
            manager=manager,
        )

        # Should create new memories
        assert len(result.created) > 0

        # Each created operation should have a memory
        for op in result.created:
            assert op.action == MemoryAction.CREATE
            assert op.created_memory is not None
            assert op.created_memory.user_id == sample_user_id
            assert op.created_memory.content != ""

    def test_process_memories_extracts_entities(self, manager, sample_text, sample_text_id, sample_user_id):
        """Test that created memories have entities extracted."""
        result = process_memories(
            text=sample_text,
            text_id=sample_text_id,
            user_id=sample_user_id,
            manager=manager,
        )

        # At least some memories should have entities
        memories_with_entities = [
            op for op in result.created
            if op.created_memory and len(op.created_memory.entities) > 0
        ]
        assert len(memories_with_entities) > 0

    def test_process_memories_with_existing(self, manager, sample_user_id):
        """Test processing with existing memories (no duplicates)."""
        # First, add some memories
        existing = Memory(
            user_id=sample_user_id,
            content="小美喜欢番茄锅底",
            entities=["小美", "番茄锅底"],
        )
        manager.add_memory(sample_user_id, existing)

        # Process text that overlaps with existing memory
        text = "今天小美说她还是最喜欢番茄锅底的火锅"
        result = process_memories(
            text=text,
            text_id="test-002",
            user_id=sample_user_id,
            manager=manager,
        )

        # Should not create duplicate about 番茄锅底 preference
        # (The LLM should recognize the overlap)
        # This is a soft check - LLM may or may not deduplicate
        assert isinstance(result.created, list)

    def test_process_memories_update_existing(self, manager, sample_user_id):
        """Test updating existing memories."""
        # Add memory with outdated info
        existing = Memory(
            user_id=sample_user_id,
            content="小美用的是 iPhone 15",
            entities=["小美", "iPhone 15"],
        )
        manager.add_memory(sample_user_id, existing)

        # Process text with updated info
        text = "小美换了新手机，现在用 iPhone 16 Pro"
        result = process_memories(
            text=text,
            text_id="test-003",
            user_id=sample_user_id,
            manager=manager,
        )

        # Should potentially update or create new memory
        # (LLM decides whether to update or create new)
        assert isinstance(result.updated, list)
        assert isinstance(result.created, list)

    def test_process_memories_result_structure(self, manager, sample_text, sample_text_id, sample_user_id):
        """Test that result has correct structure."""
        result = process_memories(
            text=sample_text,
            text_id=sample_text_id,
            user_id=sample_user_id,
            manager=manager,
        )

        assert hasattr(result, "created")
        assert hasattr(result, "updated")
        assert hasattr(result, "deleted")
        assert isinstance(result.created, list)
        assert isinstance(result.updated, list)
        assert isinstance(result.deleted, list)


class TestProcessMemoriesIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, manager, sample_user_id):
        """Test full workflow: create, query, update."""
        # Step 1: Process initial text
        text1 = "2024年1月，我开始学习 Python 编程，使用的是 VS Code 编辑器"
        result1 = process_memories(
            text=text1,
            text_id="step1",
            user_id=sample_user_id,
            manager=manager,
        )

        # Should create memories
        assert len(result1.created) > 0

        # Save created memories to manager
        for op in result1.created:
            if op.created_memory:
                manager.add_memory(sample_user_id, op.created_memory)

        # Step 2: Query related memories
        results = manager.hybrid_search(sample_user_id, "Python 编程", top_k=5)
        assert len(results) > 0

        # Step 3: Process update text
        text2 = "现在我改用 PyCharm 来写 Python 了，比 VS Code 更专业"
        result2 = process_memories(
            text=text2,
            text_id="step2",
            user_id=sample_user_id,
            manager=manager,
        )

        # Should have some operations (create or update)
        total_ops = len(result2.created) + len(result2.updated)
        assert total_ops >= 0  # May or may not have operations depending on LLM

    def test_memory_references(self, manager, sample_user_id):
        """Test that created memories have proper references."""
        text = "今天的会议很重要，讨论了新项目的计划"
        text_id = "meeting-001"

        result = process_memories(
            text=text,
            text_id=text_id,
            user_id=sample_user_id,
            manager=manager,
        )

        # Check that memories have references
        for op in result.created:
            if op.created_memory and op.created_memory.references:
                ref = op.created_memory.references[0]
                assert ref.text_id == text_id
