"""
Tests for data structures.
"""

import json

from memory_core.models import (
    Memory,
    MemoryAction,
    MemoryOperation,
    MemoryOperationResult,
    MemoryReference,
    ScoredMemory,
)


class TestMemoryAction:
    """Tests for MemoryAction enum."""

    def test_action_values(self):
        assert MemoryAction.CREATE == "create"
        assert MemoryAction.UPDATE == "update"
        assert MemoryAction.DELETE == "delete"

    def test_action_is_string(self):
        assert isinstance(MemoryAction.CREATE, str)


class TestMemoryReference:
    """Tests for MemoryReference."""

    def test_creation(self):
        ref = MemoryReference(text_id="text-001", spans=[(0, 10), (20, 30)])
        assert ref.text_id == "text-001"
        assert ref.spans == [(0, 10), (20, 30)]


class TestMemory:
    """Tests for Memory class."""

    def test_minimal_creation(self):
        memory = Memory(user_id="user-001", content="Test content")
        assert memory.user_id == "user-001"
        assert memory.content == "Test content"
        assert memory.id == ""
        assert memory.entities == []
        assert memory.category == ""
        assert memory.occurred_string is None
        assert memory.occurred_at is None

    def test_full_creation(self, sample_memory):
        assert sample_memory.id == "mem-001"
        assert sample_memory.user_id == "test-user-001"
        assert "小美" in sample_memory.entities
        assert sample_memory.category == "美食"

    def test_to_str_for_dense_retrieval(self, sample_memory):
        result = sample_memory.to_str_for_dense_retrieval()
        assert result == sample_memory.content

    def test_to_str_for_update(self, sample_memory):
        result = sample_memory.to_str_for_update()
        data = json.loads(result)
        assert data["content"] == sample_memory.content
        assert data["entities"] == sample_memory.entities
        assert data["category"] == sample_memory.category
        assert data["occurred_string"] == sample_memory.occurred_string
        # Should not include id, created_at, etc.
        assert "id" not in data
        assert "created_at" not in data

    def test_to_str_for_insert(self, sample_memory):
        result = sample_memory.to_str_for_insert()
        data = json.loads(result)
        assert data["content"] == sample_memory.content
        assert data["occurred_string"] == sample_memory.occurred_string
        # Should only include content and occurred_string
        assert "entities" not in data
        assert "category" not in data

    def test_get_entities(self, sample_memory):
        entities = sample_memory.get_entities()
        assert entities == sample_memory.entities
        # Should be a copy, not the same list
        entities.append("new")
        assert "new" not in sample_memory.entities

    def test_to_dict(self, sample_memory):
        data = sample_memory.to_dict()
        assert data["id"] == sample_memory.id
        assert data["user_id"] == sample_memory.user_id
        assert data["content"] == sample_memory.content
        assert data["entities"] == sample_memory.entities
        assert data["category"] == sample_memory.category
        assert len(data["references"]) == 1
        assert data["references"][0]["text_id"] == "text-001"

    def test_from_dict(self, sample_memory):
        data = sample_memory.to_dict()
        restored = Memory.from_dict(data)
        assert restored.id == sample_memory.id
        assert restored.user_id == sample_memory.user_id
        assert restored.content == sample_memory.content
        assert restored.entities == sample_memory.entities
        assert len(restored.references) == 1
        assert restored.references[0].text_id == "text-001"

    def test_roundtrip_serialization(self, sample_memory):
        data = sample_memory.to_dict()
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        restored = Memory.from_dict(loaded)
        assert restored.content == sample_memory.content
        assert restored.entities == sample_memory.entities


class TestScoredMemory:
    """Tests for ScoredMemory."""

    def test_creation(self, sample_memory):
        scored = ScoredMemory(memory=sample_memory, score=0.95)
        assert scored.memory == sample_memory
        assert scored.score == 0.95


class TestMemoryOperation:
    """Tests for MemoryOperation."""

    def test_create_operation(self, sample_memory):
        op = MemoryOperation(
            action=MemoryAction.CREATE,
            created_memory=sample_memory,
        )
        assert op.action == MemoryAction.CREATE
        assert op.memory_id == ""
        assert op.created_memory == sample_memory
        assert op.update_dict is None
        assert op.updated_memory is None

    def test_update_operation(self, sample_memory):
        update_dict = {"content": "Updated content"}
        op = MemoryOperation(
            action=MemoryAction.UPDATE,
            memory_id=sample_memory.id,
            update_dict=update_dict,
            updated_memory=sample_memory,
        )
        assert op.action == MemoryAction.UPDATE
        assert op.memory_id == sample_memory.id
        assert op.update_dict == update_dict

    def test_delete_operation(self, sample_memory):
        op = MemoryOperation(
            action=MemoryAction.DELETE,
            memory_id=sample_memory.id,
        )
        assert op.action == MemoryAction.DELETE
        assert op.memory_id == sample_memory.id


class TestMemoryOperationResult:
    """Tests for MemoryOperationResult."""

    def test_empty_result(self):
        result = MemoryOperationResult(created=[], updated=[], deleted=[])
        assert len(result.created) == 0
        assert len(result.updated) == 0
        assert len(result.deleted) == 0

    def test_with_operations(self, sample_memory):
        create_op = MemoryOperation(
            action=MemoryAction.CREATE,
            created_memory=sample_memory,
        )
        delete_op = MemoryOperation(
            action=MemoryAction.DELETE,
            memory_id="mem-002",
        )
        result = MemoryOperationResult(
            created=[create_op],
            updated=[],
            deleted=[delete_op],
        )
        assert len(result.created) == 1
        assert len(result.deleted) == 1
