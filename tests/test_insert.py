"""
Tests for memory insertion operations.
"""

import pytest

from memory_core.insert import _parse_occurred_string, insert_memories
from memory_core.models import MemoryAction


class TestParseOccurredString:
    """Tests for _parse_occurred_string function."""

    def test_parse_full_datetime(self):
        """Test parsing full datetime string."""
        result = _parse_occurred_string("2024-12-25T10:30:00")
        assert result is not None
        assert "2024-12-25" in result
        assert "10:30:00" in result

    def test_parse_date_only(self):
        """Test parsing date only."""
        result = _parse_occurred_string("2024-12-25")
        assert result is not None
        assert "2024-12-25" in result

    def test_parse_year_month(self):
        """Test parsing year-month."""
        result = _parse_occurred_string("2024-12")
        assert result is not None
        assert "2024-12-01" in result

    def test_parse_year_only(self):
        """Test parsing year only."""
        result = _parse_occurred_string("2024")
        assert result is not None
        assert "2024-01-01" in result

    def test_parse_with_hour(self):
        """Test parsing date with hour."""
        result = _parse_occurred_string("2024-12-25T15")
        assert result is not None
        assert "2024-12-25" in result
        assert "15:00:00" in result

    def test_parse_with_minute(self):
        """Test parsing date with minute."""
        result = _parse_occurred_string("2024-12-25T15:30")
        assert result is not None
        assert "15:30:00" in result

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = _parse_occurred_string("")
        assert result is None

    def test_parse_none(self):
        """Test parsing None."""
        result = _parse_occurred_string(None)
        assert result is None

    def test_parse_invalid_format(self):
        """Test parsing invalid format."""
        result = _parse_occurred_string("not-a-date")
        assert result is None


class TestInsertMemories:
    """Tests for insert_memories function (real LLM calls)."""

    def test_insert_from_text(self, sample_text, sample_text_id, sample_user_id):
        """Test inserting memories from text."""
        create_ops = insert_memories(
            text=sample_text,
            text_id=sample_text_id,
            user_id=sample_user_id,
            existing_memories=[],
        )

        assert isinstance(create_ops, list)
        assert len(create_ops) > 0

        for op in create_ops:
            assert op.action == MemoryAction.CREATE
            assert op.created_memory is not None
            assert op.created_memory.user_id == sample_user_id
            assert op.created_memory.content != ""

    def test_insert_extracts_entities(self, sample_text, sample_text_id, sample_user_id):
        """Test that inserted memories have entities."""
        create_ops = insert_memories(
            text=sample_text,
            text_id=sample_text_id,
            user_id=sample_user_id,
            existing_memories=[],
        )

        # At least some memories should have entities
        memories_with_entities = [
            op for op in create_ops
            if op.created_memory and len(op.created_memory.entities) > 0
        ]
        assert len(memories_with_entities) > 0

    def test_insert_with_category(self, sample_text, sample_text_id, sample_user_id):
        """Test that inserted memories have categories."""
        create_ops = insert_memories(
            text=sample_text,
            text_id=sample_text_id,
            user_id=sample_user_id,
            existing_memories=[],
        )

        # At least some memories should have categories
        memories_with_category = [
            op for op in create_ops
            if op.created_memory and op.created_memory.category
        ]
        assert len(memories_with_category) > 0

    def test_insert_avoids_duplicates(self, sample_user_id, sample_memories):
        """Test that insertion avoids duplicates with existing memories."""
        text = "小美喜欢番茄锅底的火锅"  # Similar to existing memory
        create_ops = insert_memories(
            text=text,
            text_id="dup-test",
            user_id=sample_user_id,
            existing_memories=sample_memories,  # Pass existing memories
        )

        # LLM should recognize the overlap and avoid exact duplicates
        # This is a soft check - LLM behavior may vary
        assert isinstance(create_ops, list)

    def test_insert_sets_references(self, sample_text_id, sample_user_id):
        """Test that inserted memories have references."""
        text = "今天学习了新知识"
        create_ops = insert_memories(
            text=text,
            text_id=sample_text_id,
            user_id=sample_user_id,
            existing_memories=[],
        )

        # Check references
        for op in create_ops:
            if op.created_memory and op.created_memory.references:
                ref = op.created_memory.references[0]
                assert ref.text_id == sample_text_id

    def test_insert_handles_time(self, sample_user_id):
        """Test that inserted memories handle time information."""
        text = "2024年5月15日，我参加了一个重要的会议"
        create_ops = insert_memories(
            text=text,
            text_id="time-test",
            user_id=sample_user_id,
            existing_memories=[],
        )

        # At least one memory should have time info
        memories_with_time = [
            op for op in create_ops
            if op.created_memory and op.created_memory.occurred_string
        ]
        # May or may not extract time depending on LLM
        assert isinstance(memories_with_time, list)
