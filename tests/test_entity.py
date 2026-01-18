"""
Tests for entity extraction.

Uses real LLM calls as specified in design document.
"""

import pytest

from memory_core.entity import extract_entities


class TestExtractEntities:
    """Tests for extract_entities function."""

    def test_extract_from_long_text(self, sample_text):
        """Test entity extraction from long text."""
        entities = extract_entities(sample_text)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Should extract key entities from the sample text
        entities_lower = [e.lower() for e in entities]
        # Check for some expected entities (case-insensitive)
        expected_entities = ["小美", "深圳", "海底捞", "华强北", "iphone", "上海", "python", "fastapi"]
        found_count = sum(1 for exp in expected_entities if any(exp.lower() in e for e in entities_lower))
        # Should find at least half of expected entities
        assert found_count >= 4, f"Found entities: {entities}"

    def test_extract_from_short_query(self):
        """Test entity extraction from short query."""
        query = "小美喜欢什么口味的火锅？"
        entities = extract_entities(query)

        assert isinstance(entities, list)
        # Should extract key entities from query
        entities_lower = [e.lower() for e in entities]
        assert any("小美" in e for e in entities_lower) or any("火锅" in e for e in entities_lower)

    def test_extract_uniqueness(self):
        """Test that entities are deduplicated."""
        text = "小美和小美一起去了海底捞，海底捞的服务很好"
        entities = extract_entities(text)

        # Check no duplicates
        assert len(entities) == len(set(entities))

    def test_empty_text(self):
        """Test extraction from empty text."""
        entities = extract_entities("")
        assert isinstance(entities, list)
