"""
Tests for synonym generation.

Uses real LLM calls as specified in design document.
"""

import pytest

from memory_core.synonym import generate_synonyms


class TestGenerateSynonyms:
    """Tests for generate_synonyms function."""

    def test_generate_for_common_entities(self):
        """Test synonym generation for common entities."""
        entities = ["老婆", "手机", "电脑"]
        synonyms = generate_synonyms(entities)

        assert isinstance(synonyms, dict)
        # All input entities should have entries
        for entity in entities:
            assert entity in synonyms

        # "老婆" should have synonyms like 妻子, 太太, etc.
        wife_synonyms = synonyms.get("老婆", [])
        assert isinstance(wife_synonyms, list)

    def test_generate_for_technical_terms(self):
        """Test synonym generation for technical terms."""
        entities = ["Python", "JavaScript", "人工智能"]
        synonyms = generate_synonyms(entities)

        assert isinstance(synonyms, dict)
        for entity in entities:
            assert entity in synonyms

    def test_empty_list(self):
        """Test with empty entity list."""
        synonyms = generate_synonyms([])
        assert synonyms == {}

    def test_all_entities_covered(self):
        """Test that all entities get an entry even without synonyms."""
        entities = ["XYZ123", "特殊名词ABC"]  # Unlikely to have synonyms
        synonyms = generate_synonyms(entities)

        for entity in entities:
            assert entity in synonyms
            assert isinstance(synonyms[entity], list)

    def test_batch_processing(self):
        """Test batch processing with small batch size."""
        entities = ["老婆", "手机", "电脑", "汽车", "飞机"]
        synonyms = generate_synonyms(entities, batch_size=2)

        assert isinstance(synonyms, dict)
        assert len(synonyms) == len(entities)
