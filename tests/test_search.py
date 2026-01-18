"""
Tests for search string generation.

Uses real LLM calls as specified in design document.
"""

import pytest

from memory_core.search import generate_search_strings


class TestGenerateSearchStrings:
    """Tests for generate_search_strings function."""

    def test_generate_from_multi_topic_text(self, sample_text):
        """Test generating search strings from multi-topic text."""
        search_strings = generate_search_strings(sample_text)

        assert isinstance(search_strings, list)
        assert len(search_strings) > 0

        # Should generate multiple search strings for multi-topic text
        # The sample text has 4 distinct topics: restaurant, shopping, travel, programming
        assert len(search_strings) >= 3, f"Got only {len(search_strings)} search strings"

    def test_search_strings_are_concise(self, sample_text):
        """Test that search strings are reasonably concise."""
        search_strings = generate_search_strings(sample_text)

        for ss in search_strings:
            # Each search string should be shorter than original text
            assert len(ss) < len(sample_text), f"Search string too long: {ss}"

    def test_coverage(self, sample_text):
        """Test that search strings cover key topics."""
        search_strings = generate_search_strings(sample_text)

        # Combine all search strings
        combined = " ".join(search_strings).lower()

        # Should cover major topics
        topics_found = 0
        if "火锅" in combined or "海底捞" in combined or "番茄" in combined:
            topics_found += 1
        if "iphone" in combined or "华强北" in combined:
            topics_found += 1
        if "上海" in combined or "出差" in combined:
            topics_found += 1
        if "python" in combined or "fastapi" in combined:
            topics_found += 1

        assert topics_found >= 2, f"Only {topics_found} topics covered in: {search_strings}"

    def test_single_topic_text(self):
        """Test generation from single topic text."""
        text = "我喜欢吃川菜，特别是麻婆豆腐和回锅肉"
        search_strings = generate_search_strings(text)

        assert isinstance(search_strings, list)
        assert len(search_strings) >= 1
