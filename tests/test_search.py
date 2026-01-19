"""检索字符串生成测试（真实 LLM 调用）"""

import pytest
from memory_core.search import generate_search_strings


class TestGenerateSearchStrings:
    """generate_search_strings 测试"""

    def test_basic_generation(self, test_text):
        """基本生成测试"""
        results = generate_search_strings(test_text)

        assert isinstance(results, list)
        assert len(results) > 0

        # 打印结果以便调试
        print(f"\n生成的检索字符串:")
        for i, s in enumerate(results):
            print(f"  {i + 1}. {s}")

    def test_results_are_sentences(self, test_text):
        """结果应该是完整句子，不是单词"""
        results = generate_search_strings(test_text)

        for s in results:
            # 每个结果应该至少有 5 个字符
            assert len(s) >= 5, f"检索字符串太短: {s}"

        print(f"\n生成 {len(results)} 个检索字符串")

    def test_covers_topics(self, test_text):
        """应覆盖主要话题"""
        results = generate_search_strings(test_text)
        combined = " ".join(results).lower()

        # 检查是否覆盖了主要话题
        topics = ["火锅", "海底捞", "iphone", "华强北", "出差", "上海", "python", "fastapi"]
        covered = [t for t in topics if t.lower() in combined]

        print(f"\n覆盖的话题: {covered}")
        print(f"生成的检索字符串: {results}")

        # 应该覆盖大部分话题
        assert len(covered) >= 4, f"期望覆盖至少 4 个话题，实际覆盖: {covered}"

    def test_empty_text(self):
        """空文本"""
        results = generate_search_strings("")
        assert isinstance(results, list)

    def test_short_text(self):
        """短文本"""
        results = generate_search_strings("我今天去北京出差了")
        assert isinstance(results, list)
        print(f"\n短文本检索字符串: {results}")
