"""关键词抽取测试（真实 LLM 调用）"""

import pytest
from memory_core.keyword import extract_keywords


class TestExtractKeywords:
    """extract_keywords 测试"""

    def test_basic_extraction(self, test_text):
        """基本抽取测试"""
        keywords = extract_keywords(test_text)

        assert isinstance(keywords, list)
        assert len(keywords) > 0

        # 打印结果以便调试
        print(f"\n抽取的关键词: {keywords}")

    def test_contains_expected_entities(self, test_text):
        """应包含预期的实体"""
        keywords = extract_keywords(test_text)

        # 转换为小写集合便于比较
        keywords_lower = [k.lower() for k in keywords]
        keywords_str = " ".join(keywords_lower)

        # 至少应该包含一些关键实体
        expected_any = ["小美", "海底捞", "iphone", "深圳", "华强北", "上海", "python", "fastapi", "flask"]
        found = [e for e in expected_any if e.lower() in keywords_str or e in keywords]

        print(f"\n抽取的关键词: {keywords}")
        print(f"匹配到的预期实体: {found}")

        assert len(found) >= 3, f"期望匹配至少 3 个实体，实际匹配: {found}"

    def test_empty_text(self):
        """空文本"""
        keywords = extract_keywords("")
        assert isinstance(keywords, list)

    def test_short_text(self):
        """短文本"""
        keywords = extract_keywords("我喜欢吃苹果")
        assert isinstance(keywords, list)
        print(f"\n短文本关键词: {keywords}")

    def test_returns_unique_keywords(self, test_text):
        """返回去重的关键词"""
        keywords = extract_keywords(test_text)

        # 检查无重复
        assert len(keywords) == len(set(keywords))
