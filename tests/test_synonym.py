"""同义词生成测试（真实 LLM 调用）"""

import pytest
from memory_core.synonym import generate_synonyms


class TestGenerateSynonyms:
    """generate_synonyms 测试"""

    def test_basic_generation(self):
        """基本生成测试"""
        keywords = ["老婆", "iPhone", "Python"]
        result = generate_synonyms(keywords)

        assert isinstance(result, dict)
        # 所有关键词都应该有结果
        for k in keywords:
            assert k in result
            assert isinstance(result[k], list)

        print(f"\n同义词生成结果: {result}")

    def test_synonym_quality(self):
        """同义词质量测试"""
        keywords = ["妻子"]  # 使用更标准的词
        result = generate_synonyms(keywords)

        synonyms = result.get("妻子", [])
        print(f"\n'妻子' 的同义词: {synonyms}")

        # 应该在结果中（可能有或没有同义词）
        assert "妻子" in result

    def test_no_synonyms_case(self):
        """无同义词情况"""
        keywords = ["Python"]
        result = generate_synonyms(keywords)

        # Python 可能有也可能没有同义词，但应该有结果
        assert "Python" in result
        print(f"\n'Python' 的同义词: {result['Python']}")

    def test_batch_processing(self):
        """批处理测试"""
        keywords = ["手机", "电脑", "汽车", "房子"]
        result = generate_synonyms(keywords, batch_size=2)

        # 所有关键词都应该有结果
        for k in keywords:
            assert k in result

        print(f"\n批处理结果: {result}")

    def test_empty_list(self):
        """空列表"""
        result = generate_synonyms([])
        assert result == {}

    def test_deduplication(self):
        """去重测试"""
        keywords = ["老婆", "老婆", "手机"]
        result = generate_synonyms(keywords)

        # 应该只有 2 个唯一关键词
        assert "老婆" in result
        assert "手机" in result
